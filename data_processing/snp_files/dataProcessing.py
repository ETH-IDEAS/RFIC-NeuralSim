import skrf as rf
import numpy as np
from enum import Enum
# in this updated code, the IMN and OMN are treated in a differet, while the ISM is another way, generating 3 different performance series.
# Enum for the SNP Type
class NetworkType(Enum):
    IMN = 1
    ISM = 2
    OMN = 3

# Enum for PowerGain
class GainType(Enum):
    Gt = 0 # Transducer Gain
    Ga = 1 # Available Gain
    Gp = 2 # Operating Gain
    Gmax = 3 # Maximum Available Gain

def gainCalc(s2p:rf.Network, Zload=50,type=GainType.Gp):
    """
    Calculate the gain of a two-port network
    """
    if type == GainType.Gmax:
        return s2p.max_gain
    
    # Calculate the power gain
    s11 = s2p.s[:,0,0]
    s12 = s2p.s[:,0,1]
    s21 = s2p.s[:,1,0]
    s22 = s2p.s[:,1,1]

    # Calculate reflection coefficients
    z0_p1 = s2p.z0[0][0]
    z0_p2 = s2p.z0[0][1] # Use port 2 impedance as reference

    gamma_s = 0 # Assume input is matched
    gamma_l = (Zload - z0_p2)/(Zload + z0_p2)

    gamma_in = s11 + s12*s21*gamma_l/(1 - s22*gamma_l)
    gamma_out = s22 + s12*s21*gamma_s/(1 - s11*gamma_s)

    if type == GainType.Gt:
        gain = s21
    elif type == GainType.Ga:
        gain = (1 - np.abs(gamma_s)**2)*(np.abs(s21)**2)/((np.abs(1 - s11*gamma_s)**2)*(1-np.abs(gamma_out)**2))
    elif type == GainType.Gp:
        gain = (np.abs(s21)**2)*(1-np.abs(gamma_l)**2)/((1-np.abs(gamma_in)**2)*(np.abs(1 - s22*gamma_l)**2))
    else:
        raise ValueError(f"Invalid GainType {type}")

    return gain


"""
SP2Impedance Configurations
---
4, 5 are the DC branches of 0, 1
6, 7 are the DC branches of 2, 3
Achtung: not consistent with real layout
---
IMN
      G  O.C.
     +4--6+
     0    2
  P1 |    | P2
  C1 |    | C2
     1    3--GND
     +5--7+
      G
---
OMN
    O.C.  G
     +4--6+
     0    2
  P2 |    | P1
  C2 |    | C1
GND--1    3
     +5--7+
    O.C.  G
---
ISM
      G  G
     +4--6+
     0    2
  P1 |    | P2
  C1 |    | C2
     1    3
     +5--7+
      G  G
"""

def SP2Impedance(snp_file, freq_start_ghz, freq_end_ghz, numSteps, C1_fF, C2_fF, network_type = NetworkType.IMN, snp_file_core = None):
    """
    Given sparameter file, apply correct terminations and differential-to-single-ended conversion according to network type
    Output the skrf network object with input impedance and passive efficiency when loaded with ideal tuning capacitors

    Returns: Tuple of performance numbers vs frequency: input impedance, passive efficiency, maximum available gain and the skrf network object
    """
    if not (freq_start_ghz is None and freq_end_ghz is None and numSteps is None):
        # Read the SNP file
        if isinstance(snp_file, str):
            snp = rf.Network(snp_file)
        else:
            snp = snp_file
        # Define the frequency range
        snp = snp[f'{freq_start_ghz}GHz-{freq_end_ghz}GHz']
        snp.resample(numSteps)
    else:
        snp = snp_file

    # Read the core SNP file for ISM
    if network_type == NetworkType.ISM and snp_file_core is not None:
        snp_core = rf.Network(snp_file_core)
        snp_core.resample(snp.frequency)

    # Create short and open networks
    # short = rf.Network(frequency=snp.frequency, s=-1*np.ones((snp.frequency.npoints, 1, 1)))
    # open = rf.Network(frequency=snp.frequency, s=np.ones((snp.frequency.npoints, 1, 1)))
    freq = snp.frequency
    short = rf.Circuit.Ground(frequency=freq, name='gnd')
    open = rf.Circuit.Open(frequency=freq, name='open') 
    open2 = rf.Circuit.Open(frequency=freq, name='open2')

    # Tuning Caps
    lumped_media = rf.media.DefinedGammaZ0(frequency=freq)
    C1 = lumped_media.capacitor(C1_fF*1e-15,name="C1")
    C2 = lumped_media.capacitor(C2_fF*1e-15,name="C2")

    P1 = rf.Circuit.Port(freq, name='P1',z0=50)
    P2 = rf.Circuit.Port(freq, name='P2',z0=50)
    P3 = rf.Circuit.Port(freq, name='P3',z0=50)

    base_connections = [
        [(P1,0),(snp,0),(C1,0)],
        [(P2,0),(snp,1),(C1,1)],
        [(P3,0),(snp,2),(C2,0)],
    ]

    # Terminate SnP impedance
    if network_type == NetworkType.IMN:
        terminations = [
            [(snp,3),(snp,4),(snp,5),(short,0),(C2,1)], # DC Bias port to ground
            [(snp,6),(open,0)], # Cap ports floating
            [(snp,7),(open2,0)] # Cap ports floating
        ]
    elif network_type == NetworkType.OMN:
        base_connections = [ # Reset base connections
            [(P1,0),(snp,2),(C1,0)],
            [(P2,0),(snp,3),(C1,1)],
            [(P3,0),(snp,0),(C2,0)],
        ]
        terminations = [
            [(snp,1),(snp,6),(snp,7),(short,0),(C2,1)], # DC Bias port to ground
            [(snp,4),(open,0)], # Cap ports floating
            [(snp,5),(open2,0)]  # Cap ports floating
        ]
    elif network_type == NetworkType.ISM:
        P4 = rf.Circuit.Port(freq, name='P4')
        terminations = [
            [(P4,0),(snp,3),(C2,1)],
            [(snp,4),(snp,5),(snp,6),(snp,7),(short,0)] # DC Bias port to ground
        ]
    
    connections = base_connections + terminations
    reducedSnP = rf.Circuit(connections)

    # Apply correct terminations to power core
    if snp_file_core is not None:
        P1_core = rf.Circuit.Port(freq, name='port1')
        P2_core = rf.Circuit.Port(freq, name='port2')
        ground = rf.Circuit.Ground(frequency=freq, name='gnd1')
        powerCore_connections = [
            [(P1_core,0),(snp_core,0)],
            [(P2_core,0),(snp_core,1)],
            [(ground,0),(snp_core,2),(snp_core,3)]
        ]
        snp_core_reduced = rf.Circuit(powerCore_connections).network
        snp_core_reduced.se2gmm(p=1) # P1 and P2 differential

    # Convert to two port
    reducedNetwork = reducedSnP.network
    if network_type == NetworkType.IMN or network_type == NetworkType.OMN:
        reducedNetwork.se2gmm(p=1) # P1 and P2 differential, P3 single ended
        # Extract Diff to SE two-port
        s11 = reducedNetwork.s[:,0,0]
        s12 = reducedNetwork.s[:,0,2]
        s21 = reducedNetwork.s[:,2,0]
        s22 = reducedNetwork.s[:,2,2]
        row1 = np.expand_dims(np.column_stack((s11,s12)),axis=2)
        row2 = np.expand_dims(np.column_stack((s21,s22)),axis=2)
        spdata = np.concatenate((row1,row2),axis=2)
        zport = [reducedNetwork.z0[0][0], reducedNetwork.z0[0][2]]
        
    elif network_type == NetworkType.ISM:
        reducedNetwork.se2gmm(p=2) # P1 and P2 differential, P3 and P4 differential
        # Extract Diff to Diff two-port
        spdata = reducedNetwork.s[:,0:2,0:2]
        zport = reducedNetwork.z0[0][0:2]
    


    # Calculate differential input impedance
    S11dd =reducedNetwork.s[:,0,0]
    z0diff = reducedNetwork.z0[0][0]
    Zin = z0diff*(1+S11dd)/(1-S11dd)


    #Equivalent Two-port differential
    eq2port = rf.Network(frequency=freq, s=spdata, z0=zport)

    if network_type == NetworkType.ISM and snp_file_core is not None:
    # Calculate the equivalent two-port differential impedance with correct loading
        snp_core_diff = rf.Network(frequency=freq,s=snp_core_reduced.s[:,0,0],z0=snp_core_reduced.z0[0][0])
        loaded = eq2port ** snp_core_diff
        S11dd = loaded.s[:,0,0]
        z0diff = loaded.z0[0][0]
        Zin = z0diff*(1+S11dd)/(1-S11dd)
    
    gmax = gainCalc(eq2port, 50, GainType.Gmax)
    if network_type == NetworkType.IMN or network_type == NetworkType.OMN:
        operating_gain = gainCalc(eq2port, 50, GainType.Gp)
    elif network_type == NetworkType.ISM:
        if snp_file_core is not None:
            s11diff = snp_core_reduced.s[:,0,0]
            z0diff = reducedNetwork.z0[0][0]
            Zin_powerCore = z0diff*(1+s11diff)/(1-s11diff)
        else:
            Zin_powerCore = 50
        operating_gain = gainCalc(eq2port, Zin_powerCore, GainType.Gp)

    passive_efficiency = operating_gain*100
    f_ghz = freq.f/1e9

    return ([(f_ghz,Zin),(f_ghz,passive_efficiency), (f_ghz,gmax)], reducedNetwork)