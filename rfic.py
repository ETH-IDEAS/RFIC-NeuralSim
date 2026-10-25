import numpy as np
from skrf import Frequency, Network
from data_processing.snp_files.dataProcessing import SP2Impedance
import torch
from concurrent.futures import ProcessPoolExecutor
import os


# Worker-local globals for parallel evaluation.
_WORKER_NETWORKS = None


def _init_worker_networks(f_batch_np, SM_batch_np):
    global _WORKER_NETWORKS
    _WORKER_NETWORKS = []
    for b in range(SM_batch_np.shape[0]):
        freq = Frequency.from_f(f_batch_np[b], 'ghz')
        _WORKER_NETWORKS.append(Network(frequency=freq, s=SM_batch_np[b], name=f"network_{b}"))


def _compute_one_ccomb(task):
    flat_idx, b, C1, C2, row_freq_idx = task
    [(_, Zin), (_, passive_efficiency), (_, gmax)], _ = SP2Impedance(_WORKER_NETWORKS[b], None, None, None, C1, C2)
    max_gain = 10 * np.log10(gmax)
    Zin_mag = np.abs(Zin)
    Zin_phase = np.angle(Zin)
    out = np.stack(
        [
            Zin_mag[row_freq_idx],
            Zin_phase[row_freq_idx],
            passive_efficiency[row_freq_idx],
            max_gain[row_freq_idx],
        ],
        axis=-1,
    )
    return flat_idx, out

def reconstruct_s_params_matrix(SM):
    '''
    Take an S-parameter matrix of shape (..., 8, 8) and reconstruct it to have the upper triangular part filled.
    '''
    shape = SM.shape
    assert SM.shape[-1] % 2 == 0 and SM.shape[-2] == SM.shape[-1] and len(shape) >= 3, f"Invalid S-parameter shape {shape}"
    n = SM.shape[-1]  # n = 8

    # Fill odd rows from the even row above. For odd columns, use j-1; for even columns, use j+1 (clamped).
    cols = torch.arange(n, device=SM.device)
    src_cols = torch.where(cols % 2 == 1, cols - 1, cols + 1).clamp(0, n - 1)
    SM[..., 1::2, :] = SM[..., 0::2, :][..., src_cols]

    # Mirror upper triangle into lower triangle.
    upper = torch.triu(SM)
    return upper + upper.transpose(-1, -2) - torch.diag_embed(torch.diagonal(upper, dim1=-2, dim2=-1))

def get_s_params_matrix(s_params, interleaved_mag_phase: bool = True):
    '''
    Take a S-parameter vector of shape (..., 40) and reconstruct it to be the full S-parameter matrix.
    The return shape is (..., 8, 8) where the last two dimensions are the S-parameter matrix. 
    The input S-parameter vector is in the format of (mag, phase) pairs for the upper triangular part of the S-parameter matrix.

    If interleaved_mag_phase is True, the input is [mag0, phase0, mag1, phase1, ...].
    If False, the input is [mag0..mag19, phase0..phase19].
    '''
    shape = s_params.shape
    assert len(shape) >= 2 and shape[-1] == 40, f"Invalid S-parameter shape {shape}"

    if interleaved_mag_phase:
        s_param_mag = s_params[..., 0::2]
        s_param_phase = s_params[..., 1::2]
    else:
        s_param_mag = s_params[..., :20]
        s_param_phase = s_params[..., 20:]

    # complex upper-triangular entries encoded by (magnitude, phase)
    s_param_complex = torch.polar(s_param_mag, s_param_phase)  # (..., 20)
    S = torch.zeros((*shape[:-1], 8, 8), dtype=s_param_complex.dtype, device=s_params.device)
    S[..., 0, :] = s_param_complex[..., :8]
    S[..., 2, 2:] = s_param_complex[..., 8:14]
    S[..., 4, 4:] = s_param_complex[..., 14:18]
    S[..., 6, 6:] = s_param_complex[..., 18:]
    return reconstruct_s_params_matrix(S)



def batch_SM_to_results(SM, spec, f=None, num_workers=None):
    '''
    Convert batched S-parameter matrices to results.

    SM: torch.Tensor of shape (B, FB, 8, 8).
    spec: torch.Tensor of shape (B, C_combs, FB, 3) containing (C1, C2, f) tuples.
    f: optional 1D frequency grid shared by all batch samples. If None, inferred per batch element from spec.
    num_workers: multiprocessing workers used inside each batch element.

    Returns:
        torch.Tensor of shape (B, C_combs, FB, 4).
    '''
    assert SM.dim() == 4 and SM.shape[-2:] == (8, 8), f"Invalid batched S-parameter shape: {SM.shape}"
    assert spec.dim() == 4 and spec.shape[-1] == 3, f"Invalid batched spec shape: {spec.shape}"
    B, FB = SM.shape[0], SM.shape[1]
    C_combs = spec.shape[1]
    assert B == spec.shape[0], f"Batch size mismatch: SM={B}, spec={spec.shape[0]}"
    assert FB == spec.shape[2], f"Frequency axis mismatch: SM={FB}, spec={spec.shape[2]}"

    spec = torch.round(spec, decimals=4)  # round to 4 decimals to avoid floating point issues
    spec_np = spec.detach().cpu().numpy()
    SM_np = SM.detach().cpu().numpy()

    if f is None:
        f_batch_np = spec_np[:, 0, :, 2]
    else:
        f = np.asarray(f)
        if f.ndim == 1:
            f_batch_np = np.repeat(f[None, :], B, axis=0)
        elif f.ndim == 2:
            assert f.shape[0] == B, f"Expected f shape ({B}, FB), got {f.shape}"
            f_batch_np = f
        else:
            raise ValueError(f"Invalid f shape: {f.shape}")

    assert f_batch_np.shape == (B, FB), f"f must have shape ({B}, {FB}), got {f_batch_np.shape}"
    for b in range(B):
        assert np.all(np.diff(f_batch_np[b]) > 0), f"Frequencies must be ascending and unique at batch index {b}."

    tasks = []
    flat_idx = 0
    for b in range(B):
        f_b = f_batch_np[b]
        for c in range(C_combs):
            C1, C2 = float(spec_np[b, c, 0, 0]), float(spec_np[b, c, 0, 1])
            # ensure C1/C2 are constant across the frequency axis for each row
            assert np.allclose(spec_np[b, c, :, 0], C1), f"C1 varies at batch {b}, C_comb {c}."
            assert np.allclose(spec_np[b, c, :, 1], C2), f"C2 varies at batch {b}, C_comb {c}."
            row_freq_idx = np.searchsorted(f_b, spec_np[b, c, :, 2])
            tasks.append((flat_idx, b, C1, C2, row_freq_idx))
            flat_idx += 1

    if num_workers is None:
        num_workers = min(len(tasks), os.cpu_count() or 1)

    if num_workers <= 1:
        _init_worker_networks(f_batch_np, SM_np)
        out_items = [_compute_one_ccomb(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker_networks,
            initargs=(f_batch_np, SM_np),
        ) as ex:
            out_items = list(ex.map(_compute_one_ccomb, tasks))

    out_items.sort(key=lambda x: x[0])
    flat_results = np.stack([x[1] for x in out_items], axis=0)  # (B*C_combs, FB, 4)
    results = flat_results.reshape(B, C_combs, FB, 4)
    return torch.from_numpy(results).float()


