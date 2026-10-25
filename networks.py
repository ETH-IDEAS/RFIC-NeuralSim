import torch
from torch import nn
from typing import Callable, Optional, Tuple
from utils import conv_shape
from torchvision.models.resnet import BasicBlock, conv1x1
import math

class Modulate(nn.Module):
    '''
    This module has no learnable parameters. It applies modulation to the input tensor x as:
        out = x * weight + bias
    where weight and bias are set externally via the set_modulate_parameters method.

    It works for any input shape, as long as the weight and bias are of compatible shape.
    '''
    def __init__(self, *shape):
        super(Modulate, self).__init__()
        self.weight:Optional[torch.Tensor] = None
        self.bias:Optional[torch.Tensor] = None
        self.shape = shape

    def set_modulate_parameters(self, gamma:torch.Tensor, beta:torch.Tensor):
        self.weight = gamma
        self.bias = beta

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert self.weight is not None and self.bias is not None, "Modulation parameters not set."
        out = x * self.weight + self.bias
        return out

def modulate_net(net, modulation_params:torch.Tensor):
    '''
    modulation_params: (FB, modulate_size)
    Sets the modulation parameters for the Simulator based on the modulation_params.
    '''
    # extract modulation parameters for each Modulate layer
    idx = 0
    for module in net.modules():
        if isinstance(module, Modulate):
            param_num = math.prod(module.shape)  # number of parameters for this Modulate layer
            gamma = modulation_params[..., idx:idx+param_num].view((-1, *module.shape))
            beta = modulation_params[..., idx+param_num:idx+2*param_num].view((-1, *module.shape))
            module.set_modulate_parameters(gamma, beta)
            idx += 2 * param_num
    assert idx == modulation_params.shape[-1], "Not all modulation parameters were used."

def modulation_size(net:nn.Module) -> int:
    """Return the number of gamma/beta values required by all modulation sites."""
    return sum(
        2 * math.prod(module.shape)
        for module in net.modules()
        if isinstance(module, Modulate)
    )

class Reshape(nn.Module):
    '''
    This module has no learnable parameters. It reshapes the input tensor x to the specified shape.
    '''
    def __init__(self, *shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x.view(-1, *self.shape)
    
def FourierEncoding(freq:torch.Tensor, half_dim:int=10) -> torch.Tensor:
    '''
    Each element f is expanded to a vector of shape (2*half_dim+1,) as:
    [f, sin(2^i *pi*f), cos(2^i * pi * f)] for i in [0, half_dim-1]

    Input:
        freq: tensor of shape (..., )
        half_dim: int, the half dimension of the encoding
    Output:
        encoded_freq: tensor of shape (..., 2*half_dim)
    '''
    # Create frequency scales: [2^0, 2^1, ..., 2^(half_dim-1)]
    i = torch.arange(half_dim, device=freq.device, dtype=freq.dtype)
    freq_scales = (2 ** i) * math.pi  # shape: (half_dim,)
    phases = freq.unsqueeze(-1) * freq_scales  # shape: (..., half_dim)
    # Compute sin and cos embeddings
    sin_embed = torch.sin(phases)
    cos_embed = torch.cos(phases)
    # Concatenate along the last dimension
    encoded_freq = torch.cat((freq.unsqueeze(-1), sin_embed, cos_embed), dim=-1)  # shape: (..., 2*half_dim+1)
    return encoded_freq

SimulatorBuilder = Callable[[Tuple, int, int, int], nn.Sequential]


def build_res_simulator(
    input_shape:Tuple,
    num_FB:int,
    base_width:int,
    output_dim:int,
) -> nn.Sequential:
    """Build the original residual Simulator."""
    spatial_dim1 = conv_shape(*input_shape[1:], kernel_size=3, stride=2, padding=1)
    spatial_dim2 = conv_shape(*spatial_dim1, kernel_size=3, stride=2, padding=1)
    return nn.Sequential(
        nn.Conv2d(input_shape[0], base_width, 3, stride=1, padding=1),
        nn.LayerNorm([base_width, *input_shape[1:]]),
        nn.ReLU(inplace=True),
        BasicBlock(base_width, base_width*2, stride=2, downsample=nn.Sequential(
            conv1x1(base_width, base_width*2, stride=2),
            nn.LayerNorm([base_width*2, *spatial_dim1]),
        ), norm_layer=lambda x: nn.LayerNorm([x, *spatial_dim1])),
        Reshape(1, base_width*2, *spatial_dim1),
        Modulate(base_width*2, *spatial_dim1),
        Reshape(base_width*2, *spatial_dim1),
        BasicBlock(base_width*2, base_width*2, stride=1, norm_layer=lambda x: nn.LayerNorm([x, *spatial_dim1])),
        Reshape(num_FB, base_width*2, *spatial_dim1),
        Modulate(base_width*2, *spatial_dim1),
        Reshape(base_width*2, *spatial_dim1),
        BasicBlock(base_width*2, base_width*4, stride=2, downsample=nn.Sequential(
            conv1x1(base_width*2, base_width*4, stride=2),
            nn.LayerNorm([base_width*4, *spatial_dim2]),
        ), norm_layer=lambda x: nn.LayerNorm([x, *spatial_dim2])),
        Reshape(num_FB, base_width*4, *spatial_dim2),
        Modulate(base_width*4, *spatial_dim2),
        Reshape(base_width*4, *spatial_dim2),
        BasicBlock(base_width*4, base_width*4, norm_layer=lambda x: nn.LayerNorm([x, *spatial_dim2])),
        Reshape(num_FB, base_width*4, *spatial_dim2),
        Modulate(base_width*4, *spatial_dim2),
        Reshape(base_width*4*math.prod(spatial_dim2)),
        nn.Linear(base_width*4*math.prod(spatial_dim2), output_dim),
    )


def build_cnn_simulator(
    input_shape:Tuple,
    num_FB:int,
    base_width:int,
    output_dim:int,
) -> nn.Sequential:
    """Build the parameter-matched, residual-free CNN Simulator."""
    spatial_dim1 = conv_shape(*input_shape[1:], kernel_size=3, stride=2, padding=1)
    spatial_dim2 = conv_shape(*spatial_dim1, kernel_size=5, stride=2, padding=2)
    return nn.Sequential(
        nn.Conv2d(input_shape[0], base_width, 3, stride=1, padding=1),
        nn.LayerNorm([base_width, *input_shape[1:]]),
        nn.ReLU(inplace=True),
        nn.Conv2d(base_width, base_width*2, 3, stride=2, padding=1),
        nn.LayerNorm([base_width*2, *spatial_dim1]),
        nn.ReLU(inplace=True),
        Reshape(1, base_width*2, *spatial_dim1),
        Modulate(base_width*2, *spatial_dim1),
        Reshape(base_width*2, *spatial_dim1),
        nn.Conv2d(base_width*2, base_width*2, 3, stride=1, padding=1),
        nn.LayerNorm([base_width*2, *spatial_dim1]),
        nn.ReLU(inplace=True),
        Reshape(num_FB, base_width*2, *spatial_dim1),
        Modulate(base_width*2, *spatial_dim1),
        Reshape(base_width*2, *spatial_dim1),
        nn.Conv2d(base_width*2, base_width*4, 5, stride=2, padding=2),
        nn.LayerNorm([base_width*4, *spatial_dim2]),
        nn.ReLU(inplace=True),
        Reshape(num_FB, base_width*4, *spatial_dim2),
        Modulate(base_width*4, *spatial_dim2),
        Reshape(base_width*4, *spatial_dim2),
        nn.Conv2d(base_width*4, base_width*4, 5, stride=1, padding=2),
        nn.LayerNorm([base_width*4, *spatial_dim2]),
        nn.ReLU(inplace=True),
        Reshape(num_FB, base_width*4, *spatial_dim2),
        Modulate(base_width*4, *spatial_dim2),
        Reshape(base_width*4*math.prod(spatial_dim2)),
        nn.Linear(base_width*4*math.prod(spatial_dim2), output_dim),
    )

class ForwardModel(nn.Module):
    '''
    Main residual forward-model pipeline with complementary response.

    Params:
        num_FB: length of the frequency band (for unfiltered data, num_FB=501)
        sparam_size: size of the output s-parameter vector (default 40)

    Input:
        Mask: Tensor of shape (2, 18, 18)
        C1, C2, Frequency: Floats
    Output:
        Zin_mag, Zin_phase, PassiveEfficiency, MaxGain: Floats
        s_params: Tensor of shape (40, )

    Remarks:
        Since Reshape and Modulate layers do not have learnable parameters, the state_dict may be loaded from a model with different FB settings.
    '''
    def __init__(self, sparam_size:int=40, input_shape:Tuple=(2,18,18), num_FB:int=501, base_width:int=32, freq_encoding_half_dim:int=10, cap_encoding_half_dim:int=5, results_stat:Optional[torch.Tensor]=None, sparams_stat:Optional[torch.Tensor]=None, _simulator_builder:SimulatorBuilder=build_res_simulator):
        super(ForwardModel, self).__init__()
        self.sparam_size:int = sparam_size
        self.num_FB = num_FB
        self.input_shape:Tuple = input_shape
        self.freq_encoding_half_dim:int = freq_encoding_half_dim
        self.cap_encoding_half_dim:int = cap_encoding_half_dim
        self.register_buffer("results_stat", results_stat if results_stat is not None else torch.empty(0))
        self.register_buffer("sparams_stat", sparams_stat if sparams_stat is not None else torch.empty(0))
        zcomp_dim = base_width
        self.zcomp_dim:int = zcomp_dim

        self.Simulator = _simulator_builder(
            input_shape,
            num_FB,
            base_width,
            sparam_size+zcomp_dim,
        )

        # the conditioning network for frequency on Simulator modulation
        modulate_size = modulation_size(self.Simulator)
        self.Conditioner = nn.Sequential(
            nn.Linear(2*freq_encoding_half_dim+1, 4*base_width),
            nn.LayerNorm(4*base_width),
            nn.SiLU(inplace=True),
            nn.Linear(4*base_width, 4*base_width),
            nn.LayerNorm(4*base_width),
            nn.SiLU(inplace=True),
            nn.Linear(4*base_width, modulate_size)
        )

        self.Solver = nn.Sequential(
            nn.Linear(sparam_size+zcomp_dim+2*(2*cap_encoding_half_dim+1), base_width*4),
            nn.LayerNorm(base_width*4),
            nn.ReLU(inplace=True),
            nn.Linear(base_width*4, base_width*4),
            nn.LayerNorm(base_width*4),
            nn.ReLU(inplace=True),
            nn.Linear(base_width*4, base_width*4),
            nn.LayerNorm(base_width*4),
            nn.ReLU(inplace=True),
            nn.Linear(base_width*4, 4)
        )

    def _compute_dependency_penalty(self, pre_linear_feat:torch.Tensor) -> torch.Tensor:
        '''
        Penalize cross-covariance between the S-parameter and Zcomp heads:
            Cov(s, r) where
            s = W1 A + b1, r = W2 A + b2, A = pre_linear_feat.
            pre_linear_feat: (N*FB, D), where D is the input dim of Simulator's last linear layer.

        Uses Cov(s, r) = W1 Cov(A) W2^T, with W1 and Cov(A) detached so
        gradients flow only to W2 (Zcomp head).
        '''
        if self.zcomp_dim == 0:
            return torch.tensor(0.0, device=pre_linear_feat.device, dtype=pre_linear_feat.dtype)
        if pre_linear_feat.shape[0] < 2:
            return torch.tensor(0.0, device=pre_linear_feat.device, dtype=pre_linear_feat.dtype)

        out_layer = self.Simulator[-1]
        assert isinstance(out_layer, nn.Linear), "Simulator last layer must be nn.Linear."
        weight = out_layer.weight  # (sparam_size + zcomp_dim, in_dim)
        W1 = weight[:self.sparam_size].detach()  # sparam head (detached)
        W2 = weight[self.sparam_size:]  # Zcomp head (trainable by this loss)

        # Build Cov(A) from detached features so this branch does not track autograd.
        feat = pre_linear_feat.detach()
        feat_centered = feat - feat.mean(dim=0, keepdim=True)
        cov_A = (feat_centered.T @ feat_centered) / (feat_centered.shape[0] - 1)  # (in_dim, in_dim)

        # Cov(s, r) = W1 Cov(A) W2^T
        cov_sr = W1 @ cov_A @ W2.T  # (sparam_size, zcomp_dim)
        return cov_sr.square().mean()

    def forward(self, mask:torch.Tensor, spec:torch.Tensor, fb:torch.Tensor, return_dependency_penalty:bool=False):
        '''
        mask: (N, 2, 18, 18), i.e., N masks with 2 channels and 18x18 size;
        spec: (N, C, 2), e.g., (BS, 100, 2); 
        fb: (FB, ), the frequency band.

        Returns:
        sparam: (N, FB, sparam_size)
        out: (N, C, FB, 4), i.e., (N masks, C (C1, C2), FB frequencies), and 4 performance metrics (Zin_mag, Zin_phase, PassiveEfficiency, MaxGain).
        '''
        # compute modulation parameters from frequency
        freq_encoding = FourierEncoding(fb, half_dim=self.freq_encoding_half_dim) # (FB, 2*half_dim+1)
        modulation_params = self.Conditioner(freq_encoding) # (FB, modulate_size)
        modulate_net(self.Simulator, modulation_params)
        # get the spatial representation from mask
        simulator_feat = self.Simulator[:-1](mask) # (N*FB, D), D=Simulator[-1].in_features
        mask_freq_repr = self.Simulator[-1](simulator_feat) # (N * FB, embed_dim)
        dependency_penalty = self._compute_dependency_penalty(simulator_feat) if return_dependency_penalty else None
        mask_freq_repr = mask_freq_repr.view(len(mask), len(fb), -1) # (N, FB, embed_dim)
        # extract sparam
        sparam = mask_freq_repr[..., :self.sparam_size] # (N, FB, sparam_size)

        # compute modulation parameters from C1, C2
        Cond = FourierEncoding(spec[..., :2], self.cap_encoding_half_dim) # (N, C, 2, 2*cap_encoding_half_dim+1)
        Cond = Cond.view(Cond.shape[0], Cond.shape[1], -1) # (N, C, 2*(2*cap_encoding_half_dim+1))
        solver_input = torch.cat([mask_freq_repr.unsqueeze(1).expand(-1, Cond.shape[1], -1, -1), Cond.unsqueeze(2).expand(-1, -1, len(fb), -1)], dim=-1) # (N, C, FB, embed_dim+2*(2*cap_encoding_half_dim+1))
        out = self.Solver(solver_input) # (N, C, FB, 4)

        if return_dependency_penalty:
            return [sparam, out], dependency_penalty
        return [sparam, out]


class ResForwardModel(ForwardModel):
    """Forward model using the original residual Simulator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _simulator_builder=build_res_simulator, **kwargs)


class CNNForwardModel(ForwardModel):
    """Forward model using the parameter-matched, residual-free CNN Simulator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _simulator_builder=build_cnn_simulator, **kwargs)
