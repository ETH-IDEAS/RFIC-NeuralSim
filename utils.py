import torch
import os, json
from tqdm.auto import tqdm
from typing import Optional
import random
import numpy as np
from typing import Tuple, Generic, TypeVar
T = TypeVar('T', float, torch.Tensor, np.ndarray)






def seed_everything(seed:int, rank:int=0, strict=False):
    random.seed(seed+rank)
    np.random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.cuda.manual_seed_all(seed+rank)
    if strict:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Statistics(Generic[T]):
    '''
    If momentum = None, calculate the average of all the values.
    Else if momentum in (0, 1), calculate the exponential moving average.
    '''
    def __init__(self, momentum:Optional[float]=None):
        self.n:int = 0
        self._avg:Optional[T] = None
        self._last:Optional[T] = None
        self.momentum = momentum

    @property
    def avg(self) -> T:
        if self._avg is None:
            raise ValueError("Average has not been computed yet.")
        return self._avg
    
    @property
    def last(self) -> T:
        if self._last is None:
            raise ValueError("No last value recorded yet.")
        return self._last
    
    @torch.no_grad()
    def update(self, x:T, num:int=1):
        if self._avg is None:
            self._avg = x
        else:
            delta = x - self.avg
            if self.momentum is None:
                self._avg += delta * (num / (self.n + num))
            else:
                self._avg += delta * (1 - self.momentum)
        self.n += num
        self._last = x

    @classmethod
    def get_statistics(cls: type["Statistics[T]"], k: int, **kwargs) -> list["Statistics[T]"]:
        return [cls(**kwargs) for _ in range(k)]

from tqdm.auto import tqdm
def make_pbar(iterable, desc):
    return tqdm(
        iterable,
        desc=desc,
        disable=not is_main_process(),
        dynamic_ncols=True,   # adapt to terminal width
        leave=False,
        smoothing=0.1,
    )

    
def write_perf_to_json(perf_dict, save_root, filename:str="log.json"):
    filepath = os.path.join(save_root, filename)
    with open(filepath, "w") as f:
        json.dump(perf_dict, f, indent=4)




    
def conv_shape(height:int, width:int, kernel_size:int|Tuple[int, int], stride:int|Tuple[int, int], padding:int|Tuple[int, int]):
    """
    Compute output height and width after a Conv2d with given parameters.
    Assumes dilation = 1.

    Parameters
    ----------
    height : int
        Input height (H).
    width : int
        Input width (W).
    kernel_size : int or tuple
        Kernel size. If int, same for H and W.
    stride : int or tuple
        Stride. If int, same for H and W.
    padding : int or tuple
        Padding. If int, same for H and W.

    Returns
    -------
    (out_height, out_width) : tuple of ints
        The resulting spatial dimensions after convolution.
    """
    # unpack kernel_size
    if isinstance(kernel_size, int):
        k_h = k_w = kernel_size
    else:
        k_h, k_w = kernel_size

    # unpack stride
    if isinstance(stride, int):
        s_h = s_w = stride
    else:
        s_h, s_w = stride

    # unpack padding
    if isinstance(padding, int):
        p_h = p_w = padding
    else:
        p_h, p_w = padding

    out_height = (height + 2 * p_h - k_h) // s_h + 1
    out_width  = (width  + 2 * p_w - k_w) // s_w + 1
    return out_height, out_width



# ----------- distributed training utils --------------

import torch.distributed as dist

def init_distributed():
    """Return (is_dist, local_rank, global_rank, world_size)."""
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        return True, local_rank, dist.get_rank(), dist.get_world_size()
    else:
        # single process, still pick a device
        return False, 0, 0, 1
    
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def is_using_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()

def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        try:
            dist.barrier()          # optional but helps ensure all ranks finished
        except Exception:
            pass
        dist.destroy_process_group()


@torch.no_grad()
def reduce_mean(value:torch.Tensor, count:int=1) -> Tuple[torch.Tensor, int]:
    """Sample-weighted mean across ranks. Safe if batches differ in size."""
    if not dist.is_initialized():
        return value, count
    dev = value.device
    vsum = value.detach() * count
    cnt  = torch.tensor(count, device=dev)
    dist.all_reduce(vsum, op=dist.ReduceOp.SUM)
    dist.all_reduce(cnt,  op=dist.ReduceOp.SUM)
    return (vsum / (cnt.clamp_min(1e-12))).to(value.dtype), int(cnt.item())


