"""
PyTorch dataloading for a Zarr dataset with arrays:
- results:  float32, shape (N, 200, 3, 501, 7) -> (N, 200, 501, 7)
- s_params: float32, shape (N, 501, 20, 2)     -> (N, 501, 40)
- masks:    bool,    shape (N, 2, 18, 18)

The Zarr root directory also contains:
- train_idx.npy, val_idx.npy, test_idx.npy
- results_mean_std.npy, sparams_mean_std.npy  # [mean, std] pairs

"""
import numpy as np
import os
import torch
import zarr
from utils import is_using_ddp
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Tuple, Sequence
import random

class ZarrTripletDataset(Dataset):
    """
    Map-style dataset that returns (results, s_params, masks) for given indices.

    Notes
    -----
    - Uses lazy open so that each DataLoader worker opens its own Zarr handles.
    - normalization using mean/std (.npy files).
    - `indices` selects a subset (e.g., train/val/test splits).
    - sim_type selects which simulation type to return from results. 0: IMN, 1: OMN, or 2: ISM.
    - The last row of the mask is always 0; can be stripped if desired.
    """
    def __init__(
        self,
        zarr_dir: str,
        sim_type: int = 0,
        indices: Optional[Sequence[int]] = None,
        normalize_results: bool = False,
        normalize_sparams: bool = False,
        results_stats_path: Optional[str] = None,
        sparams_stats_path: Optional[str] = None,
        transform=None
    ):
        assert sim_type in (0, 1, 2), "sim_type must be 0, 1, or 2"
        self.sim_type = sim_type
        self.zarr_dir = zarr_dir
        self.indices = np.asarray(indices) if indices is not None else None
        self.transform = transform

        # Normalize flags and stats
        self.normalize_results = normalize_results
        self.normalize_sparams = normalize_sparams

        self.results_mean = None
        self.results_std = None
        self.sparams_mean = None
        self.sparams_std = None

        if self.normalize_results:
            assert results_stats_path and os.path.exists(results_stats_path), "Results stats file required for normalization."
            stats = np.load(results_stats_path, allow_pickle=False)
            self.results_mean = torch.from_numpy(np.asarray(stats[0, sim_type], dtype=np.float32))
            self.results_std  = torch.from_numpy(np.asarray(stats[1, sim_type], dtype=np.float32))

        if self.normalize_sparams:
            assert sparams_stats_path and os.path.exists(sparams_stats_path), "S-params stats file required for normalization."
            stats = np.load(sparams_stats_path, allow_pickle=False)
            self.sparams_mean = torch.from_numpy(np.asarray(stats[0], dtype=np.float32))
            self.sparams_std  = torch.from_numpy(np.asarray(stats[1], dtype=np.float32))

        # Light-weight open to fetch length (no heavy IO)
        root = zarr.open(self.zarr_dir, mode="r")
        total_len = root["masks"].shape[0]
        self._length = int(total_len) if self.indices is None else int(len(self.indices))

        # Defer heavy Zarr handles to workers
        self._root = None
        self._results = None
        self._sparams = None
        self._masks = None

    def _lazy_open(self):
        if self._root is None:
            self._root = zarr.open(self.zarr_dir, mode="r")
            self._masks   = self._root["masks"]     # bool
            self._results = self._root["results"]   # float32
            self._sparams = self._root["s_params"]  # float32 (..., 2)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._lazy_open()
        idx = int(self.indices[i]) if self.indices is not None else i

        # NumPy arrays decoded on demand, chunk-wise
        m = self._masks[idx]     # shape (2, 18, 18),      bool
        m = torch.from_numpy(np.asarray(m, dtype=np.float32))

        r = self._results[idx][:, self.sim_type]   # shape (200, 501, 7), float32
        s = self._sparams[idx]   # shape (501, 20, 2),     float32

        # Convert to tensors
        r = torch.from_numpy(np.asarray(r, dtype=np.float32))
        s = torch.from_numpy(np.asarray(s, dtype=np.float32))

        # normalization (broadcast over leading dims as needed)
        if self.normalize_results and (self.results_mean is not None) and (self.results_std is not None):
            # shapes must be broadcastable to r; typical stats are per-channel/axis
            view_shape = [1] * (r.dim()-1) + [-1]
            r = (r - self.results_mean.view(view_shape)) / (self.results_std.view(view_shape) + 1e-8)

        if self.normalize_sparams and (self.sparams_mean is not None) and (self.sparams_std is not None):
            view_shape = [1] * (s.dim()-1) + [-1]
            s = (s - self.sparams_mean.view(view_shape)) / (self.sparams_std.view(view_shape) + 1e-8)

        s = s.view(*s.shape[:-2], -1)  # flatten last dim
        
        if self.transform:
            r, s, m = self.transform(r, s, m)

        return r, s, m

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, r, s, m):
        for t in self.transforms:
            r, s, m = t(r, s, m)
        return r, s, m
    
class ChooseOne:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, r, s, m):
        t = random.choice(self.transforms)
        r, s, m = t(r, s, m)
        return r, s, m
    
# randomly move the full mask down by 1 pixel
# this is valid because the last row is always zero
class RandomDownShift:
    def __call__(self, r, s, m):
        if random.random() < 0.5:
            m = torch.roll(m, shifts=1, dims=1)  # down shift (height)
        return r, s, m
    
class HorizonalMaskFlip:
    def __call__(self, r, s, m):
        m = torch.flip(m, dims=(2,))  # horizontal flip (width)
        return r, s, m

class VerticalMaskFlip:
    def __call__(self, r, s, m):
        m = torch.flip(m, dims=(1,))  # vertical flip (height)
        return r, s, m
    
class MaskRotate:
    def __call__(self, r, s, m):
        # randomly rotate mask by 90, 180, or 270 degrees
        k = random.randint(1, 3)
        m = torch.rot90(m, k=k, dims=(1, 2))
        return r, s, m
    
class Identity:
    def __call__(self, r, s, m):
        return r, s, m

def _seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def make_loaders(
    zarr_dir: str,
    sim_type: int = 0,
    batch_size: int = 16,
    num_workers: int = 8,
    pin_memory: bool = True,
    normalize_results: bool = True,
    normalize_sparams: bool = True,
    aug_train: bool = False,
    shuffle_test: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[DistributedSampler], Tuple]:
    """
    Build train/val/test DataLoaders from split index files in the Zarr root.
    """
    # Load split indices
    train_idx = np.load(os.path.join(zarr_dir, "train_idx.npy"))
    val_idx   = np.load(os.path.join(zarr_dir, "val_idx.npy"))
    test_idx  = np.load(os.path.join(zarr_dir, "test_idx.npy"))

    # stats files
    results_stats_path = os.path.join(zarr_dir, "results_mean_std.npy")
    sparams_stats_path = os.path.join(zarr_dir, "sparams_mean_std.npy")

    transform = None
    if aug_train:
        train_transform = Compose([RandomDownShift(), 
                                   ChooseOne([HorizonalMaskFlip(),
                                              VerticalMaskFlip(),
                                              MaskRotate(),
                                              Identity()])])
    else:
        train_transform = transform

    train_ds = ZarrTripletDataset(
        zarr_dir=zarr_dir,
        sim_type=sim_type,
        indices=train_idx,
        normalize_results=normalize_results,
        normalize_sparams=normalize_sparams,
        results_stats_path=results_stats_path if normalize_results else None,
        sparams_stats_path=sparams_stats_path if normalize_sparams else None,
        transform=train_transform,
    )
    val_ds = ZarrTripletDataset(
        zarr_dir=zarr_dir,
        sim_type=sim_type,
        indices=val_idx,
        normalize_results=normalize_results,
        normalize_sparams=normalize_sparams,
        results_stats_path=results_stats_path if normalize_results else None,
        sparams_stats_path=sparams_stats_path if normalize_sparams else None,
        transform=transform,
    )
    test_ds = ZarrTripletDataset(
        zarr_dir=zarr_dir,
        sim_type=sim_type,
        indices=test_idx,
        normalize_results=normalize_results,
        normalize_sparams=normalize_sparams,
        results_stats_path=results_stats_path if normalize_results else None,
        sparams_stats_path=sparams_stats_path if normalize_sparams else None,
        transform=transform,
    )

    # --- samplers (DDP-aware)
    if is_using_ddp():
        # For training we want equal steps across ranks → drop_last=True here.
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        # Rank-strided evaluation covers each layout exactly once, without padding.
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        val_sampler = range(rank, len(val_ds), world_size)
        test_sampler = range(rank, len(test_ds), world_size)
    else:
        train_sampler = val_sampler = test_sampler = None

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_seed_worker,
    )

    # --- loaders (no shuffle if sampler is set)
    train_loader = DataLoader(train_ds, sampler=train_sampler, shuffle=(train_sampler is None), drop_last=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, sampler=val_sampler,   shuffle=False, drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, sampler=test_sampler,  shuffle=(test_sampler is None and shuffle_test), drop_last=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, train_sampler, (2, 18, 18)

