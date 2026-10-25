#!/usr/bin/env python3
"""Build a Zarr dataset from processed NumPy shards.

The input directory must contain matching result, S-parameter, and mask shards.
The script writes a Zarr directory store, train/validation/test split indices,
and normalization statistics consumed by the training pipeline.
"""
import argparse
import os
import re
from glob import glob
from typing import List, Tuple
from tqdm.auto import tqdm

import numpy as np
import zarr
from numcodecs import Blosc, BitRound

# --------- Naming conventions ---------
SUFFIX_RESULTS = "_valid_results.npy"
SUFFIX_SPARAMS = "_valid_s_params.npy"
SUFFIX_MASK    = "_mask.npy"

STEM_PATTERN = re.compile(r"(\d+)_(\d+)_")  # e.g., "0_2000_", "10000_12000_"

# --------- Helpers ---------
def list_result_stems(data_dir: str) -> List[str]:
    """Return sorted stems like '0_2000_', '10000_12000_', ... based on the results files."""
    paths = sorted(glob(os.path.join(data_dir, f"*{SUFFIX_RESULTS}")))
    stems = []
    for p in paths:
        base = os.path.basename(p)
        m = STEM_PATTERN.match(base)
        if m:
            stems.append(m.group(0))  # includes trailing underscore
    stems.sort(key=lambda s: int(STEM_PATTERN.match(s).group(1)))
    return stems

def match_triplet_paths(data_dir: str, stem: str) -> Tuple[str, str, str]:
    """Return (results_path, s_params_path, masks_path). Raise if missing."""
    rp = os.path.join(data_dir, f"{stem}{SUFFIX_RESULTS.lstrip('_')}")
    sp = os.path.join(data_dir, f"{stem}{SUFFIX_SPARAMS.lstrip('_')}")
    mp = os.path.join(data_dir, f"{stem}{SUFFIX_MASK.lstrip('_')}")
    if not os.path.exists(rp) or not os.path.exists(sp) or not os.path.exists(mp):
        raise FileNotFoundError(f"Missing triplet for stem {stem}")
    return rp, sp, mp

def choose_shuffle_for_dtype(dtype: np.dtype) -> int:
    """SHUFFLE/BITSHUFFLE choice (BitShuffle is good for float32)."""
    return Blosc.BITSHUFFLE if dtype.kind == "f" else Blosc.SHUFFLE

# --------- Core builder ---------
def build_zarr(
    data_dir: str,
    out_dir: str,
    chunk_len: int = 4,
    zstd_level: int = 5,
    float_bitround_keepbits: int = 16,
):
    # Never remove or reuse an existing path (including the raw input directory).
    if os.path.lexists(out_dir):
        raise FileExistsError(f"Output already exists: {out_dir}. Choose a new output path.")
    if chunk_len < 1:
        raise ValueError("chunk_len must be positive")
    stems = list_result_stems(data_dir)
    if not stems:
        raise SystemExit(f"No *{SUFFIX_RESULTS} files found in {data_dir}")

    # ---- Pass 0: determine N and post-conversion inner shapes with minimal I/O
    # We only need headers here; mmap_mode='r' avoids loading payload into RAM.
    # Post-conversion shapes:
    #   results: (n, ...)                    -> float32 (same inner shape)
    #   s_params: (n, ..., ) complex128      -> float32 with an extra last dim: (..., 2)
    #   masks: (n, ...) int64                -> bool (same inner shape)
    N = 0
    results_inner_shape = None
    sparams_inner_shape_conv = None
    masks_inner_shape = None

    # Peek first triplet to establish canonical inner shapes (post-conversion)
    first_rp, first_sp, first_mp = match_triplet_paths(data_dir, stems[0])
    r0 = np.load(first_rp, mmap_mode="r")
    s0 = np.load(first_sp, mmap_mode="r")
    m0 = np.load(first_mp, mmap_mode="r")
    results_inner_shape = r0.shape[1:]
    sparams_inner_shape_conv = s0.shape[1:] + (2,)  # stacking real/imag at the last axis
    masks_inner_shape = m0.shape[1:]
    del r0, s0, m0

    # Compute total N and verify per-triplet first dims match (minimal, necessary check)
    for stem in stems:
        rp, sp, mp = match_triplet_paths(data_dir, stem)
        r_mm = np.load(rp, mmap_mode="r")
        s_mm = np.load(sp, mmap_mode="r")
        m_mm = np.load(mp, mmap_mode="r")
        n_r, n_s, n_m = r_mm.shape[0], s_mm.shape[0], m_mm.shape[0]
        if not (n_r == n_s == n_m):
            raise ValueError(f"Length mismatch in triplet {stem}: {n_r}, {n_s}, {n_m}")
        # Also ensure inner shapes consistent across shards (post-conversion)
        if r_mm.shape[1:] != results_inner_shape:
            raise ValueError(f"results inner shape mismatch at {stem}")
        if (s_mm.shape[1:] + (2,)) != sparams_inner_shape_conv:
            raise ValueError(f"s_params inner shape mismatch at {stem} (after stacking)")
        if m_mm.shape[1:] != masks_inner_shape:
            raise ValueError(f"masks inner shape mismatch at {stem}")
        N += n_r
        del r_mm, s_mm, m_mm

    print(f"Total samples (N): {N}")
    print(f"results inner shape: {results_inner_shape}")
    print(f"s_params inner shape (after stacking real/imag): {sparams_inner_shape_conv}")
    print(f"masks inner shape: {masks_inner_shape}")

    # ---- Create Zarr store and arrays
    os.makedirs(out_dir, exist_ok=False)
    store = zarr.DirectoryStore(out_dir)
    root = zarr.group(store=store, overwrite=False)

    # All float arrays get BitRound(keepbits) to reduce disk usage
    float_filter = BitRound(keepbits=int(float_bitround_keepbits)) if float_bitround_keepbits > 0 else None

    # Dtypes after conversion
    results_dtype = np.float32
    sparams_dtype = np.float32  # stacked real/imag
    masks_dtype   = np.bool_

    # Compressors
    results_compressor = Blosc(cname="zstd", clevel=zstd_level, shuffle=choose_shuffle_for_dtype(np.dtype("float32")))
    sparams_compressor = Blosc(cname="zstd", clevel=zstd_level, shuffle=choose_shuffle_for_dtype(np.dtype("float32")))
    masks_compressor   = Blosc(cname="zstd", clevel=zstd_level, shuffle=choose_shuffle_for_dtype(np.dtype("bool")))

    z_results = root.create(
        name="results",
        shape=(N,) + results_inner_shape,
        chunks=(chunk_len,results_inner_shape[0],1) + results_inner_shape[2:],
        dtype=results_dtype,
        compressor=results_compressor,
        filters=[float_filter] if float_filter else None,
        overwrite=True,
    )
    z_sparams = root.create(
        name="s_params",
        shape=(N,) + sparams_inner_shape_conv,
        chunks=(chunk_len,) + sparams_inner_shape_conv,
        dtype=sparams_dtype,
        compressor=sparams_compressor,
        filters=[float_filter] if float_filter else None,
        overwrite=True,
    )
    z_masks = root.create(
        name="masks",
        shape=(N,) + masks_inner_shape,
        chunks=(chunk_len,) + masks_inner_shape,
        dtype=masks_dtype,
        compressor=masks_compressor,
        filters=None,
        overwrite=True,
    )

    # ---- Streaming write (at most four layouts in memory, not a whole shard)
    cursor = 0
    for stem in stems:
        rp, sp, mp = match_triplet_paths(data_dir, stem)
        r_mm = np.load(rp, mmap_mode="r")
        s_mm = np.load(sp, mmap_mode="r")
        m_mm = np.load(mp, mmap_mode="r")
        print(f"Writing {stem} ...")
        for start in range(0, r_mm.shape[0], min(chunk_len, 4)):
            stop = min(start + min(chunk_len, 4), r_mm.shape[0])
            r_np, s_np, m_np = r_mm[start:stop], s_mm[start:stop], m_mm[start:stop]
            n = stop - start
            # ----------- Convert types -----------
            # results -> float32
            r_np = np.array(r_np, dtype=np.float32, copy=True)

            # s_params complex128 -> stack [mag, phase] in last dim as float32
            # Shape: (n, *s_inner) -> (n, *s_inner, 2)
            s_mag = np.abs(s_np).astype(np.float32)
            s_phase = np.angle(s_np).astype(np.float32)
            s_np_conv = np.stack([s_mag, s_phase], axis=-1)

            # masks int64 -> bool
            if m_np.dtype != np.bool_:
                m_np = m_np.astype(np.bool_)

            # ----------- convert zin from (real, img) to (mag, phase) -----------
            r_np_zin_real = r_np[..., 3]
            r_np_zin_imag = r_np[..., 4]
            r_np_zin_mag  = np.sqrt(r_np_zin_real**2 + r_np_zin_imag**2)
            r_np_zin_phase= np.arctan2(r_np_zin_imag, r_np_zin_real)
            r_np[..., 3] = r_np_zin_mag
            r_np[..., 4] = r_np_zin_phase

            # Write to Zarr
            sl = slice(cursor, cursor + n)
            z_results[sl] = r_np
            z_sparams[sl] = s_np_conv
            z_masks[sl]   = m_np
            cursor += n

            del r_np, s_np, s_np_conv, s_mag, s_phase, m_np
        del r_mm, s_mm, m_mm

    # ---- Provenance + consolidate metadata
    root.attrs.update(
        {
            "num_samples": int(N),
            "chunk_len": int(chunk_len),
            "zstd_level": int(zstd_level),
            "bitround_keepbits": int(float_bitround_keepbits),
            "results_dtype": "float32",
            "s_params_dtype": "float32 (stacked mag/phase in last dim)",
            "masks_dtype": "bool",
        }
    )
    zarr.consolidate_metadata(store)

    print("\nDone. Zarr dataset created at:", os.path.abspath(out_dir))
    print(root.tree())


def _mean_std_zarr(z: zarr.Array, reduce_axes:Tuple) -> np.ndarray:
    """
    Online batch-merge stats over all axes except the last (population stats, ddof=0).
    Keeps only (n, mean, var), per the 'simple batch stat updates' formulas (https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html):
        mean' = mean + (mean_b - mean) * nb/(n+nb)
        var'  = var + (var_b - var) * nb/(n+nb) + (mean_b - mean)^2 * n*nb/(n+nb)^2
    Returns: array of shape (2, last_dim): [mean; std]
    """
    last_dim  = z.shape[-1]
    block_len = z.chunks[0]             # stream in chunk-aligned blocks

    # Running aggregates
    n    = 0
    mean = np.zeros(last_dim, dtype=z.dtype)
    var  = np.zeros(last_dim, dtype=z.dtype)

    N0 = z.shape[0]
    for start in tqdm(range(0, N0, block_len)):
        stop = min(start + block_len, N0)
        b = z[start:stop]                               # block view

        # Per-block (population) stats along all axes except last
        mean_b = b.mean(axis=reduce_axes)               # shape (last_dim,)
        var_b  = b.var(axis=reduce_axes)                # ddof=0, shape (last_dim,)
        nb = int(np.prod(b.shape[:-1]))                 # elements reduced into each last-dim entry
        if nb == 0:
            raise ValueError("Empty block encountered during stats computation.")

        if n == 0:
            # Initialize from first block
            n, mean, var = nb, mean_b, var_b
            continue

        # Merge current block into running aggregates
        delta = mean_b - mean
        n_new = n + nb
        mean += delta * (nb / n_new)
        var  += (var_b - var) * (nb / n_new) + (delta ** 2) * (n * nb / (n_new ** 2))
        n = n_new

    std = np.sqrt(np.maximum(var, 0.0))
    return np.stack([mean, std], axis=0)


def stat_zarr(dataset_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std of:
      - 'results'  → shape (2, 3, last_dim), float32, where 3 corresponds to 3 different simulation outputs
      - 's_params' → shape (2, last_dim), float32
    """
    root = zarr.open(dataset_path, mode="r")
    z_results  = root["results"]
    z_sparams  = root["s_params"]

    results_stat  = _mean_std_zarr(z_results, reduce_axes=(0,1,3))
    sparams_stat  = _mean_std_zarr(z_sparams, reduce_axes=tuple(range(z_results.ndim-1)))

    return results_stat, sparams_stat

def split_zarr(dataset_path:str, split_ratios=(0.7, 0.1, 0.2), seed=123):
    """Split an existing Zarr dataset into train/val/test subsets."""
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"
    N = zarr.open(dataset_path, mode='r')["masks"].shape[0]
    all_idx = np.random.default_rng(seed=seed).permutation(N)
    n_train = int(split_ratios[0] * N)
    n_val   = int(split_ratios[1] * N)
    train_idx = all_idx[:n_train]
    val_idx   = all_idx[n_train:n_train+n_val]
    test_idx  = all_idx[n_train+n_val:]
    np.save(os.path.join(dataset_path, "train_idx.npy"), train_idx)
    np.save(os.path.join(dataset_path, "val_idx.npy"), val_idx)
    np.save(os.path.join(dataset_path, "test_idx.npy"), test_idx)

def parse_args():
    p = argparse.ArgumentParser(description="Build Zarr dataset from chunked .npy shards.")
    p.add_argument("--data-dir", required=True,
                   help="Folder containing *_valid_results.npy, *_valid_s_params.npy, and *_mask.npy shards.")
    p.add_argument("--out", required=True,
                   help="Output Zarr directory store path.")
    p.add_argument("--chunk-len", type=int, default=4, help="Chunk length along axis 0.")
    p.add_argument("--zstd-level", type=int, default=5, help="Blosc Zstd compression level (1-9).")
    p.add_argument("--float-bitround", type=int, default=16,
                   help="BitRound keepbits applied to all float32 arrays (results & s_params).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_zarr(
        data_dir=args.data_dir,
        out_dir=args.out,
        chunk_len=args.chunk_len,
        zstd_level=args.zstd_level,
        float_bitround_keepbits=args.float_bitround,
    )
    split_zarr(args.out, split_ratios=(0.7, 0.1, 0.2), seed=123)
    results_stat, sparams_stat = stat_zarr(args.out)

    # overwrite frequency stats to normalize freq from [10.0001, 60.0001] to [-1, 1]
    freq_mean = (9.99 + 60.01) / 2
    freq_std  = (60.01 - 9.99) / 2
    results_stat[0, :, 2] = freq_mean
    results_stat[1, :, 2] = freq_std

    # overwrite C1 stats to normalize C1 from [0, 400] to [-1, 1]
    c1_mean = (-0.01 + 400.01) / 2
    c1_std  = (400.01 - (-0.01)) / 2
    results_stat[0, :, 0] = c1_mean
    results_stat[1, :, 0] = c1_std

    # overwrite C2 stats to normalize C2 from [10, 400] to [-1, 1]
    c2_mean = (9.99 + 400.01) / 2
    c2_std  = (400.01 - 9.99) / 2
    results_stat[0, :, 1] = c2_mean
    results_stat[1, :, 1] = c2_std

    print("Results mean/std:\n", results_stat)
    print("S_params mean/std:\n", sparams_stat)

    np.save(os.path.join(args.out, "results_mean_std.npy"), results_stat)
    np.save(os.path.join(args.out, "sparams_mean_std.npy"), sparams_stat)

