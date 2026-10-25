"""Original Figure1 dual-branch stability study; see studies/README.md for usage and aggregation."""

import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple

import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loaders import make_loaders
from networks import ForwardModel, FourierEncoding, modulate_net
from rfic import batch_SM_to_results, get_s_params_matrix
from utils import Statistics, make_pbar, seed_everything


def parse_args():
    p = argparse.ArgumentParser(
        description="Instability study with dual trends: Solver (noisy [s,r]) vs GT solver (noisy s_gt)."
    )
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--sim-type", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--loading-workers", type=int, default=2)
    p.add_argument("--max-batches", type=int, default=25, help="Test batches used per sigma.")
    p.add_argument("--std-batches", type=int, default=125, help="Batches used to estimate representation std.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--rfic-num-workers", type=int, default=8)
    p.add_argument(
        "--forward-model-path",
        type=str,
        required=True,
        help="Checkpoint with residual dimensions (not sparam-only).",
    )
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    return p.parse_args()


def infer_base_width(state_dict: Dict[str, torch.Tensor]) -> int:
    return int(state_dict["Simulator.0.weight"].shape[0])


def infer_disable_zcomp(state_dict: Dict[str, torch.Tensor], sparam_size: int = 40) -> bool:
    pat = re.compile(r"^Simulator\.(\d+)\.weight$")
    last_linear = None
    last_idx = -1
    for k, v in state_dict.items():
        m = pat.match(k)
        if m is None or v.ndim != 2:
            continue
        idx = int(m.group(1))
        if idx > last_idx:
            last_idx = idx
            last_linear = v
    if last_linear is None:
        raise ValueError("Could not infer Simulator output dim from checkpoint.")
    out_dim = int(last_linear.shape[0])
    return out_dim == sparam_size


def load_pretrained_forward_model(
    ckpt_path: str,
    input_shape,
    num_fb: int,
    results_stat: torch.Tensor,
    sparams_stat: torch.Tensor,
    device: torch.device,
) -> ForwardModel:
    state_dict = torch.load(ckpt_path, map_location="cpu")
    base_width = infer_base_width(state_dict)
    disable_zcomp = infer_disable_zcomp(state_dict=state_dict, sparam_size=40)
    if disable_zcomp:
        raise ValueError(f"Expected residual dimensions, got sparam-only checkpoint: {ckpt_path}")
    model = ForwardModel(
        input_shape=input_shape,
        num_FB=num_fb,
        base_width=base_width,
        results_stat=results_stat.to(device),
        sparams_stat=sparams_stat.to(device),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def get_mask_freq_repr(model: ForwardModel, mask: torch.Tensor, fb_norm: torch.Tensor) -> torch.Tensor:
    freq_encoding = FourierEncoding(fb_norm, half_dim=model.freq_encoding_half_dim)
    modulation_params = model.Conditioner(freq_encoding)
    modulate_net(model.Simulator, modulation_params)
    simulator_feat = model.Simulator[:-1](mask)
    repr_clean = model.Simulator[-1](simulator_feat).view(mask.shape[0], len(fb_norm), -1)
    return repr_clean


@torch.no_grad()
def output_from_repr(model: ForwardModel, repr_in: torch.Tensor, spec_norm: torch.Tensor, fb_norm: torch.Tensor) -> torch.Tensor:
    cap_cond = FourierEncoding(spec_norm[..., :2], model.cap_encoding_half_dim)
    cap_cond = cap_cond.view(cap_cond.shape[0], cap_cond.shape[1], -1)
    solver_input = torch.cat(
        [
            repr_in.unsqueeze(1).expand(-1, cap_cond.shape[1], -1, -1),
            cap_cond.unsqueeze(2).expand(-1, -1, len(fb_norm), -1),
        ],
        dim=-1,
    )
    return model.Solver(solver_input)


def update_r2_stats(avg_first, avg_sec, avg_rss, target: torch.Tensor, pred: torch.Tensor, reduce_dims, n: int):
    avg_first.update(target.mean(dim=reduce_dims), n)
    avg_sec.update(target.square().mean(dim=reduce_dims), n)
    avg_rss.update((target - pred).square().mean(dim=reduce_dims), n)


def finalize_r2(avg_first, avg_sec, avg_rss) -> torch.Tensor:
    data_var = (avg_sec.avg - avg_first.avg.square()) * avg_first.n / (avg_first.n - 1)
    data_var = torch.clamp(data_var, min=1e-12)
    return 1 - avg_rss.avg / data_var


@torch.no_grad()
def estimate_repr_std(
    test_loader,
    model: ForwardModel,
    fb_norm: torch.Tensor,
    max_batches: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d_total = model.sparam_size + model.zcomp_dim
    sum_z = torch.zeros(d_total, dtype=torch.float64)
    sum_z2 = torch.zeros(d_total, dtype=torch.float64)
    sum_sgt = torch.zeros(model.sparam_size, dtype=torch.float64)
    sum_sgt2 = torch.zeros(model.sparam_size, dtype=torch.float64)
    n_z = 0
    n_sgt = 0

    for bidx, batch in enumerate(make_pbar(test_loader, desc="estimating std")):
        if bidx >= max_batches:
            break
        _, s_gt, m = batch
        z = get_mask_freq_repr(
            model=model,
            mask=m.to(device=device, dtype=torch.float32),
            fb_norm=fb_norm.to(device=device, dtype=torch.float32),
        ).cpu()
        z_flat = z.reshape(-1, d_total).to(dtype=torch.float64)
        sgt_flat = s_gt.reshape(-1, model.sparam_size).to(dtype=torch.float64)

        sum_z += z_flat.sum(dim=0)
        sum_z2 += z_flat.square().sum(dim=0)
        sum_sgt += sgt_flat.sum(dim=0)
        sum_sgt2 += sgt_flat.square().sum(dim=0)
        n_z += z_flat.shape[0]
        n_sgt += sgt_flat.shape[0]

    if n_z < 2 or n_sgt < 2:
        raise RuntimeError(f"Not enough samples for std estimation: n_z={n_z}, n_sgt={n_sgt}")

    mean_z = sum_z / float(n_z)
    var_z = (sum_z2 - float(n_z) * mean_z.square()) / float(n_z - 1)
    std_z = var_z.clamp_min(1e-12).sqrt().to(dtype=torch.float32)
    std_s = std_z[: model.sparam_size]
    std_r = std_z[model.sparam_size :]

    mean_sgt = sum_sgt / float(n_sgt)
    var_sgt = (sum_sgt2 - float(n_sgt) * mean_sgt.square()) / float(n_sgt - 1)
    std_sgt = var_sgt.clamp_min(1e-12).sqrt().to(dtype=torch.float32)
    return std_s, std_r, std_sgt


@torch.no_grad()
def evaluate_one_setting(
    test_loader,
    model: ForwardModel,
    fb_norm: torch.Tensor,
    results_stat: torch.Tensor,
    sparams_stat: torch.Tensor,
    sigma: float,
    max_batches: int,
    std_s: torch.Tensor,
    std_r: torch.Tensor,
    std_sgt: torch.Tensor,
    rfic_num_workers: int,
    seed: int,
    device: torch.device,
):
    seed_everything(seed, strict=False)

    # model representation / result stats
    avg_first_repr_m, avg_sec_repr_m, avg_rss_repr_m = Statistics[torch.Tensor].get_statistics(3)
    avg_first_res_m, avg_sec_res_m, avg_rss_res_m = Statistics[torch.Tensor].get_statistics(3)
    avg_mae_repr_m = Statistics()
    avg_mae_res_m = Statistics()

    # gt solver representation / result stats
    avg_first_repr_g, avg_sec_repr_g, avg_rss_repr_g = Statistics[torch.Tensor].get_statistics(3)
    avg_first_res_g, avg_sec_res_g, avg_rss_res_g = Statistics[torch.Tensor].get_statistics(3)
    avg_mae_repr_g = Statistics()
    avg_mae_res_g = Statistics()

    # for rfic branch de/normalization
    c1c2_mean = results_stat[0, :2].to(device=device, dtype=torch.float32)
    c1c2_std = results_stat[1, :2].to(device=device, dtype=torch.float32)
    f_mean = results_stat[0, 2].to(device=device, dtype=torch.float32)
    f_std = results_stat[1, 2].to(device=device, dtype=torch.float32)
    out_mean = results_stat[0, 3:].to(device=device, dtype=torch.float32).view(1, 1, 1, 4)
    out_std = results_stat[1, 3:].to(device=device, dtype=torch.float32).view(1, 1, 1, 4)
    s_mean = sparams_stat[0].to(device=device, dtype=torch.float32)
    s_std = sparams_stat[1].to(device=device, dtype=torch.float32)
    fb_norm_dev = fb_norm.to(device=device, dtype=torch.float32)
    fb_denorm = fb_norm_dev * f_std + f_mean

    std_s = std_s.to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_r = std_r.to(device=device, dtype=torch.float32).view(1, 1, -1)
    std_sgt = std_sgt.to(device=device, dtype=torch.float32).view(1, 1, -1)

    for bidx, batch in enumerate(make_pbar(test_loader, desc=f"gaussian sigma={sigma:.4f}")):
        if bidx >= max_batches:
            break
        r, s_gt, m = batch
        spec_norm = r[..., 0, :2].to(device=device, dtype=torch.float32)
        result_gt = r[..., 3:]
        s_gt_dev = s_gt.to(device=device, dtype=torch.float32)
        m = m.to(device=device, dtype=torch.float32)

        # branch A: model representation z=[s_pred, r_pred]
        z_clean = get_mask_freq_repr(model=model, mask=m, fb_norm=fb_norm_dev)
        s_pred = z_clean[..., : model.sparam_size]
        r_pred = z_clean[..., model.sparam_size :]
        z_noisy = torch.cat(
            [
                s_pred + torch.randn_like(s_pred) * std_s * sigma,
                r_pred + torch.randn_like(r_pred) * std_r * sigma,
            ],
            dim=-1,
        )
        pred_result_model = output_from_repr(
            model=model,
            repr_in=z_noisy,
            spec_norm=spec_norm,
            fb_norm=fb_norm_dev,
        ).cpu()

        z_clean_cpu = z_clean.cpu()
        z_noisy_cpu = z_noisy.cpu()
        mae_repr_m = (z_noisy_cpu - z_clean_cpu).abs().mean()
        mae_res_m = (pred_result_model - result_gt).abs().mean()
        avg_mae_repr_m.update(mae_repr_m.item(), z_clean_cpu.numel())
        avg_mae_res_m.update(mae_res_m.item(), result_gt.numel())

        n_repr_m = int(np.prod(z_clean_cpu.shape[:-1]))
        n_res = int(np.prod(result_gt.shape[:-2]))
        update_r2_stats(avg_first_repr_m, avg_sec_repr_m, avg_rss_repr_m, z_clean_cpu, z_noisy_cpu, (0, 1), n_repr_m)
        update_r2_stats(avg_first_res_m, avg_sec_res_m, avg_rss_res_m, result_gt, pred_result_model, (0, 1), n_res)

        # branch B: GT solver representation=s_gt
        s_noisy_gt = s_gt_dev + torch.randn_like(s_gt_dev) * std_sgt * sigma
        s_noisy_gt_cpu = s_noisy_gt.cpu()
        mae_repr_g = (s_noisy_gt_cpu - s_gt).abs().mean()
        avg_mae_repr_g.update(mae_repr_g.item(), s_gt.numel())

        s_noisy_denorm = s_noisy_gt * s_std + s_mean
        sm = get_s_params_matrix(s_noisy_denorm, interleaved_mag_phase=True)
        spec_denorm = spec_norm * c1c2_std.view(1, 1, 2) + c1c2_mean.view(1, 1, 2)
        c1c2_grid = spec_denorm.unsqueeze(2).expand(-1, -1, len(fb_norm), -1)
        f_grid = fb_denorm.view(1, 1, -1, 1).expand(spec_denorm.shape[0], spec_denorm.shape[1], -1, -1)
        spec_rfic = torch.cat([c1c2_grid, f_grid], dim=-1)
        pred_result_denorm_rfic = batch_SM_to_results(sm, spec_rfic, num_workers=rfic_num_workers)
        pred_result_denorm_rfic = pred_result_denorm_rfic.to(device=device, dtype=torch.float32)
        pred_result_rfic = ((pred_result_denorm_rfic - out_mean) / out_std).cpu()
        mae_res_g = (pred_result_rfic - result_gt).abs().mean()
        avg_mae_res_g.update(mae_res_g.item(), result_gt.numel())

        n_repr_g = int(np.prod(s_gt.shape[:-1]))
        update_r2_stats(avg_first_repr_g, avg_sec_repr_g, avg_rss_repr_g, s_gt, s_noisy_gt_cpu, (0, 1), n_repr_g)
        update_r2_stats(avg_first_res_g, avg_sec_res_g, avg_rss_res_g, result_gt, pred_result_rfic, (0, 1), n_res)

    r2_repr_m = finalize_r2(avg_first_repr_m, avg_sec_repr_m, avg_rss_repr_m)
    r2_res_m = finalize_r2(avg_first_res_m, avg_sec_res_m, avg_rss_res_m)
    r2_repr_g = finalize_r2(avg_first_repr_g, avg_sec_repr_g, avg_rss_repr_g)
    r2_res_g = finalize_r2(avg_first_res_g, avg_sec_res_g, avg_rss_res_g)

    return {
        "r2_repr_model_mean": r2_repr_m.mean().item(),
        "r2_result_model_mean": r2_res_m.mean().item(),
        "mae_repr_model_mean": avg_mae_repr_m.avg,
        "mae_result_model_mean": avg_mae_res_m.avg,
        "r2_repr_gtsolver_mean": r2_repr_g.mean().item(),
        "r2_result_gtsolver_mean": r2_res_g.mean().item(),
        "mae_repr_gtsolver_mean": avg_mae_repr_g.avg,
        "mae_result_gtsolver_mean": avg_mae_res_g.avg,
    }


def main():
    args = parse_args()
    seed_everything(args.seed, strict=False)

    noise_dist = "gaussian"
    noise_sigmas = np.arange(0.0, 0.2001, 0.01).tolist()

    _, _, test_loader, _, input_shape = make_loaders(
        zarr_dir=args.data_dir,
        sim_type=args.sim_type,
        batch_size=args.batch_size,
        num_workers=args.loading_workers,
        pin_memory=False,
        normalize_results=True,
        normalize_sparams=True,
        aug_train=False,
        shuffle_test=True,
    )

    first_batch = next(iter(test_loader))[0]
    fb_norm = first_batch[0, 0, :, 2].cpu()
    results_stat = torch.from_numpy(np.load(os.path.join(args.data_dir, "results_mean_std.npy")))[:, args.sim_type].cpu() + 1e-8
    sparams_stat = torch.from_numpy(np.load(os.path.join(args.data_dir, "sparams_mean_std.npy"))).cpu() + 1e-8

    device = torch.device(args.device)
    model = load_pretrained_forward_model(
        ckpt_path=args.forward_model_path,
        input_shape=input_shape,
        num_fb=fb_norm.shape[0],
        results_stat=results_stat,
        sparams_stat=sparams_stat,
        device=device,
    )

    std_s, std_r, std_sgt = estimate_repr_std(
        test_loader=test_loader,
        model=model,
        fb_norm=fb_norm,
        max_batches=args.std_batches,
        device=device,
    )
    print(
        f"[std] from {args.std_batches} batches | "
        f"mean(std_s)={std_s.mean().item():.6f}, mean(std_r)={std_r.mean().item():.6f}, "
        f"mean(std_sgt)={std_sgt.mean().item():.6f}"
    )

    records = []
    for sigma in noise_sigmas:
        m = evaluate_one_setting(
            test_loader=test_loader,
            model=model,
            fb_norm=fb_norm,
            results_stat=results_stat,
            sparams_stat=sparams_stat,
            sigma=sigma,
            max_batches=args.max_batches,
            std_s=std_s,
            std_r=std_r,
            std_sgt=std_sgt,
            rfic_num_workers=args.rfic_num_workers,
            seed=args.seed,
            device=device,
        )
        rec = {
            "dist": noise_dist,
            "sigma": sigma,
            **m,
            # aliases for compatibility with previous plotting names
            "r2_sparam_mean": m["r2_repr_model_mean"],
            "r2_result_outputnet_mean": m["r2_result_model_mean"],
            "r2_result_sp2imp_mean": m["r2_result_gtsolver_mean"],
            "r2_result_mean": m["r2_result_gtsolver_mean"],
        }
        records.append(rec)
        print(
            f"[{noise_dist}] sigma={sigma:.4f} | "
            f"model repr R2={rec['r2_repr_model_mean']:.4f}, model result R2={rec['r2_result_model_mean']:.4f} | "
            f"GT repr R2={rec['r2_repr_gtsolver_mean']:.4f}, GT result R2={rec['r2_result_gtsolver_mean']:.4f}"
        )

    os.makedirs(args.save_dir, exist_ok=True)
    out_json = os.path.join(args.save_dir, "rfic_instability_records.json")
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON to {out_json}")


if __name__ == "__main__":
    main()
