import torch
import torch.nn as nn
from networks import CNNForwardModel, ForwardModel, ResForwardModel
from loaders import make_loaders
from utils import Statistics, write_perf_to_json, seed_everything, init_distributed, is_main_process, reduce_mean, is_using_ddp, cleanup_dist, make_pbar
from typing import Optional, Tuple
import numpy as np
from datetime import datetime
import argparse
import os

def compute_loss(pred, Y, loss_fn):
    s, r = Y
    ps, pr = pred
    loss = loss_fn(ps, s) + loss_fn(pr, r)
    return loss

def prepare_batch(batch, device, args):
    r, s, m = batch
    spec = r[..., 0, :2]      # (N, C, 2)
    result = r[..., 3:]       # (N, C, FB, 4)
    m = m.to(device, non_blocking=True)
    spec = spec.to(device, non_blocking=True)
    s = s.to(device, non_blocking=True)
    result = result.to(device, non_blocking=True)

    fb = freq_band.to(device)
    X = [m, spec, fb]
    Y = [s, result]
    return X, Y, m, spec, s

def unwrap_forward_model(model):
    if hasattr(model, "module") and isinstance(model.module, ForwardModel):
        return model.module
    return model

def train_loop(epoch_idx, model, train_loader, device, optimizer, lr_scheduler, loss_fn, args):
    model.train()
    avg_loss = Statistics(momentum=0.99)

    pbar = make_pbar(train_loader, desc=f"Training Epoch {epoch_idx}")
    for batch_idx, batch in enumerate(pbar):
        X, Y, m, spec, s = prepare_batch(batch, device, args)

        optimizer.zero_grad(set_to_none=True)

        if args.dependency_reg > 0:
            pred, dependency_penalty = model(*X, return_dependency_penalty=True)
        else:
            pred, dependency_penalty = model(*X), torch.tensor(0.0, device=device)
        loss = compute_loss(pred, Y, loss_fn) + args.dependency_reg * dependency_penalty

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()

        loss, N = reduce_mean(loss, len(m))
        if is_main_process():
            avg_loss.update(loss.item(), N)
            pbar.set_postfix_str(f"Loss: {avg_loss.avg:.4f}")

    if is_main_process():
        return avg_loss.avg
    else:
        return torch.tensor(float("inf"))

@torch.no_grad()
def val_loop(epoch_idx:Optional[int], model, val_loader, device, loss_fn, args):
    model.eval()
    # Evaluation shards can be uneven or empty; no collectives inside the loop.
    eval_model = unwrap_forward_model(model)
    totals = torch.zeros(2, dtype=torch.float64, device=device)

    desc_str = f"Validation Epoch {epoch_idx}" if epoch_idx is not None else "Testing"
    pbar = make_pbar(val_loader, desc=desc_str)
    for batch_idx, batch in enumerate(pbar):
        X, Y, m, _, _ = prepare_batch(batch, device, args)
        pred = eval_model(*X)
        loss = compute_loss(pred, Y, loss_fn)
        totals[0] += loss.double() * len(m)
        totals[1] += len(m)

    if is_using_ddp():
        torch.distributed.all_reduce(totals)
    return (totals[0] / totals[1]).item()


@torch.no_grad()
def compute_R2(model, loader, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-component 1-SSE/SST, pooling all observations, not components.

    EM pools layout/frequency; design pools layout/context/frequency.
    Population centered sums are batch/rank merged; constant targets yield NaN.
    """
    model.eval()
    eval_model = unwrap_forward_model(model)
    # Rows: count, mean, centered sum of squares, prediction SSE.
    stats = [torch.zeros((4, width), dtype=torch.float64, device=device)
             for width in (eval_model.sparam_size, 4)]
    pbar = make_pbar(loader, desc="Computing R2")
    for batch in pbar:
        X, Y, _, _, _ = prepare_batch(batch, device, args)
        pred = eval_model(*X)
        for acc, target, prediction in zip(stats, Y, pred):
            target = target.double().reshape(-1, acc.shape[1])
            prediction = prediction.double().reshape_as(target)
            count = target.shape[0]
            mean = target.mean(dim=0)
            delta = mean - acc[1]
            total = acc[0] + count
            acc[2] += ((target - mean).square()).sum(dim=0) + delta.square() * acc[0] * count / total
            acc[1] += delta * count / total
            acc[0] = total
            acc[3] += (target - prediction).square().sum(dim=0)

    scores = []
    for acc in stats:
        if is_using_ddp():
            gathered = [torch.empty_like(acc) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, acc)
            acc = torch.zeros_like(acc)
            for part in gathered:
                if part[0, 0] == 0:
                    continue
                total = acc[0] + part[0]
                delta = part[1] - acc[1]
                acc[2] += part[2] + delta.square() * acc[0] * part[0] / total
                acc[1] += delta * part[0] / total
                acc[0] = total
                acc[3] += part[3]
        scores.append(torch.where(acc[2] > 0, 1 - acc[3] / acc[2], torch.nan))
    return tuple(scores)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim-type', type=int, default=0, choices=[0,1,2], help='The simulation type that the network learns.')
    parser.add_argument('--data-dir', type=str, default="data/", help='The root of the dataset.')


    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument("--save-root", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--load-forward-model", type=str, default=None, help="Path to the pretrained forward model")

    parser.add_argument('--base-width', type=int, default=64, choices=[32,64], help='The base width of the networks.')
    parser.add_argument(
        '--simulator-arch', choices=['resnet', 'cnn'], default='resnet',
        help='Simulator architecture: residual (default) or the manuscript plain CNN.',
    )
    parser.add_argument('--loading-workers', type=int, default=8, help='The number of workers in data loaders.')

    parser.add_argument('--dependency-reg', type=float, default=0.0, help='Weight for dependency regularization between the S-parameter and Zcomp heads.')


    args = parser.parse_args()

    return args

def save_current_model(model, save_root:str):
    if hasattr(model, "module") and isinstance(model.module, ForwardModel):
        torch.save(model.module.state_dict(), os.path.join(save_root, f"model.pth"))
    else:
        torch.save(model.state_dict(), os.path.join(save_root, f"model.pth"))

def run():
    args = parse_args()

    # Set up the device
    is_dist, local_rank, global_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Set the seed
    seed_everything(args.seed, rank=global_rank, strict=False)

    # Set up logging
    if is_main_process():
        local_log = {"train/loss": [], "val/loss": [], "start_time": datetime.now().strftime("%Y/%m/%d %H:%M:%S")}

    # Set up the loaders
    train_loader, val_loader, test_loader, train_sampler, input_shape = make_loaders(
        zarr_dir=args.data_dir,
        sim_type=args.sim_type,
        batch_size=args.batch_size,
        num_workers=args.loading_workers,
        pin_memory=True,
        normalize_results=True,
        normalize_sparams=True,
        aug_train=True,
    )
    global freq_band
    freq_band = next(iter(train_loader))[0][0, 0, :, 2].to(device)

    # Set up save root identically on every rank.
    save_root = os.path.join(args.save_root, "forward_model", f"simtype_{args.sim_type}")
    if args.weight_decay > 0:
        save_root = os.path.join(save_root, f"wd_{args.weight_decay}")
    args.save_root = save_root
    if is_main_process():
        os.makedirs(save_root, exist_ok=True)
        args.save_root = save_root
        print(f"Saving to {save_root}")
        write_perf_to_json(vars(args), save_root, filename="args.json")


    # Load normalization stats and attach to model
    global results_stat, sparams_stat
    results_stat = torch.from_numpy(np.load(os.path.join(args.data_dir, "results_mean_std.npy")))[:, args.sim_type].to(device) + 1e-8  # (2, 7)
    sparams_stat = torch.from_numpy(np.load(os.path.join(args.data_dir, "sparams_mean_std.npy"))).to(device) + 1e-8

    # Set up the model
    model_cls = ResForwardModel if args.simulator_arch == 'resnet' else CNNForwardModel
    model = model_cls(
        input_shape=input_shape,
        num_FB=len(freq_band),
        base_width=args.base_width,
        results_stat=results_stat,
        sparams_stat=sparams_stat,
    )

    if args.load_forward_model:
        model.load_state_dict(torch.load(args.load_forward_model, map_location="cpu"))
        if is_main_process():
            print(f"Loaded ForwardModel from {args.load_forward_model}")
            # keep a copy at the save root in case testing is performed without training
            save_current_model(model, args.save_root)

    model = model.to(device)
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False, broadcast_buffers=False)


    # Set up the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # One cosine decay over all optimizer updates.
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=1e-6)


    best_val_loss = float("inf")
    for epoch_idx in range(args.epochs):
        # Train the model
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_idx)  # ensures different shuffling each epoch across ranks when in DDP
        train_loss = train_loop(epoch_idx, model, train_loader, device, optimizer, lr_scheduler, nn.MSELoss(reduction="mean"), args)

        # Validate the model
        val_loss = val_loop(epoch_idx, model, val_loader, device, nn.MSELoss(reduction="mean"), args)

        # Log the results
        if is_main_process():
            local_log[f"train/loss"].append(train_loss)
            local_log[f"val/loss"].append(val_loss)

            # Save the model; Simulator and Solver are saved separately
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                local_log["val/best_loss"] = best_val_loss
                print(f"Validation loss improved to {best_val_loss:.4f} at Epoch {epoch_idx}. Saving model...")
                save_current_model(model, args.save_root)


            # Write the results to the log file
            write_perf_to_json(local_log, save_root)
        
    # Rank zero finishes saving before every rank loads the same best weights.
    if is_using_ddp():
        torch.distributed.barrier()
    if is_main_process():
        print("Loading best model for testing...")
    unwrap_forward_model(model).load_state_dict(
        torch.load(os.path.join(save_root, "model.pth"), map_location=device, weights_only=True)
    )

    r2_s, r2_r = compute_R2(model, test_loader, device, args)
    if is_main_process():
        # store r2s and r2r
        np.save(os.path.join(save_root, "r2_s.npy"), r2_s.cpu().numpy())
        np.save(os.path.join(save_root, "r2_r.npy"), r2_r.cpu().numpy())
        print("Result R2 mean and std:")
        print(r2_r.mean().item(), r2_r.std().item())
        print("sparam R2 mean and std:")
        print(r2_s.mean().item(), r2_s.std().item())

    test_loss = val_loop(None, model, test_loader, device, nn.MSELoss(reduction="mean"), args)
    if is_main_process():
        print(f"Test loss: {test_loss:.4f}; Best val loss: {best_val_loss:.4f}")
        local_log["test/loss"] = test_loss

        local_log["test/r2s_mean"] = r2_s.mean().item()
        local_log["test/r2s_std"] = r2_s.std().item()
        local_log["test/r2r_mean"] = r2_r.mean().item()
        local_log["test/r2r_std"] = r2_r.std().item()
        local_log["end_time"] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        write_perf_to_json(local_log, save_root)

if __name__ == '__main__':
    try:
        run()
    finally:
        cleanup_dist()
