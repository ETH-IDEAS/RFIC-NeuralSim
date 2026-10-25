# RFIC-NeuralSim

Main training code for the 5.4M and 21M neural simulators, covering input, interstage, and output matching. The original dataset is not distributed because of commercial licensing restrictions.

## Install

Use Python 3.11–3.13. From the repository root, an NVIDIA/CUDA 11.8 setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

For other platforms, choose the corresponding [PyTorch 2.6 wheels](https://pytorch.org/get-started/previous-versions/#v260).

## 1. Prepare the dataset

Place the original NumPy shards in one directory. Each shard has a shared numeric prefix, such as `0_2000_`, and these three files:

| Filename | Shape | Contents |
|---|---|---|
| `<prefix>valid_results.npy` | `(N,C,3,F,7)` | Circuit settings and design results |
| `<prefix>valid_s_params.npy` | `(N,F,20)` | Complex S-parameters in the original channel order |
| `<prefix>mask.npy` | `(N,2,18,18)` | Binary layout masks, with a zero last row |

`N`, `C`, and `F` count layouts, circuit contexts, and frequencies. The matching axis is input, output, interstage. Result fields are `[C1 (fF), C2 (fF), frequency (GHz), Re(Zin), Im(Zin), efficiency (%), gain (dB)]`, with impedance in ohms. Preserve the original filenames, row order, and channel order.

Run the existing preparation script once, choosing a new output directory:

```bash
python data_processing/dataset/create_zarr.py \
  --data-dir /path/to/raw_numpy --out /path/to/dataset.zarr
```

This produces the training-ready Zarr dataset, including train/validation/test splits and normalization statistics. No separate preprocessing commands are needed.

## 2. Train

Run from the repository root. These commands train on the full training partition for 500 epochs using eight GPUs.

**5.4M model:**

```bash
torchrun --standalone --nproc_per_node=8 forward_train.py \
  --data-dir /path/to/dataset.zarr --base-width 32 --sim-type 0 \
  --epochs 500 --batch-size 4 --lr 1e-4 --weight-decay 0.01 \
  --loading-workers 2 --save-root runs/5.4m
```

**21M model:**

```bash
torchrun --standalone --nproc_per_node=8 forward_train.py \
  --data-dir /path/to/dataset.zarr --base-width 64 --sim-type 0 \
  --epochs 500 --batch-size 4 --lr 1e-4 --weight-decay 0.01 \
  --loading-workers 2 --save-root runs/21m
```

- **Matching:** `--sim-type 0` selects input, `1` output, and `2` interstage.
- **One GPU:** replace `torchrun --standalone --nproc_per_node=8` with `python`. Batch size is per GPU.
- **CNN:** add `--simulator-arch cnn` and use a separate save root. The manuscript’s CNN comparison uses `--epochs 100`; the residual architecture is the default.

Training saves the best checkpoint and automatically reports test-set EM and design $R^2$. Outputs are under `<save-root>/forward_model/simtype_<n>/wd_0.01/`.

## Figure 1 study

The solver-stability experiment is in [`studies/`](studies/README.md), with its run command and required inputs.

## GT solver from EM responses

The original GT solver is in `rfic.py` and `data_processing/snp_files/dataProcessing.py`. Supply **denormalized** EM magnitude/phase values and circuit settings.

```python
from rfic import get_s_params_matrix, batch_SM_to_results

# em: torch tensor (B, F, 40), interleaved magnitude/phase in radians
# spec: torch tensor (B, C, F, 3), containing C1 (fF), C2 (fF), frequency (GHz)
S = get_s_params_matrix(em)
results = batch_SM_to_results(S, spec, num_workers=0)  # input matching
```

`results` has shape `(B,C,F,4)`: impedance magnitude (ohms), phase (radians), efficiency (%), and maximum gain (dB). Use the same ascending frequency grid for EM responses and settings, with capacitor values constant across frequency.

For output or interstage matching, call the original circuit solver directly:

```python
import skrf as rf
from data_processing.snp_files.dataProcessing import SP2Impedance, NetworkType

network = rf.Network(frequency=rf.Frequency.from_f(freq_ghz, "ghz"),
                     s=S[0].detach().cpu().numpy(), z0=50, name="em")
perf, reduced = SP2Impedance(network, None, None, None, 50, 80,
                           network_type=NetworkType.OMN)  # C1=50, C2=80 fF
(f, zin), (_, efficiency_percent), (_, gain_linear) = perf
```

Use `NetworkType.IMN`, `OMN`, or `ISM` for input, output, or interstage matching. This direct call returns complex impedance and linear gain; convert gain with `10*log10(gain_linear)` for dB. The default interstage calculation uses the original 50-ohm load; a core network can be supplied through `snp_file_core`. The EM decoder assumes the original layout’s reciprocal/paired-port symmetry.

## Citation and license

```bibtex
@inproceedings{mao2026neuralsim,
  title={End-to-End Neural Modeling of EM Response and Design Performance for Free-Form RFIC Passives},
  author={Mao, Yuhao and Chu, Chenhao and Vechev, Martin and Wang, Hua},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

Code is released under the [MIT license](LICENSE). Dependencies retain their respective licenses. This work is a collaboration between the IDEAS Group and SRI Lab at ETH Zurich.
