# Figure1 solver stability

## Usage

From the repository root, with the root requirements installed:

```sh
python studies/rfic_instability_inspection.py \
  --data-dir /path/to/original-converted.zarr \
  --forward-model-path /path/to/original/model.pth \
  --save-dir /path/to/study-output \
  --sim-type 0 --batch-size 2 --loading-workers 1 \
  --max-batches 25 --std-batches 125 --seed 1 --rfic-num-workers 8
```