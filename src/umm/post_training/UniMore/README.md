# UniMore

UniMore adapts BAGEL through two branches trained independently from the same
pretrained model:

- **Generation:** a grounded phrase supervises its image crop with the native
  image-generation objective.
- **Understanding:** a full image supervises its complete caption with positive
  cross-entropy and full-caption ranking. Negative captions omit a detail, alter
  a detail, or come from another image.

The exported branches are combined once:

```text
W = W_base + ((W_t2i - W_base) + 0.02 * (W_i2t - W_base))
```

The merge performs the differences, weighting, sum and residual addition in
BF16, matching the released checkpoint. Casting intermediate calculations to
FP32 changes rounding and does not reproduce the same weights.

## Integration

Everything specific to UniMore lives in this directory. No existing TorchUMM
file, CLI registration, configuration or other post-training method is changed.
Use `python -m umm.post_training.UniMore`; the existing `umm` commands are unchanged.
Inference reuses TorchUMM's BAGEL adapter and pipeline. The supplied configs point
to the included BAGEL runtime. Standard benchmarks use TorchUMM's evaluators;
RealUnify's additional entry point is local to this package.

| Path | Purpose |
|---|---|
| `cli.py`, `__main__.py` | Method command-line entry point |
| `training.py` | Launch one independent branch with torchrun |
| `data.py` | Validate and relocate the fixed training manifests |
| `checkpoints.py` | Download, LoRA export and fixed BF16 merge |
| `config.py` | TorchUMM config loading with required-path validation |
| `evaluation.py`, `scripts/` | Benchmark dispatch and RealUnify tools |
| `configs/train/` | One final training preset per branch |
| `configs/data/` | The corresponding fixed data definitions |
| `configs/inference.yaml` | One inference example for the final model |
| `third_party/` | Licensed BAGEL training and inference runtimes |
| `tests/` | Numerical, data and integration regression tests |

## Setup

Run commands from the TorchUMM repository root, using Python 3.10 and CUDA-enabled
PyTorch. Keep weights, datasets, caches and outputs outside the source tree.

```bash
export UNIMORE=src/umm/post_training/UniMore
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
pip install -e .
```

Use `requirements/train.txt` in this directory for training, and
`requirements/inference.txt` in a separate environment for BAGEL inference.
Install `flash-attn==2.7.4.post1` against the corresponding PyTorch/CUDA build.
Set the storage and model paths listed in [.env.example](.env.example).
Only these five configuration files are included; they all belong to the same
final recipe. Benchmark-specific configurations remain in TorchUMM.
`UNIMORE_ROOT` and `TORCHUMM_ROOT` are derived automatically by the method entry
point. Benchmark scorers require their TorchUMM dependencies and judge weights.
Run MMBench/MathVista extraction in a Qwen3-compatible environment, separately
from the Transformers4.49 BAGEL inference environment.

## Released Model

```bash
python -m umm.post_training.UniMore download --model unimore --output "$UNIMORE_MODEL"
python -m umm.post_training.UniMore infer --config "$UNIMORE/configs/inference.yaml"
```

The final model and both branches are available in
[HuggingSelf/test0519](https://huggingface.co/HuggingSelf/test0519).
[Checkpoints](docs/checkpoints.md) lists pinned revisions, paths and checksums.

## Training

Prepare the fixed manifests and image assets following [Data](docs/data.md).
Both branches start from the original BAGEL weights, not from a merged model.

```bash
python -m umm.post_training.UniMore download --model base --output "$BAGEL_MODEL_PATH"
python -m umm.post_training.UniMore train --config "$UNIMORE/configs/train/t2i.yaml"
python -m umm.post_training.UniMore train --config "$UNIMORE/configs/train/i2t.yaml"
```

Add `--dry-run` to inspect a launch command without creating outputs or starting
training. Launch actual training through your compute scheduler.

| Setting | Generation | Understanding |
|---|---:|---:|
| Records | 2,460 | 4,000 |
| Optimizer updates | 1,000 | 600 |
| Learning rate | 5e-5 | 1.8e-5 |
| Warmup updates | 50 | 60 |
| Image size | 256 | 112 |
| Distributed ranks | 4 | 2 |
| Packed-token target / limit | 256 / 512 | 768 / 1,536 |

Both branches use LoRA rank16, alpha32, dropout0.05, AdamW, a constant learning
rate after warmup, training seed4396 and data seed42. Understanding uses three
full-caption negatives and ranking margin0.2. The released data order, packing,
negative sampling and loss reductions are retained.

## Export and Merge

```bash
python -m umm.post_training.UniMore export --base "$BAGEL_MODEL_PATH" --route t2i \
  --checkpoint "$UNIMORE_WORK/training/t2i/checkpoints/0000999/model.safetensors" \
  --output "$UNIMORE_WORK/models/t2i"
python -m umm.post_training.UniMore export --base "$BAGEL_MODEL_PATH" --route i2t \
  --checkpoint "$UNIMORE_WORK/training/i2t/checkpoints/0000599/model.safetensors" \
  --output "$UNIMORE_WORK/models/i2t"
python -m umm.post_training.UniMore merge --base "$BAGEL_MODEL_PATH" \
  --t2i "$UNIMORE_WORK/models/t2i" --i2t "$UNIMORE_WORK/models/i2t" \
  --output "$UNIMORE_WORK/models/unimore"
```

Export preserves the learned generation-side time/latent projections and the
understanding-side connector as well as the language-model LoRA updates.
Export and merge run on CPU; allow approximately 180GB host memory. Output
directories must be empty. The merge accepts BF16 exported checkpoints and has
no search or parameter-sweep interface.

## Evaluation

Reuse TorchUMM's existing benchmark configurations and point them at the exported
model. Keep outputs separate from the base-model results. For example:

```bash
python -m umm.cli.main eval --config configs/eval/mmmu/mmmu_bagel.yaml \
  --set "inference.backbone_cfg.model_path=$UNIMORE_MODEL" \
  --set "inference.backbone_cfg.bagel_root=$PWD/$UNIMORE/third_party/bagel" \
  --set "inference.backbone_cfg.offload_folder=$UNIMORE_WORK/offload" \
  --set "mmmu.cache_dir=$UNIMORE_WORK/cache/mmmu" \
  --set "mmmu.out_dir=$UNIMORE_WORK/evaluation/mmmu" \
  --set "mmmu.score_output_path=$UNIMORE_WORK/evaluation/mmmu/summary.json"
```

For generation, apply the settings in `configs/inference.yaml`: 512px, 50 steps,
text/image CFG4/1, timestep shift3 and seed42. Do not assume that a benchmark's
base-model defaults have the same resolution. Follow TorchUMM's benchmark
instructions for data preparation and separate scoring environments. The
UniMore-local RealUnify tools expose their arguments with `--help`.

## Tests

```bash
python -B -m pytest -p no:cacheprovider "$UNIMORE/tests"
```

The tests cover negative captions, packed-loss values and gradients, unchanged
model math, LoRA export, BF16 rounding, config paths and TorchUMM delegation.
To verify all tensors against a downloaded released model without writing a new
checkpoint, run on a CPU compute node:

```bash
python -B -m umm.post_training.UniMore.tests.verify_checkpoint \
  --base "$BAGEL_MODEL_PATH" --t2i "$UNIMORE_WORK/models/t2i" \
  --i2t "$UNIMORE_WORK/models/i2t" --released "$UNIMORE_MODEL"
```

## Acknowledgments

Built on BAGEL and TorchUMM. Upstream copyright and Apache-2.0 license notices
are retained under `third_party/`. Only the final UniMore method is included;
there are no experiment-search controllers or machine-specific job scripts.
