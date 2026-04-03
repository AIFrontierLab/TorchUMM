# UniGame Pipeline Integration

This folder contains UMM's `unigame` train pipeline adapter.

## Janus-Pro mode

- Use `train.pipeline: unigame` and `train.backbone: janus_pro`.
- Training runs inside the UMM CLI process.
- All training parameters are provided from `configs/posttrain/unigame.yaml`.
- No standalone UniGame `main.py` entrypoint is used.

## Example config

```yaml
train:
  pipeline: unigame
  backbone: janus_pro
  dataset_path: /path/to/vqav2
  model_path: deepseek-ai/Janus-Pro-7B
  batch_size: 8
  val_batch_size: 8
  num_workers: 4
```

## Run

```bash
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unigame.yaml
```
