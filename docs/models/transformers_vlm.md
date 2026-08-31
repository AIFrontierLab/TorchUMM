# Transformers VLMs

TorchUMM integrates image-to-text VLMs through a shared lazy Transformers
adapter. Install the optional dependencies before using them:

```bash
pip install -e '.[vlm]'
```

Each model has its own module under
`src/umm/backbones/transformers_vlm/integrations/`, config under
`configs/inference/`, and construction smoke test under `tests/`.

Weights are downloaded only when the selected backbone receives its first
understanding request. Set `local_files_only: true` to require a local cache,
or set `device_map: auto` when loading a model across available devices.

## Supported models

| Backbone | Hugging Face checkpoint | Notes |
|---|---|---|
| `qwen2_5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | General image understanding |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Requires `trust_remote_code` |

Both expose image understanding through the unified `understanding` interface.
Video inputs are intentionally rejected by this adapter even for checkpoints
whose upstream model cards describe video support.

## Example

```yaml
inference:
  backbone: qwen2_5_vl_7b
  backbone_cfg:
    model_path: Qwen/Qwen2.5-VL-7B-Instruct
    device: cuda:0
    torch_dtype: bfloat16
  request:
    task: understanding
    prompt: "What is happening in this image?"
    images:
      - assets/torchumm_frame.png
    params:
      max_new_tokens: 256
```

Run it with:

```bash
umm infer --config configs/inference/qwen2_5_vl_7b_understanding.yaml
```

## Model-specific notes

### Qwen2.5-VL

Use `qwen2_5_vl_7b`. The default chat format is handled by the Transformers
image-to-text pipeline.

### InternVL3

Use `internvl3_8b`. Its upstream implementation uses custom Transformers code,
so `trust_remote_code` is enabled by default.
