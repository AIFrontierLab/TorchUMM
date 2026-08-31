# Vision-language extension

This directory is an opt-in TorchUMM extension. It contains the shared
Transformers adapter, model declarations, example configs, and smoke tests. Each
model has its own module under `integrations/`, config under `configs/`, and
construction smoke test under `tests/`.

Install its dependencies before using it:

```bash
pip install 'transformers>=4.51' 'torch>=2.5' 'accelerate>=1.0' 'Pillow>=10.0'
```

The extension does not register models on import. Register them explicitly:

```python
from extensions.vlm import register

register()
```

Run a config through the dedicated entry point from the repository root:

```bash
PYTHONPATH=src:. python -m extensions.vlm.infer \
  --config extensions/vlm/configs/qwen2_5_vl_7b_understanding.yaml
```

Model loading is lazy: weights are downloaded only when the selected backbone
receives its first understanding request. Set `local_files_only: true` to
require a local cache, or `device_map: auto` to load a model across available
devices. Smoke tests inject fake pipelines and do not download checkpoints:

```bash
PYTHONPATH=src:. pytest -q extensions/vlm/tests
```

## Supported models

| Backbone | Hugging Face checkpoint | Notes |
|---|---|---|
| `qwen2_5_vl_7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | General image understanding |
| `internvl3_8b` | `OpenGVLab/InternVL3-8B` | Requires `trust_remote_code` |

Both expose image understanding through the unified `understanding` interface.
Video inputs are intentionally rejected by this adapter even for checkpoints
whose upstream model cards describe video support.

### Qwen2.5-VL

Use `qwen2_5_vl_7b`. The default chat format is handled by the Transformers
image-to-text pipeline.

### InternVL3

Use `internvl3_8b`. Its upstream implementation uses custom Transformers code,
so `trust_remote_code` is enabled by default.
