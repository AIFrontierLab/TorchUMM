# Text-to-image extension

This directory is an opt-in TorchUMM extension. It contains the shared adapter,
model declarations, example configs, and smoke tests without changing any file
in TorchUMM's core `src/` tree.

The extension does not register models on import. Register them explicitly:

```python
from extensions.text2image import register

register()
```

Run a config through the dedicated entry point from the repository root:

```bash
PYTHONPATH=src:. python -m extensions.text2image.infer \
  --config extensions/text2image/configs/flux1_schnell_generation.yaml
```

Model loading is lazy. Set `local_files_only: true` for offline checks, or
`ephemeral_cache: true` to put downloaded weights in a temporary directory that
is deleted when inference finishes. Smoke tests inject fake pipelines and do not
download checkpoints:

```bash
PYTHONPATH=src:. pytest -q extensions/text2image/tests
```
