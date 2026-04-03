# DeepGen

Unified multimodal model (5B: 3B VLM + 2B DiT) supporting text-to-image generation and image editing via a diffusers-compatible pipeline.

- **Original repository:** <https://github.com/deepgenteam/DeepGen>
- **Backbone key:** `deepgen`
- **Capabilities:** Generation, Editing

## Dependencies

The model environment is managed via the `deepgen` image defined in `modal/images.py` (Python 3.10, torch 2.8.0, diffusers 0.35.2). For local setup, install the dependencies listed in `model/deepgen/requirements.txt`.

### Flash Attention (recommended)

DeepGen benefits from [Flash Attention](https://github.com/Dao-AILab/flash-attention) for faster inference. The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Inference

### CLI

```bash
# Generation (text-to-image)
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/deepgen_generation.yaml

# Editing (image-to-image)
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/deepgen_editing.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="deepgen", backbone_cfg={
    "model_path": "/model_cache/deepgen/DeepGen-1.0-diffusers",
    "seed": 42,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="deepgen", task="generation",
    prompt="A cat sitting on a rainbow",
    params={"num_inference_steps": 50, "guidance_scale": 4.0},
))

# Editing
result = pipeline.run(InferenceRequest(
    backbone="deepgen", task="editing",
    prompt="Change the background to a beach",
    images=["path/to/image.jpg"],
    params={"num_inference_steps": 50, "guidance_scale": 4.0},
))
```

**Note:** DeepGen does **NOT** support image understanding (VQA). Although its architecture includes a 3B VLM (Qwen2.5 VL-3B), this component is used internally to provide semantic guidance for generation via Stacked Channel Bridging (SCB) — it does not expose a standalone understanding interface. As a result, DeepGen cannot run benchmarks that require understanding capabilities (e.g., Uni-MMMU, UEval, MME, MMMU).

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| GenEval   | `configs/eval/geneval/modal_geneval_deepgen.yaml` |
| DPG-Bench | `configs/eval/dpg_bench/modal_dpg_bench_deepgen.yaml` |
| WISE      | `configs/eval/wise/modal_wise_deepgen.yaml` |
| GEdit-Bench | `configs/eval/gedit/modal_gedit_deepgen.yaml` |

> **Not supported:** Uni-MMMU, UEval, and all VLM understanding benchmarks — these require image understanding capabilities that DeepGen does not provide.

```bash
# Example: run GenEval on Modal
modal run modal/run.py --model deepgen --eval-config modal_geneval_deepgen

# Example: run WISE on Modal
modal run modal/run.py --model deepgen --eval-config modal_wise_deepgen
```

## Key Configuration Parameters

All evaluation parameters follow the [official DeepGen repo](https://github.com/deepgenteam/DeepGen) (`EVAL.md`).

| Parameter | Default | Notes |
|-----------|---------|-------|
| `height` / `width` | 512 | All benchmarks use 512x512 per official EVAL.md |
| `num_inference_steps` | 50 | |
| `guidance_scale` | 4.0 | Exception: DPG-Bench uses 7.5 (per official `dpg_bench.py`) |
| `seed` | 42 | |
| `negative_prompt` | (see adapter) | Editing only |
