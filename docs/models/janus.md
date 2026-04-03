# Janus

Original unified multimodal model from DeepSeek with decoupled visual encoding for understanding and generation.

- **Original repository:** <https://github.com/deepseek-ai/Janus>
- **Paper:** [Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation](https://arxiv.org/abs/2410.13848)
- **Backbone key:** `janus_pro`
- **Capabilities:** Understanding, Generation (NO Editing)

## Architecture

Janus uses separate vision encoders for understanding and generation tasks, unified through a shared LLM backbone. Key design:

- **Understanding encoder:** SigLIP vision encoder
- **Generation:** Autoregressive VQ token prediction (576 discrete tokens per image)
- **LLM base:** DeepSeek-LLM-1.3B

## Supported Variants

| Variant | HuggingFace | Parameters |
|---------|-------------|------------|
| Janus-1.3B | `deepseek-ai/Janus-1.3B` | 1.3B |

## Relationship to Janus-Pro and JanusFlow

All three models share the same repository but differ in architecture and scale:

| Aspect | Janus | Janus-Pro | JanusFlow |
|--------|-------|-----------|-----------|
| Generation method | VQ autoregressive | VQ autoregressive | Rectified flow ODE |
| Parameters | 1.3B | 7B | 1.3B |
| Image tokens | 576 discrete | 576 discrete | Continuous (30 steps) |
| External VAE | No | No | SDXL VAE |

Janus and Janus-Pro share the same backbone adapter (`janus_pro`). Switch between them by changing `model_path`.

## Dependencies

The model environment is managed via the `janus_pro` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/Janus/requirements.txt`.

## Inference

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="janus_pro", backbone_cfg={
    "model_path": "/path/to/Janus-1.3B",
    "janus_root": "/path/to/model/Janus",
    "seed": 42,
    "torch_dtype": "bfloat16",
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="janus_pro", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding
result = pipeline.run(InferenceRequest(
    backbone="janus_pro", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))
```

## Supported Benchmarks

Same configs as Janus-Pro — change `model_path` to `Janus-1.3B`:

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_janus_pro.yaml` |
| GenEval   | `configs/eval/geneval/geneval_janus_pro.yaml` |
| WISE      | `configs/eval/wise/wise_janus_pro.yaml` |
| UEval     | `configs/eval/ueval/ueval_janus_pro.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/uni_mmmu_janus_pro.yaml` |
| MME       | `configs/eval/mme/mme_janus_pro.yaml` |
| MMMU      | `configs/eval/mmmu/mmmu_janus_pro.yaml` |
| MMBench   | `configs/eval/mmbench/mmbench_janus_pro.yaml` |
| MM-Vet    | `configs/eval/mmvet/mmvet_janus_pro.yaml` |
| MathVista | `configs/eval/mathvista/mathvista_janus_pro.yaml` |

## Key Configuration Parameters

- **Generation:** `seed`, `torch_dtype` (cfg_weight=5.0, parallel_size=4 are model defaults)
- **Understanding:** uses `VLChatProcessor` for image preprocessing
