# JanusFlow

Rectified flow variant of the Janus architecture from DeepSeek, using continuous ODE-based generation with an external SDXL VAE.

- **Original repository:** <https://github.com/deepseek-ai/Janus>
- **Backbone key:** `janus_flow`
- **Capabilities:** Understanding, Generation (NO Editing)

## Supported Variants

| Variant | HuggingFace | Parameters |
|---------|-------------|------------|
| JanusFlow-1.3B | `deepseek-ai/JanusFlow-1.3B` | 1.3B |

## Dependencies

The model environment is managed via the `janus_flow` image defined in `modal/images.py`. Key additional dependency compared to Janus-Pro: `diffusers` (for `AutoencoderKL` SDXL VAE).

## Architecture Note

JanusFlow differs from Janus-Pro in its generation approach:

- **Janus-Pro:** Autoregressive VQ tokens (576 discrete tokens per image)
- **JanusFlow:** Rectified flow ODE (30 continuous steps, decoded by external SDXL VAE)

JanusFlow uses `ShallowUViTEncoder`/`ShallowUViTDecoder` for encoding/decoding latents, with the LLM predicting velocity fields at each ODE step. The SDXL VAE (`stabilityai/sdxl-vae`) must be loaded in **bfloat16** (fp16 produces garbage output).

Understanding is nearly identical to Janus-Pro.

## Inference

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="janus_flow", backbone_cfg={
    "model_path": "/path/to/JanusFlow-1.3B",
    "janus_root": "/path/to/Janus",
    "vae_path": "/path/to/sdxl-vae",
    "seed": 42,
    "torch_dtype": "bfloat16",
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="janus_flow", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding
result = pipeline.run(InferenceRequest(
    backbone="janus_flow", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))
```

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/modal_dpg_bench_janus_flow.yaml` |
| GenEval   | `configs/eval/geneval/modal_geneval_janus_flow.yaml` |
| WISE      | `configs/eval/wise/modal_wise_janus_flow.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/modal_uni_mmmu_janus_flow.yaml` |

```bash
# Example: run GenEval on Modal
modal run modal/run.py --model janus_flow --eval-config modal_geneval_janus_flow

# Example: run Uni-MMMU on Modal
modal run modal/run.py --model janus_flow --eval-config modal_uni_mmmu_janus_flow
```

## Key Configuration Parameters

- **Generation:** `cfg_weight=2.0`, `num_inference_steps=30`, `parallel_size=5`, `img_size=384` (configurable to 1024)
- **Understanding:** uses `VLChatProcessor` for image preprocessing; returns `text` and `sft_format`
- **VAE:** must use bfloat16 dtype; loaded separately from model weights
