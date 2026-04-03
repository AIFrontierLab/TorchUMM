# OmniGen2

Unified multimodal model supporting understanding, generation, and editing.

- **Original repository:** <https://github.com/VectorSpaceLab/OmniGen2>
- **Backbone key:** `omnigen2`
- **Capabilities:** Understanding, Generation, Editing

## Dependencies

The model environment is managed via the `omnigen2` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/OmniGen2/requirements.txt`.

### Flash Attention (recommended)

OmniGen2 supports [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.7.4) for faster inference (Flash2Varlen attention + fused SwiGLU). Without it, the model falls back to standard attention — functional but noticeably slower. The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Inference

### CLI

```bash
# Generation (text-to-image)
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/omnigen2_generation.yaml

# Understanding
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/omnigen2_understanding.yaml

# Editing
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/omnigen2_editing.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="omnigen2", backbone_cfg={
    "model_path": "/path/to/omnigen2_weights",
    "seed": 42,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="omnigen2", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding (requires image input)
result = pipeline.run(InferenceRequest(
    backbone="omnigen2", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))

# Editing
result = pipeline.run(InferenceRequest(
    backbone="omnigen2", task="editing",
    prompt="Change the background to a beach",
    images=["path/to/image.jpg"],
))
```

**Note:** OmniGen2 uses two internal pipelines: `OmniGen2Pipeline` for generation/editing and `OmniGen2ChatPipeline` for understanding. Both are loaded lazily on first use.

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_omnigen2.yaml` |
| GenEval   | `configs/eval/geneval/geneval_omnigen2.yaml` |
| WISE      | `configs/eval/wise/wise_omnigen2.yaml` |
| GEdit-Bench | `configs/eval/gedit/modal_gedit_omnigen2.yaml` |
| UEval     | `configs/eval/ueval/ueval_omnigen2.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/uni_mmmu_omnigen2.yaml` |
| MME       | `configs/eval/mme/mme_omnigen2.yaml` |
| MMMU      | `configs/eval/mmmu/mmmu_omnigen2.yaml` |
| MMBench   | `configs/eval/mmbench/mmbench_omnigen2.yaml` |
| MM-Vet    | `configs/eval/mmvet/mmvet_omnigen2.yaml` |
| MathVista | `configs/eval/mathvista/mathvista_omnigen2.yaml` |

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_omnigen2.yaml

# Example: run MME
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_omnigen2.yaml
```

## Key Configuration Parameters

- **Generation / Editing:** `enable_cpu_offload`, `enable_sequential_cpu_offload`, `torch_dtype`
- **Understanding:** standard VLM generation params

## Known Issues & Fixes

### GPU OOM from Duplicate Pipeline Loading

**Problem:** OmniGen2 uses two internal pipelines — `OmniGen2Pipeline` (generation/editing) and `OmniGen2ChatPipeline` (understanding). Both share the same underlying components (Qwen2.5-VL mllm, transformer, VAE, scheduler, processor). When loaded independently via `from_pretrained`, the model weights are duplicated in GPU memory, easily causing OOM on multi-task benchmarks like Uni-MMMU that require both understanding and generation in a single run.

**Fix:** The adapter now loads model weights once, then constructs the second pipeline from shared component references. Both pipelines point to the same GPU tensors — zero additional memory overhead.

### Understanding Returns `<|img|>` Instead of Text

**Problem:** When `OmniGen2ChatPipeline.__call__` is used for understanding tasks, its internal system prompt is `"You are a helpful assistant that generates high-quality images..."`, which biases the model toward image generation. For complex prompts (e.g., Uni-MMMU jigsaw/maze tasks that mention "generate images"), the model immediately outputs the `<|img|>` special token, which triggers the stop string — resulting in no text reasoning at all.

**Fix:** The adapter bypasses `chat_pipeline.__call__` and calls `mllm.generate` directly with a text-analysis-focused system prompt: `"You are a helpful assistant that analyzes images and provides detailed text descriptions and reasoning."` Additionally, `<|img|>` is removed from the stop strings so that any reasoning text following the token is preserved, and `max_new_tokens` is configurable (default 512 vs. the hardcoded 256).

### Editing Failures Block Generation Fallback

**Problem:** When `edit()` catches pipeline exceptions internally and returns an error dict, the upstream `generate_image_from_context` function in the evaluation pipeline cannot distinguish this from a successful result. The fallback to pure text-to-image generation (which would succeed) is never triggered.

**Fix:** The `edit()` method no longer catches pipeline exceptions. Errors propagate to the caller, which handles the fallback correctly. The `generate()` method (the final fallback) still catches exceptions and returns structured error dicts — this is correct since there is nowhere further to fall back.
