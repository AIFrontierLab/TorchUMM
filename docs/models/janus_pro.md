# Janus-Pro

Scaled-up multimodal model from DeepSeek (7B) with improved training and stronger multimodal capabilities.

- **Original repository:** <https://github.com/deepseek-ai/Janus>
- **Paper:** [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811)
- **Backbone key:** `janus_pro`
- **Capabilities:** Understanding, Generation (NO Editing)

## Supported Variants

| Variant | HuggingFace | Parameters |
|---------|-------------|------------|
| Janus-Pro-7B | `deepseek-ai/Janus-Pro-7B` | 7B |

Janus-Pro shares the same backbone adapter (`janus_pro`) and architecture as the original [Janus](janus.md) (1.3B), but at larger scale with better performance. See also [JanusFlow](janus_flow.md) for the rectified flow variant.

## Dependencies

The model environment is managed via the `janus_pro` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/Janus/requirements.txt`.

### Flash Attention (required)

Janus-Pro requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.7.4). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Architecture Note

Janus-Pro uses parallel generation: a single forward pass produces multiple images simultaneously (`parallel_size=4`). Generation uses classifier-free guidance (`cfg_weight=5.0`) with 576 image tokens per image.

## Inference

### CLI

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/janus_pro_generation.yaml

# Understanding
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/janus_pro_understanding.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="janus_pro", backbone_cfg={
    "model_path": "/path/to/janus_pro_weights",
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

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_janus_pro.yaml

# Example: run UEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/ueval/ueval_janus_pro.yaml
```

## Key Configuration Parameters

- **Generation:** `seed`, `torch_dtype` (cfg_weight=5.0 and parallel_size=4 are model defaults)
- **Understanding:** uses `VLChatProcessor` for image preprocessing; returns `text` and `sft_format`
