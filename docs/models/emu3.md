# Emu3

Multimodal model with separate architectures for understanding and generation.

- **Original repository:** <https://github.com/baaivision/Emu3>
- **Backbone key:** `emu3`
- **Capabilities:** Understanding, Generation (NO Editing)

## Dependencies

The model environment is managed via the `emu3` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/Emu3/requirements.txt`.

### Flash Attention (required)

Emu3 requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.5.7). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Architecture Note

Emu3 uses three separate model components:

- **Emu3-Chat** — understanding (text generation from image+text input)
- **Emu3-Gen** — image generation
- **Emu3-VisionTokenizer** — shared vision tokenizer (VQ-based)

Model weights paths: `emu3/Emu3-Chat`, `emu3/Emu3-Gen`, `emu3/Emu3-VisionTokenizer`.

Generation is VQ-tokenizer based: images are encoded as discrete tokens and generated autoregressively. Emu3 supports per-image timeout via `signal.SIGALRM` and classifier-free guidance.

## Inference

### CLI

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/emu3_generation.yaml

# Understanding
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/emu3_understanding.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="emu3", backbone_cfg={
    "model_path": "/path/to/emu3",     # directory containing Emu3-Chat / Emu3-Gen
    "vq_hub": "/path/to/emu3/Emu3-VisionTokenizer",
    "device": "cuda",
    "torch_dtype": "bfloat16",
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="emu3", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding
result = pipeline.run(InferenceRequest(
    backbone="emu3", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))
```

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_emu3.yaml` |
| GenEval   | `configs/eval/geneval/geneval_emu3.yaml` |
| WISE      | `configs/eval/wise/wise_emu3.yaml` |
| UEval     | `configs/eval/ueval/ueval_emu3.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/uni_mmmu_emu3.yaml` |
| MME       | `configs/eval/mme/mme_emu3.yaml` |
| MMMU      | `configs/eval/mmmu/mmmu_emu3.yaml` |
| MMBench   | `configs/eval/mmbench/mmbench_emu3.yaml` |
| MM-Vet    | `configs/eval/mmvet/mmvet_emu3.yaml` |
| MathVista | `configs/eval/mathvista/mathvista_emu3.yaml` |

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_emu3.yaml

# Example: run MME
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_emu3.yaml
```

## Key Configuration Parameters

- **Generation:** `attn_implementation`, `device_map`, `torch_dtype`
- **Understanding:** standard generation parameters (`max_new_tokens`, `do_sample`)
