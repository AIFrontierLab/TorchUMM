# Show-o

Original unified multimodal model from Show Lab, combining autoregressive and discrete diffusion modeling for understanding and generation.

- **Original repository:** <https://github.com/showlab/Show-o>
- **Paper:** [Show-o: One Single Transformer to Unify Multimodal Understanding and Generation (ICLR 2025)](https://arxiv.org/abs/2408.12528)
- **Backbone key:** `show_o`
- **Capabilities:** Understanding (image-based), Generation (NO Editing, NO text-only understanding)

## Architecture

Show-o uses a single transformer that processes text tokens autoregressively with causal attention and image tokens via discrete denoising diffusion with full attention. Key components:

- **LLM base:** Phi-1.5
- **Visual tokenizer:** MagVITv2 (discrete)
- **Image generation:** Discrete denoising diffusion
- **Max resolution:** 512x512

## Dependencies

The model environment is managed via the `show_o` image defined in `modal/images.py` (Python 3.10, PyTorch 2.2.1, CUDA 12.1, xformers). For local setup, install the dependencies listed in `model/Show-o/requirements.txt`.

## Relationship to Show-o2

Show-o and Show-o2 share the same backbone adapter (`ShowOBackbone`) with version-based branching. The version is auto-detected from the model's `config.json`, or can be explicitly set via `version: 1` in config.

Key differences from Show-o2:

| Aspect | Show-o (v1) | Show-o2 (v2) |
|--------|-------------|--------------|
| LLM base | Phi-1.5 | Qwen2.5 |
| Visual tokenizer | MagVITv2 (discrete) | Wan2.1 3D Causal VAE |
| Generation | Discrete diffusion | Flow matching |
| Video support | No | Yes |
| Text-only understanding | No | Yes |

## Inference

### CLI

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/modal_show_o_generation.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="show_o", backbone_cfg={
    "model_path": "/path/to/show-o-w-clip-vit-512x512",
    "show_o_root": "/path/to/model/Show-o",
    "vq_model_path": "/path/to/magvitv2",
    "version": 1,
    "seed": 42,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="show_o", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding (requires image input)
result = pipeline.run(InferenceRequest(
    backbone="show_o", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))
```

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/modal_dpg_bench_show_o.yaml` |
| GenEval   | `configs/eval/geneval/modal_geneval_show_o.yaml` |
| WISE      | `configs/eval/wise/modal_wise_show_o.yaml` |
| UEval     | `configs/eval/ueval/modal_ueval_show_o.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/modal_uni_mmmu_show_o.yaml` |
| MME       | `configs/eval/mme/modal_mme_show_o.yaml` |
| MMMU      | `configs/eval/mmmu/modal_mmmu_show_o.yaml` |
| MMBench   | `configs/eval/mmbench/modal_mmbench_show_o.yaml` |
| MM-Vet    | `configs/eval/mmvet/modal_mmvet_show_o.yaml` |
| MathVista | `configs/eval/mathvista/modal_mathvista_show_o.yaml` |

## Key Configuration Parameters

- **`model_path`**: Path to Show-o model weights (e.g., `showlab/show-o-w-clip-vit-512x512`)
- **`vq_model_path`**: Path to MagVITv2 discrete tokenizer (e.g., `showlab/magvitv2`)
- **`show_o_root`**: Path to the Show-o repository root
- **`version`**: Set to `1` for Show-o v1 (auto-detected if omitted)

## Modal Commands

```bash
# Download model weights
modal run modal/download.py --model show_o

# Run GenEval
modal run modal/run.py --model show_o --eval-config modal_geneval_show_o

# Run MME
modal run modal/run.py --model show_o --eval-config modal_mme_show_o
```
