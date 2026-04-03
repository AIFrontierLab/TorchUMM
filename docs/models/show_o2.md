# Show-o2

Unified multimodal model from Show Lab supporting understanding and generation.

- **Original repository:** <https://github.com/showlab/Show-o>
- **Backbone key:** `show_o2`
- **Capabilities:** Understanding, Generation (NO Editing)

## Dependencies

The model environment is managed via the `show_o2` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/Show-o/requirements.txt`.

### Flash Attention (required)

Show-o2 requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.7.4). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Architecture Note

Show-o2 uses subprocess-based inference (wrapping the original Show-o scripts). The backbone key is `show_o2`, but inference config filenames use the prefix `show_o_` (matching the repo directory name).

Version is auto-detected from the model's `config.json` — models containing "Showo2" in their config are treated as Show-o2.

## Inference

### CLI

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/show_o2_generation.yaml

# Understanding
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/show_o2_understanding.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="show_o2", backbone_cfg={
    "model_path": "/path/to/show_o2_weights",
    "show_o_root": "/path/to/model/Show-o",
    "vae_path": "/path/to/Wan2.1_VAE.pth",
    "seed": 42,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="show_o2", task="generation",
    prompt="A cat sitting on a rainbow",
))

# Understanding (requires image input)
result = pipeline.run(InferenceRequest(
    backbone="show_o2", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
))
```

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_show_o2.yaml` |
| GenEval   | `configs/eval/geneval/geneval_show_o2.yaml` |
| WISE      | `configs/eval/wise/wise_show_o2.yaml` |
| UEval     | `configs/eval/ueval/ueval_show_o2.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/uni_mmmu_show_o2.yaml` |
| MME       | `configs/eval/mme/mme_show_o2.yaml` |
| MMMU      | `configs/eval/mmmu/mmmu_show_o2.yaml` |
| MMBench   | `configs/eval/mmbench/mmbench_show_o2.yaml` |
| MM-Vet    | `configs/eval/mmvet/mmvet_show_o2.yaml` |
| MathVista | `configs/eval/mathvista/mathvista_show_o2.yaml` |

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_show_o2.yaml

# Example: run MME
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_show_o2.yaml
```

## Key Configuration Parameters

- **Generation:** `seed`, `torch_dtype`, `vae_path`
- **Understanding:** subprocess-based, configured via `show_o_root`
