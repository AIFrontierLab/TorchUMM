# BLIP3-o

Image generation model from Salesforce. Generation only — no understanding or editing.

- **Original repository:** <https://github.com/salesforce/BLIP3o>
- **Backbone key:** `blip3o`
- **Capabilities:** Generation ONLY

## Dependencies

The model environment is managed via the `blip3o` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/BLIP3o/requirements.txt`.

### Flash Attention (required)

BLIP3-o requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.6.2). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Inference

BLIP3-o uses subprocess-based generation (wrapping the original BLIP3-o scripts).

### CLI

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/blip3o_generation.yaml

# DPG Bench-style generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/blip3o_dpg_generation.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="blip3o", backbone_cfg={
    "model_path": "/path/to/blip3o_weights",
    "scale": 7.5,
    "seq_len": 1024,
    "top_p": 0.9,
    "top_k": 50,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="blip3o", task="generation",
    prompt="A cat sitting on a rainbow",
))
```

**Note:** This model supports generation only. Understanding and editing are not available.

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_blip3o.yaml` |
| GenEval   | `configs/eval/geneval/geneval_blip3o.yaml` |
| WISE      | `configs/eval/wise/wise_blip3o.yaml` |
| UEval     | `configs/eval/ueval/ueval_blip3o.yaml` |

No understanding benchmarks are supported.

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_blip3o.yaml

# Example: run DPG Bench
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/dpg_bench/dpg_bench_blip3o.yaml
```

## Key Configuration Parameters

- **Generation:** `scale` (guidance scale), `seq_len`, `top_p`, `top_k`, `output_dir`
