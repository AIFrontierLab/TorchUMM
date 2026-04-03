# TokenFlow

Image generation model from ByteFlow AI. Generation only — no understanding or editing.

- **Original repository:** <https://github.com/ByteFlow-AI/TokenFlow>
- **Backbone key:** `tokenflow`
- **Capabilities:** Generation ONLY

## Dependencies

The model environment is managed via the `tokenflow` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/TokenFlow/requirements.txt`.

## Inference

### CLI

```bash
# Text-to-image generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/tokenflow_generation.yaml

# DPG Bench-style generation (dense prompt)
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/tokenflow_dpg_generation.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="tokenflow", backbone_cfg={
    "model_path": "/path/to/tokenflow_weights",
    "tokenizer_path": "/path/to/tokenizer",
    "cfg": 7.5,
    "batch_size": 1,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="tokenflow", task="generation",
    prompt="A cat sitting on a rainbow",
))
```

**Note:** TokenFlow supports generation only. Understanding and editing are not available.

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_tokenflow.yaml` |
| GenEval   | `configs/eval/geneval/geneval_tokenflow.yaml` |
| WISE      | `configs/eval/wise/wise_tokenflow.yaml` |
| UEval     | `configs/eval/ueval/ueval_tokenflow.yaml` |

No understanding benchmarks are supported.

```bash
# Example: run GenEval
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_tokenflow.yaml

# Example: run DPG Bench
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/dpg_bench/dpg_bench_tokenflow.yaml
```

## Key Configuration Parameters

- **Generation:** `cfg` (guidance scale), `loop`, `mixed_precision`, `batch_size`
