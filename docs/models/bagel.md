# Bagel (BAGEL-7B-MoT)

Mixture-of-Transformer multimodal model supporting understanding, generation, and editing.

- **Original repository:** [https://github.com/jpthu17/Bagel](https://github.com/bytedance-seed/BAGEL)
- **Backbone key:** `bagel`
- **Model weights:** `BAGEL-7B-MoT` (HuggingFace)
- **Capabilities:** Understanding, Generation, Editing

## Dependencies

The model environment is managed via the `bagel` image defined in `modal/images.py`. For local setup, install the dependencies listed in `model/Bagel/requirements.txt`. The config expects `bagel_root` to point to `model/Bagel`.

### Flash Attention (required)

Bagel requires [Flash Attention](https://github.com/Dao-AILab/flash-attention) (v2.5.8). The Modal image already includes it. For local setup, install a pre-compiled wheel matching your environment — see [modal/README.md](../../modal/README.md#flash-attention-setup) for the exact environment parameters and installation instructions.

## Inference

### CLI

The inference configs below are pre-configured for Modal (cloud) paths. For local execution, copy the config and adjust `model_path` and `bagel_root` to your local paths.

```bash
# Generation
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/modal_bagel_generation.yaml

# Understanding
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/modal_bagel_understanding.yaml

# Editing
PYTHONPATH=src python -m umm.cli.main infer --config configs/inference/modal_bagel_editing.yaml
```

### Python API

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

pipeline = InferencePipeline(backbone_name="bagel", backbone_cfg={
    "model_path": "/path/to/BAGEL-7B-MoT",
    "bagel_root": "/path/to/model/Bagel",
    "max_mem_per_gpu": "80GiB",
    "seed": 42,
})

# Generation
result = pipeline.run(InferenceRequest(
    backbone="bagel", task="generation",
    prompt="A cat sitting on a rainbow",
    params={"cfg_text_scale": 7.5, "cfg_img_scale": 1.5, "num_timesteps": 50},
))

# Understanding
result = pipeline.run(InferenceRequest(
    backbone="bagel", task="understanding",
    prompt="Describe this image",
    images=["path/to/image.jpg"],
    params={"max_think_token_n": 500, "do_sample": False},
))

# Editing
result = pipeline.run(InferenceRequest(
    backbone="bagel", task="editing",
    prompt="Make the sky purple",
    images=["path/to/image.jpg"],
    params={"cfg_text_scale": 7.5, "cfg_img_scale": 1.5, "num_timesteps": 50},
))
```

## Supported Benchmarks

| Benchmark | Config |
|-----------|--------|
| DPG Bench | `configs/eval/dpg_bench/dpg_bench_bagel.yaml` |
| GenEval   | `configs/eval/geneval/geneval_bagel.yaml` |
| WISE      | `configs/eval/wise/wise_bagel.yaml` |
| GEdit-Bench | `configs/eval/gedit/modal_gedit_bagel.yaml` |
| UEval     | `configs/eval/ueval/ueval_bagel.yaml` |
| Uni-MMMU  | `configs/eval/uni_mmmu/uni_mmmu_bagel.yaml` |
| MME       | `configs/eval/mme/mme_bagel.yaml` |
| MMMU      | `configs/eval/mmmu/mmmu_bagel.yaml` |
| MMBench   | `configs/eval/mmbench/mmbench_bagel.yaml` |
| MM-Vet    | `configs/eval/mmvet/mmvet_bagel.yaml` |
| MathVista | `configs/eval/mathvista/mathvista_bagel.yaml` |

```bash
# Example: run GenEval (two-stage, handled automatically)
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/geneval/geneval_bagel.yaml

# Example: run MME
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/mme/mme_bagel.yaml
```

## Key Configuration Parameters

- **Generation / Editing:** `cfg_text_scale`, `cfg_img_scale`, `num_timesteps`
- **Understanding:** `max_think_token_n`, `do_sample`
- **Post-training:** SFT, recA, IRG, UniCot (see `configs/posttrain/`)
