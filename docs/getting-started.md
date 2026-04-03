# Getting Started

This guide walks through installation, basic inference, evaluation, and post-training with TorchUMM.

---

## Installation

```bash
# Clone the repository (including model submodules)
git clone --recursive https://github.com/ApiaoSamaa/umm_codebase.git
cd umm_codebase

# Install the core package
pip install -e .
```

!!! note "Per-model dependencies"
    Each backbone model has its own Python and PyTorch version requirements. Install only the dependencies for the model(s) you plan to use:

    ```bash
    # Example: install Bagel dependencies
    pip install -r model/Bagel/requirements.txt
    ```

    For cloud execution via [Modal](cloud.md), each model runs in an isolated container image with the correct environment --- no local dependency conflicts.

---

## CLI Usage

### Inference

```bash
# Text-to-image generation with Bagel
PYTHONPATH=src python -m umm.cli.main infer \
    --config configs/inference/modal_bagel_generation.yaml

# Image understanding with Janus-Pro
PYTHONPATH=src python -m umm.cli.main infer \
    --config configs/inference/janus_pro_understanding.yaml
```

### Evaluation

```bash
# Single-stage benchmark (DPG Bench on Bagel)
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/dpg_bench/dpg_bench_bagel.yaml

# Two-stage benchmark (GenEval on Bagel --- generation + scoring)
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/geneval/geneval_bagel.yaml
```

### Post-Training

```bash
# SFT on Bagel
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/bagel_sft.yaml
```

---

## Python API

TorchUMM exposes a programmatic interface through `InferencePipeline` and `InferenceRequest`.

```python
from umm.inference.pipeline import InferencePipeline
from umm.inference.multimodal_inputs import InferenceRequest

# Initialize the pipeline with a backbone model
pipeline = InferencePipeline(
    backbone_name="bagel",
    backbone_cfg={
        "model_path": "/path/to/BAGEL-7B-MoT",
        "max_mem_per_gpu": "80GiB",
        "seed": 42,
    },
)

# Text-to-image generation
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="generation",
    prompt="A cat sitting on a rainbow",
    params={"num_timesteps": 50},
))

# Image understanding
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="understanding",
    prompt="Describe this image in detail.",
    images=["path/to/image.jpg"],
    params={"max_think_token_n": 500, "do_sample": False},
))

# Image editing
result = pipeline.run(InferenceRequest(
    backbone="bagel",
    task="editing",
    prompt="Make the sky purple",
    images=["path/to/image.jpg"],
    params={"num_timesteps": 25},
))

# Batch inference
results = pipeline.run_many(
    [request1, request2, request3],
    batch_size=2,
)
```

### InferenceRequest Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `backbone` | `str` | Backbone model name (must match the pipeline) |
| `task` | `str` | `"generation"`, `"understanding"`, or `"editing"` |
| `prompt` | `str` | Text prompt |
| `images` | `list[str]` | Input image paths (for understanding/editing) |
| `videos` | `list[str]` | Input video paths |
| `params` | `dict` | Task-specific parameters |
| `output_path` | `str` | Path to save output |

---

## Cloud Execution

TorchUMM integrates with [Modal](https://modal.com) for cloud GPU execution. This handles environment isolation, model weight caching, and GPU scaling automatically.

```bash
# Download model weights to cloud storage (one-time)
modal run modal/download.py --model bagel

# Run evaluation on cloud GPU
modal run modal/run.py --model bagel --eval-config modal_dpg_bench_bagel
```

See the [Cloud (Modal)](cloud.md) page for setup and full command reference.

---

## AMD HPC Execution

For AMD ROCm clusters, use `amd_` prefixed eval configs with the local runner:

```bash
# Setup environment (one-time)
bash scripts/amd_migration/setup_all_envs.sh bagel

# Run evaluation
bash scripts/amd_migration/local_run.sh bagel --eval-config amd_ueval_bagel
```

Config naming: `modal_*.yaml` (cloud), `amd_*.yaml` (AMD HPC), `*.yaml` (legacy local). To regenerate AMD configs: `python scripts/generate_amd_configs.py`.
