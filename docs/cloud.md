# Cloud (Modal)

TorchUMM integrates with [Modal](https://modal.com) for running inference, evaluation, and training on cloud GPUs. Each model runs in an isolated container image with the correct Python, PyTorch, and dependency versions --- no local environment conflicts.

!!! info "Full documentation"
    This page provides a quick overview. For the complete Modal infrastructure reference, see [modal/README.md](https://github.com/ApiaoSamaa/umm_codebase/blob/main/modal/README.md) in the repository.

---

## Setup

```bash
# Install Modal
pip install modal

# Login (one-time)
modal setup
```

Create required secrets on the [Modal Dashboard](https://modal.com/secrets):

```bash
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal secret create wandb-secret WANDB_API_KEY=xxx          # for training
```

!!! info "No external APIs needed for scoring"
    UEval and WISE scoring both use local Qwen models as the evaluator — no Gemini or OpenAI API keys required. Download the evaluator weights once before running scoring steps:

    ```bash
    modal run modal/download.py --model evaluator
    ```

---

## Key Commands

### Download Model Weights

```bash
# Download a specific model (one-time per model)
modal run modal/download.py --model bagel

# Download a dataset
modal run modal/download.py --dataset ueval

# Verify what is cached
modal run modal/download.py --ls
```

### Run Evaluation

```bash
# Run an evaluation benchmark
modal run modal/run.py --model bagel --eval-config modal_dpg_bench_bagel

# Specify GPU type
modal run modal/run.py --model omnigen2 --gpu H100

# Multi-GPU
modal run modal/run.py --model wise --eval-config modal_score_wise_bagel --gpu A100-80GB:2
```

### Run Inference

```bash
modal run modal/run.py --model bagel --script inferencer.py
```

### Run Training

```bash
modal run modal/train.py --config bagel_sft
```

### Sync Code

After modifying code locally, sync it to the cloud volume:

```bash
modal run modal/download.py --sync
```

!!! tip "Minimal sync"
    For small changes, consider syncing individual files rather than the entire codebase. Use `modal run modal/download.py --sync` only when many files have changed.

---

## Architecture

Modal infrastructure is organized in `modal/`:

| File | Purpose |
| :--- | :--- |
| `config.py` | Constants: volume names, HF model/dataset IDs, paths |
| `volumes.py` | Persistent volume definitions (models, datasets, checkpoints, outputs, codebase) |
| `images.py` | Container image definitions per backbone model |
| `download.py` | Download model weights and datasets to volumes |
| `run.py` | Unified entry point for inference and evaluation |
| `train.py` | Post-training entry point |

### Volumes

| Volume | Name | Container Mount Path |
| :--- | :--- | :--- |
| Codebase | `umm-codebase` | `/workspace` |
| Model weights | `umm-model-cache` | `/model_cache` |
| Post-train weights | `umm-post-train-model-cache` | `/post_train_model_cache` |
| Datasets | `umm-datasets-cache` | `/datasets` |
| Checkpoints | `umm-checkpoints` | `/checkpoints` |
| Outputs | `umm-outputs` | `/outputs` |

!!! warning "Path convention"
    Eval config files use **container mount paths** (e.g., `/outputs/ueval/bagel`), not volume names or local paths.
