# Post-Training Methods

TorchUMM supports multiple post-training strategies for fine-tuning multimodal models (currently targeting Bagel). All methods are configured via YAML files and launched through the CLI.

---

## Supported Methods

| Method | Description | Config |
| :--- | :--- | :--- |
| **SFT** | Supervised fine-tuning on task-specific data | `configs/posttrain/bagel_sft.yaml` |
| **IRG** | Interleaved Reasoning Generation — 2-stage curriculum training | `configs/posttrain/irg_stage1.yaml` / `irg_stage2.yaml` |
| **recA** | Reconstruction Alignment — trains generation with alignment signal | `configs/posttrain/recA.yaml` |
| **UniCot** | Unified Chain-of-Thought training using LoRA | `configs/posttrain/unicot.yaml` |

!!! note "LoRA"
    LoRA is used **internally** by UniCot (lora_rank=256, lora_alpha=512) and is not a standalone training method.

---

## Usage

### CLI

```bash
# SFT on Bagel
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/bagel_sft.yaml

# IRG Stage 1
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/irg_stage1.yaml

# IRG Stage 2
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/irg_stage2.yaml

# recA
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/recA.yaml

# UniCot (LoRA-based)
PYTHONPATH=src python -m umm.cli.main train \
    --config configs/posttrain/unicot.yaml
```

### Config Structure

Post-training configs follow this structure:

```yaml
train:
  pipeline: bagel          # selects the training pipeline (bagel | recA | unicot | irg)
  cwd: src/umm/post_training/sft/bagel
  torchrun:
    nnodes: 1
    nproc_per_node: 4      # number of GPUs
  script: train/pretrain_unified_navit.py
  env:
    PYTHONPATH: .
  args:
    model_path: /model_cache/bagel/BAGEL-7B-MoT
    lr: 2e-5
    save_every: 1000
    results_dir: /checkpoints/bagel_sft
```

Key fields:

| Field | Description |
| :--- | :--- |
| `pipeline` | Selects the training dispatcher (`bagel`, `recA`, `unicot`, `irg`) |
| `cwd` | Working directory for the training script |
| `torchrun` | Multi-GPU launch params (`nnodes`, `nproc_per_node`) |
| `script` | Training script path (relative to `cwd`) |
| `env` | Environment variables (e.g., `PYTHONPATH`) |
| `args` | Training hyperparameters forwarded as CLI flags |

---

## Pipeline Dispatch Logic

The CLI routes to the correct training pipeline based on the `pipeline` field:

| `pipeline` value | Handler |
| :--- | :--- |
| `bagel` | `umm.post_training.sft.bagel.pipeline.run_bagel_train` |
| `recA` | `umm.post_training.recA.pipeline.run_reca_train` |
| `unicot` | `umm.post_training.unicot.pipeline.run_unicot_train` |
| `irg` | `umm.post_training.IRG.pipeline.run_irg_train` |

Each pipeline translates the config dict into a `torchrun` or `python` subprocess and executes the training script inside the corresponding model repo directory.

---

## Cloud Post-Training

For cloud GPU execution via Modal:

```bash
modal run modal/train.py --config bagel_sft
```

See the [Cloud (Modal)](cloud.md) page for setup and additional details.
