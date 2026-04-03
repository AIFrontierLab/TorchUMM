# Reproducing Results

This page provides the exact commands needed to reproduce the evaluation results reported in TorchUMM.

!!! tip "Two-stage vs. single-stage"
    Benchmarks with two-stage evaluation (GenEval, WISE, UEval, Uni-MMMU, GEdit-Bench) provide separate `_generate` and `_score` configs. You can also use the base config (mode: `full`) to run both stages in one command. Single-stage benchmarks (DPG Bench, MME, MMMU, MMBench, MM-Vet, MathVista) run generation and scoring together.

---

## Two-Stage Benchmarks

### GenEval on Bagel

GenEval evaluates compositional text-to-image generation. The generation step produces images; the scoring step runs an object detector to evaluate them.

```bash
# Step 1: Generate images
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/geneval/geneval_bagel_generate.yaml

# Step 2: Score generated images
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/geneval/geneval_bagel_score.yaml
```

Config files:

- `configs/eval/geneval/geneval_bagel_generate.yaml` --- generation stage
- `configs/eval/geneval/geneval_bagel_score.yaml` --- scoring stage
- `configs/eval/geneval/geneval_bagel.yaml` --- full pipeline (both stages)

### WISE on Bagel

WISE evaluates world knowledge in image generation. Scoring is performed by Qwen VL models.

```bash
# Step 1: Generate images
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/wise/wise_bagel_generate.yaml

# Step 2: Score with Qwen models
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/wise/wise_bagel_score.yaml
```

Config files:

- `configs/eval/wise/wise_bagel_generate.yaml` --- generation stage
- `configs/eval/wise/wise_bagel_score.yaml` --- scoring stage
- `configs/eval/wise/wise_bagel.yaml` --- full pipeline

### UEval on Bagel

UEval is a unified benchmark that evaluates both understanding and generation capabilities. Scoring is performed by Qwen models.

```bash
# Step 1: Generate text + image answers
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/ueval/ueval_bagel_generate.yaml

# Step 2: Score with Qwen models
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/ueval/ueval_bagel_score.yaml
```

Config files:

- `configs/eval/ueval/ueval_bagel_generate.yaml` --- generation stage
- `configs/eval/ueval/ueval_bagel_score.yaml` --- scoring stage
- `configs/eval/ueval/ueval_bagel.yaml` --- full pipeline

---

## Single-Stage Benchmarks

Single-stage benchmarks run generation and scoring in one command.

### MME on Bagel

```bash
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/mme/mme_bagel.yaml
```

### DPG Bench on Bagel

```bash
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/dpg_bench/dpg_bench_bagel.yaml
```

### MMMU, MMBench, MM-Vet, MathVista

These follow the same single-command pattern:

```bash
# MMMU
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/mmmu/mmmu_bagel.yaml

# MMBench
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/mmbench/mmbench_bagel.yaml

# MM-Vet
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/mmvet/mmvet_bagel.yaml

# MathVista
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/mathvista/mathvista_bagel.yaml
```

---

## Running on Modal

For cloud GPU execution, use the Modal equivalents with `modal_` prefixed configs:

```bash
# GenEval on Bagel (cloud)
modal run modal/run.py --model bagel \
    --eval-config modal_geneval_bagel

# WISE on Bagel (cloud, two-stage is handled automatically)
modal run modal/run.py --model bagel \
    --eval-config modal_wise_bagel

# GEdit-Bench on Bagel (cloud, two-stage editing + scoring)
modal run modal/run.py --model bagel \
    --eval-config modal_gedit_bagel

# DPG Bench on Bagel (cloud)
modal run modal/run.py --model bagel \
    --eval-config modal_dpg_bench_bagel
```

!!! note "Automatic two-stage handling"
    When a Modal eval config specifies a `score_model`, `modal/run.py` automatically detects this and runs both stages sequentially, each in the appropriate model's container image.

See the [Cloud (Modal)](../cloud.md) page for setup instructions.
