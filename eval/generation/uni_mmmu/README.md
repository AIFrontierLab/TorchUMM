# Uni-MMMU -- Evaluation Guide

## Overview

Uni-MMMU evaluates unified multimodal models on 8 reasoning-centric tasks spanning generation, understanding, and editing. It measures the bidirectional synergy between these capabilities.

Reference: https://github.com/Vchitect/Uni-MMMU

Paper: https://arxiv.org/abs/2510.13759

## Prerequisites

### 1. Dataset

Download the Uni-MMMU dataset:

```bash
modal run modal/download.py --dataset uni_mmmu
```

This downloads to `/datasets/uni_mmmu/Uni-MMMU-Eval` inside the container.

Dataset structure:
```
/datasets/uni_mmmu/Uni-MMMU-Eval/data/
├── math_data/filtered.json          # Geometry task
├── jigsaw_dataset_2x2ref/           # Jigsaw task (metadata.json + images)
├── science/dim_all.json             # Science reasoning task
├── svg/                             # Code rendering task (metadata.json + PNGs + SVGs)
├── (maze data is generated procedurally)
└── (sliding puzzle data is generated procedurally)
```

### 2. Generation Model

Download the generation backbone (e.g., BAGEL):

```bash
modal run modal/download.py --model bagel
```

Path: `/model_cache/bagel/BAGEL-7B-MoT`

### 3. Scoring Models

Two evaluator models are required for the scoring stage:

| Model | Purpose | Path | Download |
| :--- | :--- | :--- | :--- |
| Qwen2.5-VL-72B-Instruct | Overlay/image judge | `/model_cache/evaluator/Qwen2.5-VL-72B-Instruct` | `modal run modal/download.py --model evaluator` |
| Qwen3-32B | Text reasoning judge | `/model_cache/evaluator/Qwen3-32B` | `modal run modal/download.py --model evaluator` |

### 4. DreamSim (Jigsaw task only)

DreamSim is a perceptual similarity metric used exclusively by the Jigsaw task scorer. Unlike HuggingFace models, DreamSim weights are downloaded automatically by the `dreamsim` pip package (from GitHub releases) on first use.

- **Cache directory**: `/model_cache/dreamsim` (configured via `dreamsim_cache` in YAML config)
- **No manual download needed**: weights are fetched automatically on first run
- **Why DreamSim?** It fuses DINO + CLIP + OpenCLIP features, offering the best alignment with human perceptual similarity judgments compared to LPIPS or SSIM

### 5. CairoSVG (Code rendering task, optional)

The Code rendering task optionally uses `cairosvg` to rasterize SVG files. If not installed, the evaluator gracefully degrades. It is included in the Modal `uni_mmmu` image.

## Evaluation Pipeline

Uni-MMMU uses a two-stage evaluation:

### Stage 1: Generation

The generation model produces outputs for each task (overlay images, text solutions, etc.):

```bash
# Modal (cloud)
modal run modal/run.py --model bagel \
    --eval-config modal_uni_mmmu_bagel

# Local
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/uni_mmmu/uni_mmmu_bagel.yaml
```

Output structure:
```
/outputs/uni_mmmu/bagel/
├── math/{id}/          # model_image_01.png + model_text.txt + result.json
├── jigsaw/{id}/
├── science/{id}/
├── code/{id}/
├── maze/{id}/
└── sliding/{id}/
```

### Stage 2: Scoring

Evaluator models score the generated outputs:

```bash
# Modal (cloud) -- auto-detected when config has score_model
modal run modal/run.py --model bagel \
    --eval-config modal_uni_mmmu_bagel_score

# Local
PYTHONPATH=src python -m umm.cli.main eval \
    --config configs/eval/uni_mmmu/uni_mmmu_bagel_score.yaml
```

Scoring output:
```
/outputs/uni_mmmu/eval/bagel/
├── math/eval_summary.json
├── jigsaw/eval_summary.json
├── science/eval_summary.json
├── code/eval_summary.json
├── maze/eval_summary.json
├── sliding/eval_summary.json
└── overall_summary.json
```

### Task-Specific Configs

Each task can also be run independently:

```bash
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_math_geo
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_jigsaw
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_science
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_code_rendering
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_maze
modal run modal/run.py --model bagel --eval-config modal_uni_mmmu_bagel_sliding
```

## Scoring Metrics

| Task | Image Metric | Text Metric |
| :--- | :--- | :--- |
| Geometry | Overlay accuracy (Qwen2.5-VL judge) | Reasoning rigor + conclusion correctness (Qwen3 judge) |
| Jigsaw | DreamSim perceptual distance | N/A |
| Science | Qwen2.5-VL image quality judge | Qwen3 text accuracy judge |
| Code rendering | Qwen2.5-VL SVG rendering judge | N/A |
| Maze | Path correctness (automated) | N/A |
| Sliding | Tile arrangement correctness (automated) | N/A |

## Config Reference

Configs are in `configs/eval/uni_mmmu/`. Key fields:

```yaml
data_root: /datasets/uni_mmmu/Uni-MMMU-Eval/data
model_path: /model_cache/bagel/BAGEL-7B-MoT

# Generation parameters (BAGEL official defaults)
generation_cfg:
  num_timesteps: 50
  cfg_text_scale: 4.0
  cfg_img_scale: 1.0

# Editing parameters
editing_cfg:
  num_timesteps: 50
  cfg_img_scale: 2.0

# Understanding parameters
understanding_cfg:
  max_think_token_n: 1000
  do_sample: false
  think: false

# Scoring
dreamsim_cache: /model_cache/dreamsim
qwen3_model: /model_cache/evaluator/Qwen3-32B
qwen2_5_vl_model: /model_cache/evaluator/Qwen2.5-VL-72B-Instruct

seed: 42
```
