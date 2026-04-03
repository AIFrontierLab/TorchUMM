# WISE -- Data Preparation

## Overview

WISE evaluates text-to-image generation quality using VLM-based judging.

Reference: https://github.com/PKU-YuanGroup/WISE

## Data

WISE benchmark data is included in the repository at `model/WISE/`. No additional download is needed.

## Evaluation Pipeline

WISE uses a two-stage evaluation:

1. **Generation**: Model-specific inference produces images from WISE prompts.
2. **Scoring**: Evaluator model (Qwen2.5-VL-72B-Instruct) judges the generated images against the prompts.

## Scoring Model

The scoring stage requires downloading the evaluator model:

- `Qwen2.5-VL-72B-Instruct` -- cached at `/model_cache/evaluator/Qwen2.5-VL-72B-Instruct`

## Config

Example config files: `configs/eval/wise/wise_bagel_generate.yaml`, `configs/eval/wise/wise_bagel_score.yaml`
