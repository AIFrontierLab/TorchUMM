# UEval -- Data Preparation

## Overview

UEval is a unified evaluation benchmark for text-to-image generation models.

Reference: https://github.com/mdl-ueval/UEval

## Data

The dataset is hosted on HuggingFace: `primerL/UEval-all`

### Modal (cloud)

Download the dataset to the Modal volume:

```bash
modal run modal/download.py --dataset ueval
```

This downloads the data to `/datasets/ueval/UEval` inside the container.

### Local

The dataset is auto-downloaded via HuggingFace during evaluation. No manual download is needed.

## Evaluation Pipeline

UEval uses a two-stage evaluation:

1. **Generation**: Model-specific inference produces images from UEval prompts.
2. **Scoring**: Evaluator models (Qwen2.5-VL-72B + Qwen3-32B) score the generated images.

## Scoring Models

The scoring stage requires downloading the evaluator models:

- `Qwen2.5-VL-72B-Instruct` -- cached at `/model_cache/evaluator/Qwen2.5-VL-72B-Instruct`
- `Qwen3-32B` -- cached at `/model_cache/evaluator/Qwen3-32B`
