# GenEval -- Data Preparation

## Overview

GenEval evaluates text-to-image models on compositional generation tasks using object detection for scoring.

Reference: https://github.com/djghosh13/geneval

## Data

Evaluation prompts and metadata are included with the repository in `model/geneval/`. No additional download is needed.

## Evaluation Pipeline

GenEval uses a two-stage evaluation:

1. **Generation**: The model generates images from prompts.
2. **Scoring**: Object detection models verify whether the generated images contain the expected objects and relationships.

## Config

Example config file: `configs/eval/geneval/geneval_bagel.yaml`
