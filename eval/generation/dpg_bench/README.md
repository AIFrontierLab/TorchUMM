# DPG Bench -- Data Preparation

## Overview

DPG Bench (Dense Prompt Generation Benchmark) evaluates text-to-image models on detailed, compositional prompts.

Reference: https://github.com/TencentQQGYLab/ELLA

## Data

No additional download is needed. All required data is already included in this repository:

- **Prompts**: `eval/generation/dpg_bench/prompts/` (100 prompt files)
- **Metadata**: `eval/generation/dpg_bench/dpg_bench.csv`

## Scoring

Scoring uses mPLUG VQA model to measure alignment between generated images and their text prompts.

## Config

Example config file: `configs/eval/dpg_bench/dpg_bench_bagel.yaml`
