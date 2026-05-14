# UniPath Planner Post-Training

This folder contains the UniPath path planner training and offline routing code in TorchUMM's post-training interface.

Path names in this README use paper notation: p<sub>dir</sub>, p<sub>A</sub>, p<sub>U</sub>, p<sub>R</sub>,
p<sub>C</sub>, and p<sub>H</sub>. Some scripts still keep legacy cache keys internally for backward compatibility.

## Stable Planner

- BAGEL adapter used by downstream generation: `$UNIPATH_BAGEL_ADAPTER_DIR`
- planner weight used in the current report: `$UNIPATH_PLANNER_PATH`
- HF artifact path: `HuggingSelf/UniPath/processed/planner_artifacts_20260427/qfcp_single_route_fw_w003_h768_s54/planner.pt`
- policy: `qfcp_refined_pragmatic_safe`

The planner is an MLP over cached BAGEL multimodal features. The stable training recipe uses weighted BCE, feature dropout/noise, weak path-distribution regularization, and checkpoint selection by MMMU offline routing accuracy.

## Train

TorchUMM-compatible entrypoint:

```bash
export UNIPATH_ROOT=/path/to/UniPath
export UNIPATH_PLANNER_FEATURES=$UNIPATH_ROOT/processed/planner_feature_space_analysis/features/train/path_aware_pretrain_abs.pt
export UNIPATH_MMMU_FEATURES=$UNIPATH_ROOT/processed/planner_feature_space_analysis/features/mmmu/path_aware_mmmu_abs.pt
export UNIPATH_MMMU_RESULTS_ROOT=$UNIPATH_ROOT/evals/mmmu_prompt_paths_full
export UNIPATH_PLANNER_OUTPUT_DIR=$PWD/outputs/torchumm_unipath_planner
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_planner.yaml
```

Equivalent direct module call:

```bash
PYTHONPATH=src python -m umm.post_training.unipath.planner.train \
  --features-pt "$UNIPATH_PLANNER_FEATURES" \
  --output-dir ./outputs/torchumm_unipath_planner \
  --mmmu-features-pt "$UNIPATH_MMMU_FEATURES" \
  --mmmu-results-root "$UNIPATH_MMMU_RESULTS_ROOT" \
  --planner-type mlp --hidden-dim 768 --batch-size 4096 --epochs 45 \
  --learning-rate 4e-4 --weight-decay 5e-5 --train-sampler epoch \
  --feature-dropout 0.05 --feature-noise-std 0.02 \
  --single-positive-weight 3.0 --double-positive-weight 2.0 --multi-positive-weight 1.0 \
  --nondirect-positive-weight 1.3 --level-weight ocrvqa=0.25 \
  --dist-reg-weight 0.03 --dist-reg-prior 0.58,0.20,0.13,0.05,0.04 \
  --dist-reg-mode prob_kl --margin 0.03 --seed 54 --device cuda \
  --best-by mmmu_accuracy --mmmu-eval-every 1
```

## Offline MMMU Replay

This validates the planner against cached MMMU path outputs without running BAGEL generation:

```bash
PYTHONPATH=src python -m umm.post_training.unipath.planner.offline_mmmu_route \
  --planner-path "$UNIPATH_PLANNER_PATH" \
  --features-pt "$UNIPATH_MMMU_FEATURES" \
  --results-root "$UNIPATH_MMMU_RESULTS_ROOT" \
  --output-dir ./outputs/torchumm_mmmu_qfcp_safe_replay \
  --policy qfcp_refined_pragmatic_safe --query-mode raw --device cpu
```

Expected report-aligned replay on the staged-LoRA MMMU cache:

MMMU full900: `490/900 = 54.44%`

Path distribution: p<sub>dir</sub>=159, p<sub>A</sub>=43, p<sub>U</sub>=195, p<sub>R</sub>=503,
p<sub>C</sub>=0, p<sub>H</sub>=0.

`--query-mode raw` is intentional for reproducing the reported MMMU offline replay. Use
`--query-mode mmmu_query` only when you want the bucket policy to see the appended options and
answer instruction.

## Online Routing Evaluation

Online routing runs BAGEL once to extract the planner feature, selects one path, then runs BAGEL on that selected path.
It supports `mmmu`, `mmbench`, `mme`, `mathvista`, and `mmstar` through a shared entrypoint:

```bash
PYTHONPATH=src python -m umm.cli.main eval \
  --config configs/eval/unipath/online_route_mmmu.yaml
```

The direct module entrypoint is:

```bash
PYTHONPATH=src python -m umm.post_training.unipath.planner.online_route_eval \
  --benchmark mmmu \
  --planner-path "$UNIPATH_PLANNER_PATH" \
  --model-path "$UNIPATH_BAGEL_MODEL_PATH" \
  --adapter-dir "$UNIPATH_BAGEL_ADAPTER_DIR" \
  --output-dir ./outputs/unipath_online_route_mmmu
```

The online route evaluator writes:

- `metrics.json`: aggregate score, path distribution, bucket distribution, and cache paths.
- `cases.jsonl`: per-sample route, planner scores, selected path, raw response, and parsed answer.
- `cache/planner_predictions.jsonl`: reusable planner scores.
- `cache/responses.jsonl`: reusable selected-path responses.

For the current report planner, online feature extraction uses the same path-aware `all_abs` layout as the offline
feature cache: one image summary plus text features for the configured UniPath path prompts. This avoids using answer
correctness or dataset-level statistics at test time.

## Files

- `models.py`: planner architectures, checkpoint loading, and feature slicing.
- `bagel_features.py`: online BAGEL feature extraction for direct and path-aware planner layouts.
- `qfcp_policy.py`: query-form calibrated path policy. It uses only sample-local query text/options and planner scores.
- `train.py`: post-training entrypoint for planner training.
- `offline_mmmu_route.py`: offline replay against cached MMMU path outputs.
- `online_route_eval.py`: online route selection and selected-path BAGEL inference for multiple datasets.
- `pipeline.py`: TorchUMM train dispatcher wrapper for `pipeline: planner`.
