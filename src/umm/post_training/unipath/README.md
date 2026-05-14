# UniPath Post-Training

[![arXiv](https://img.shields.io/badge/arXiv-2605.11400-b31b1b.svg)](https://arxiv.org/abs/2605.11400)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-HuggingSelf%2FUniPath-yellow.svg)](https://huggingface.co/datasets/HuggingSelf/UniPath)

This folder contains the TorchUMM integration for UniPath. Keep all UniPath-specific code under this package; the
planner lives in `umm.post_training.unipath.planner`.

Path names in this README use paper notation: p<sub>dir</sub>, p<sub>A</sub>, p<sub>U</sub>, p<sub>R</sub>,
p<sub>C</sub>, and p<sub>H</sub>. Some scripts still keep legacy cache keys internally for backward compatibility.

## Local Artifacts

Download UniPath artifacts from the HF dataset repo to one local root:

```bash
export UNIPATH_ROOT=/path/to/UniPath
hf download HuggingSelf/UniPath --repo-type dataset \
  --local-dir "$UNIPATH_ROOT" \
  --include "processed/unireasoning_task_dataset_staged_with_images/**" \
  --include "processed/unireasoning_latents_20260406/**" \
  --include "processed/planner_outcomes/**" \
  --include "processed/external_scienceqa_1k/**" \
  --include "processed/external_ocrvqa_1k/**" \
  --include "processed/planner_artifacts_20260427/qfcp_single_route_fw_w003_h768_s54/**" \
  --include "processed/planner_feature_space_analysis/**" \
  --include "evals/mmmu_prompt_paths_full/**" \
  --include "checkpoints/old_lora_best_20260412/image_answer_visual/**"
```

The BAGEL base model is not part of this folder. Put it in the normal model cache, for example:

```bash
export UNIPATH_BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT
```

## Artifact Mapping

Set these paths after downloading:

```bash
export UNIPATH_ROOT=/path/to/UniPath
export UNIPATH_DATA_ROOT=$UNIPATH_ROOT/processed/unireasoning_task_dataset_staged_with_images
export UNIPATH_LATENT_CACHE_ROOT=$UNIPATH_ROOT/processed/unireasoning_latents_20260406/unireasoning_latents/cache
export UNIPATH_BAGEL_ADAPTER_DIR=$UNIPATH_ROOT/checkpoints/old_lora_best_20260412/image_answer_visual
export UNIPATH_PLANNER_PATH=$UNIPATH_ROOT/processed/planner_artifacts_20260427/qfcp_single_route_fw_w003_h768_s54/planner.pt
export UNIPATH_MMMU_RESULTS_ROOT=$UNIPATH_ROOT/evals/mmmu_prompt_paths_full
export UNIPATH_MMMU_FEATURES=$UNIPATH_ROOT/processed/planner_feature_space_analysis/features/mmmu/path_aware_mmmu_abs.pt
export UNIPATH_PLANNER_FEATURES=$UNIPATH_ROOT/processed/planner_feature_space_analysis/features/train/path_aware_pretrain_abs.pt
export UNIPATH_LORA_OUTPUT_ROOT=/path/to/unipath_runs/lora
export UNIPATH_PLANNER_OUTPUT_DIR=/path/to/unipath_runs/planner
export UNIPATH_ONLINE_ROUTE_OUTPUT_ROOT=/path/to/unipath_runs/online_route
```

If you want to reproduce the latest report planner directly, use `UNIPATH_PLANNER_PATH` above instead of retraining.

## Offline MMMU Replay

This is the fastest way to inspect UniPath routing behavior. We provide the complete staged-LoRA MMMU cached results under
`evals/mmmu_prompt_paths_full/`, plus the matching full900 planner input features. This path does not run BAGEL
generation; it only replays cached path outputs, so it is useful for:

- quickly previewing the reported MMMU routing result.
- comparing different planner checkpoints.
- trying different route-selection policies without regenerating answers.

```bash
PYTHONPATH=src python -m umm.post_training.unipath.planner.offline_mmmu_route \
  --planner-path "$UNIPATH_PLANNER_PATH" \
  --features-pt "$UNIPATH_MMMU_FEATURES" \
  --results-root "$UNIPATH_MMMU_RESULTS_ROOT" \
  --output-dir /path/to/unipath_runs/mmmu_offline_replay \
  --policy qfcp_refined_pragmatic_safe \
  --query-mode raw \
  --device cpu
```

## End-to-End Online Pipeline

The full UniPath workflow is:

1. Train the UniPath BAGEL LoRA adapter.
2. Train or load the planner.
3. Run online routing, where BAGEL extracts the planner feature, the planner selects one path, and BAGEL answers with
   that selected path.

### 1. LoRA Training

Run the four stages in order:

```bash
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_lora_understanding_text.yaml
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_lora_understanding_visual.yaml
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_lora_image_answer_plain.yaml
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_lora_image_answer_visual.yaml
```

Latent cache generation is normally automatic in the LoRA configs. To run it explicitly:

```bash
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_cache_latents.yaml
```

The adapter used by downstream routing should be:

```bash
export UNIPATH_BAGEL_ADAPTER_DIR=$UNIPATH_LORA_OUTPUT_ROOT/image_answer_visual/adapter_imitation_best
```

### 2. Planner Training

Train a planner from cached UniPath features:

```bash
PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/unipath_planner.yaml
```

Set `UNIPATH_PLANNER_PATH` to the trained `planner.pt`, or keep the downloaded report planner if you only want to run
evaluation.

### 3. Online Routing Evaluation

Online routing runs actual BAGEL inference. The checked-in example config is MMMU:

```bash
PYTHONPATH=src python -m umm.cli.main eval --config configs/eval/unipath/online_route_mmmu.yaml
```

## What Each Artifact Is For

- `processed/unireasoning_task_dataset_staged_with_images/`: four-stage LoRA train/val JSONL files.
- `processed/unireasoning_latents_20260406/unireasoning_latents/cache/`: BAGEL VAE latent cache for image-answer stages.
- `checkpoints/old_lora_best_20260412/image_answer_visual/`: final UniPath BAGEL staged-LoRA adapter used by routing eval.
- `processed/planner_outcomes/`: original multi-path outcome labels and non-path-aware planner feature cache.
- `processed/external_scienceqa_1k/` and `processed/external_ocrvqa_1k/`: auxiliary planner training outcome data.
- `processed/planner_feature_space_analysis/features/train/path_aware_pretrain_abs.pt`: path-aware planner training feature cache.
- `processed/planner_feature_space_analysis/features/mmmu/path_aware_mmmu_abs.pt`: MMMU full900 planner input features.
- `processed/planner_artifacts_20260427/qfcp_single_route_fw_w003_h768_s54/planner.pt`: current report planner weight.
- `evals/mmmu_prompt_paths_full/`: staged-LoRA MMMU cached generations for offline replay.

More planner-specific details are in `planner/README.md`.

## Citation

If UniPath is helpful for your work, please cite:

```bibtex
@misc{bai2026unipathadaptivecoordinationunderstanding,
      title={UniPath: Adaptive Coordination of Understanding and Generation for Unified Multimodal Reasoning},
      author={Hayes Bai and Yinyi Luo and Wenwen Wang and Qingsong Wen and Jindong Wang},
      year={2026},
      eprint={2605.11400},
      archivePrefix={arXiv},
      primaryClass={cs.MM},
      url={https://arxiv.org/abs/2605.11400},
}
```

This implementation is integrated in TorchUMM. Please also cite:

```bibtex
@article{luo2026torchumm,
  title={TorchUMM: A Unified Multimodal Model Codebase for Evaluation, Analysis, and Post-training},
  author={Luo, Yinyi and Wang, Wenwen and Bai, Hayes and Zhu, Hongyu and Chen, Hao and He, Pan and Savvides, Marios and Li, Sharon and Wang, Jindong},
  journal={arXiv preprint arXiv:2604.10784},
  year={2026}
}
```
