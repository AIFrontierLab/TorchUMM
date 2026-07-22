#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$repo_root" ]]; then
  repo_root="$(cd "$script_dir/../../../../.." && pwd)"
fi

model_path="${LATENTUMM_EVAL_MODEL_PATH:-/path/to/BAGEL-7B-MoT}"
data_root="${UMM_DATASETS:-/path/to/data}"
output_root="${LATENTUMM_EVAL_OUTPUT_ROOT:-$repo_root/output/latentumm_eval}"
evaluator_root="${LATENTUMM_EVALUATOR_ROOT:-/path/to/evaluator}"
bagel_root="${LATENTUMM_BAGEL_ROOT:-$repo_root/src/umm/post_training/LatentUMM}"
skip_preflight="${LATENTUMM_SKIP_PREFLIGHT:-False}"
checkpoint_file="${LATENTUMM_CHECKPOINT_FILE:-}"
if [[ -z "$checkpoint_file" ]]; then
  if [[ -e "$model_path/ema.safetensors" ]]; then
    checkpoint_file="ema.safetensors"
  else
    checkpoint_file="model.safetensors"
  fi
fi
lora_scaling="${LATENTUMM_LORA_SCALING:-}"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

rewrite_config() {
  local src="$1"
  local dst="$2"
  local bench="$3"
  sed \
    -e "s#\${UMM_CODEBASE}/src/umm/backbones/bagel/Bagel#$bagel_root#g" \
    -e "s#\${UMM_MODEL_CACHE}/bagel/BAGEL-7B-MoT#$model_path#g" \
    -e "s#\${UMM_MODEL_CACHE}/evaluator#$evaluator_root#g" \
    -e "s#\${UMM_DATASETS}#$data_root#g" \
    -e "s#data/MMMU#$data_root/MMMU#g" \
    -e "s#data/MMStar#$data_root/MMStar#g" \
    -e "s#data/MathVista#$data_root/MathVista#g" \
    -e "s#data/mme#$data_root/mme#g" \
    -e "s#output/$bench/bagel#$output_root/$bench#g" \
    "$src" > "$dst"
}

run_eval() {
  local bench="$1"
  local config="$2"
  local rewritten="$tmp_dir/${bench}.yaml"
  local args=(--config "$rewritten" --set "inference.backbone_cfg.checkpoint_file=$checkpoint_file")
  if [[ -n "$lora_scaling" ]]; then
    args+=(--set "inference.backbone_cfg.lora_scaling=$lora_scaling")
  fi
  rewrite_config "$repo_root/$config" "$rewritten" "$bench"
  python -m umm.cli.main eval "${args[@]}"
}

export PYTHONPATH="$repo_root/src:${PYTHONPATH:-}"
cd "$repo_root"

if [[ "$skip_preflight" != "True" && "$skip_preflight" != "true" ]]; then
  missing=0
  for required in \
    "$data_root/mme/MME_Benchmark_release_version" \
    "$data_root/mmbench/MMBench_TEST_EN_V11.tsv" \
    "$data_root/mmbench/MMBench_TEST_CN_V11.tsv" \
    "$data_root/MMMU" \
    "$data_root/MMStar" \
    "$data_root/MathVista/annot_testmini.json" \
    "$evaluator_root/Qwen3-32B-AWQ" \
    "$model_path/$checkpoint_file"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required benchmark asset: $required" >&2
      missing=1
    fi
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Benchmark preflight failed. Set LATENTUMM_SKIP_PREFLIGHT=True to run anyway." >&2
    exit 2
  fi
fi

run_eval mme configs/eval/mme/mme_bagel.yaml
run_eval mmmu configs/eval/mmmu/mmmu_bagel.yaml
run_eval mmbench configs/eval/mmbench/mmbench_bagel.yaml
run_eval mmstar configs/eval/mmstar/mmstar_bagel.yaml
run_eval mathvista configs/eval/mathvista/mathvista_bagel.yaml
run_eval mathvista configs/eval/mathvista/mathvista_bagel_score.yaml
