#!/usr/bin/env bash
set -euo pipefail

# Defaults for single-node runs; can override via env.
num_nodes="${num_nodes:-1}"
node_rank="${node_rank:-0}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29500}"
model_path="${model_path:-}"

# --- Path & Core Configuration ---
master_addr=$(getent hosts "${master_addr:-127.0.0.1}" | awk '{ print $1 }' || echo "127.0.0.1")
num_nodes="${num_nodes:-1}"
master_port="${master_port:-29501}"
nproc_per_node="${nproc_per_node:-4}"
model_path="${model_path:-/sciclone/data10/yluo13/project/codebase/model_cache/bagel/models/BAGEL-7B-MoT}"
wandb_project="${wandb_project:-bagel-finetune}"


if [[ -z "$model_path" ]]; then
  echo "ERROR: model_path is not set. Export model_path or inline it before running." >&2
  exit 1
fi

# Ensure imports like `from data...` resolve
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"
export WANDB_PROJECT="$wandb_project"
cd "$script_dir"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume-from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --num_worker 1 \
  --num_replicate 1 \
  --num_shard 4 \
  --save_every 1000 \
  --results_dir ./results/checkpoint/ \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240
