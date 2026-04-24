#!/usr/bin/env bash
set -euo pipefail

# ---- Cluster defaults (override via env) ----
num_nodes="${num_nodes:-1}"
node_rank="${node_rank:-0}"
master_addr="${master_addr:-127.0.0.1}"
master_port="${master_port:-29510}"
nproc_per_node="${nproc_per_node:-2}"

# ---- Model & Data ----
model_path="${model_path:-./model_cache/janus/models/Janus_pro_7B}"
data_json="${data_json:-./dataset/t2i/prompts.json}"
image_root="${image_root:-./dataset/t2i}"
image_size="${image_size:-384}"
vq_downsample="${vq_downsample:-16}"

# ---- Training ----
batch_size="${batch_size:-1}"
gradient_accumulation_steps="${gradient_accumulation_steps:-1}"
total_steps="${total_steps:-5000}"
log_every="${log_every:-1}"
save_every="${save_every:-1000}"
lr="${lr:-1e-5}"
warmup_steps="${warmup_steps:-100}"
wandb_project="${wandb_project:-janus-posttrain}"
wandb_name="${wandb_name:-t2i}"
wandb_offline="${wandb_offline:-false}"

# ---- Paths ----
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"
cd "$script_dir"

# ---- Sanity checks ----
if [[ ! -f "$data_json" ]]; then
  echo "ERROR: data_json not found: $data_json" >&2
  exit 1
fi

if [[ ! -d "$image_root" ]]; then
  echo "ERROR: image_root not found: $image_root" >&2
  exit 1
fi

if [[ ! -d "$model_path" ]]; then
  echo "ERROR: model_path not found (expected local dir): $model_path" >&2
  echo "Set model_path to a local folder or a valid HF repo id (e.g. deepseek-ai/Janus-1.3B)." >&2
  exit 1
fi

# ---- Launch ----
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/post_train.py \
  --model_path "$model_path" \
  --data_json "$data_json" \
  --image_root "$image_root" \
  --image_size "$image_size" \
  --vq_downsample "$vq_downsample" \
  --batch_size "$batch_size" \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --total_steps "$total_steps" \
  --lr "$lr" \
  --warmup_steps "$warmup_steps" \
  --log_every "$log_every" \
  --save_every "$save_every" \
  --checkpoint_dir results/checkpoints \
  --wandb_project "$wandb_project" \
  --wandb_name "$wandb_name" \
  --wandb_offline "$wandb_offline"
