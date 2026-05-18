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
model_path="${model_path:-<path-to-bagel-7b-mot-model>}"
wandb_project="${wandb_project:-bagel-finetune}"
shared_latent_teacher_ckpt="${shared_latent_teacher_ckpt:-<path-to-stage2-rollout-model-ckpt>}"
checkpoint_dir="${checkpoint_dir:-<path-to-stage2-integrated-checkpoint-dir>}"
rir_enable="${rir_enable:-False}"
rir_every="${rir_every:-11}"

# --- Anti-collapse defaults (override via env when needed) ---
# Keep trainable update small to preserve generation quality.
lr="${lr:-5e-6}"
min_lr="${min_lr:-1e-6}"
warmup_steps="${warmup_steps:-1000}"
ce_weight="${ce_weight:-0.7}"
mse_weight="${mse_weight:-1.0}"
ema_decay="${ema_decay:-0.99995}"

# Shared latent losses can over-regularize image generation when too strong.
shared_latent_weight="${shared_latent_weight:-0.02}"
shared_latent_teacher_weight="${shared_latent_teacher_weight:-0.01}"

# LoRA by default to avoid full-parameter drift; set lora_rank=0 for full finetune.
lora_rank="${lora_rank:-64}"
lora_alpha="${lora_alpha:-128}"

if [[ -z "$model_path" ]]; then
  echo "ERROR: model_path is not set. Export model_path or inline it before running." >&2
  exit 1
fi

# Ensure imports like `from data...` resolve
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${script_dir}:${PYTHONPATH:-}"
export WANDB_PROJECT="$wandb_project"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$script_dir"

echo "Checkpoint dir: $checkpoint_dir"
echo "RIR enabled: $rir_enable (every $rir_every steps)"
echo "Train profile: lr=$lr min_lr=$min_lr warmup=$warmup_steps ce=$ce_weight mse=$mse_weight"
echo "Latent profile: shared=$shared_latent_weight teacher=$shared_latent_teacher_weight"
echo "Adapter profile: lora_rank=$lora_rank lora_alpha=$lora_alpha"

torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=$nproc_per_node \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit_pt.py \
  --dataset_config_file ./data/configs/example.yaml \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --resume_from $model_path \
  --finetune_from_hf True \
  --auto_resume True \
  --resume_model_only True \
  --finetune_from_ema True \
  --rir_separate_backward True \
  --rir_rollout_lazy True \
  --rir_rollout_free_each_step True \
  --log_every 1 \
  --num_replicate 1 \
  --num_shard 4 \
  --lr $lr \
  --min_lr $min_lr \
  --lr_scheduler cosine \
  --warmup_steps $warmup_steps \
  --ema $ema_decay \
  --ce_weight $ce_weight \
  --mse_weight $mse_weight \
  --lora_rank $lora_rank \
  --lora_alpha $lora_alpha \
  --num_workers 1 \
  --save_every 1000 \
  --checkpoint_dir $checkpoint_dir \
  --expected_num_tokens 10240 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --shared_latent_weight $shared_latent_weight \
  --shared_latent_teacher_ckpt $shared_latent_teacher_ckpt \
  --shared_latent_teacher_weight $shared_latent_teacher_weight \
  --rir_enable $rir_enable \
  --rir_rollout_device cuda \
  --rir_skip_logp True \
  --rir_prompt_max_tokens 128 \
  --rir_image_height 512 \
  --rir_image_width 512 \
  --rir_num_timesteps 20 \
  --rir_every $rir_every \
  --rir_timestep_shift 3.0 \
  --rir_cfg_text_scale 4.0 \
  --rir_cfg_img_scale 1.0 \
  --rir_cfg_interval "0.4,1.0" \
  --rir_cfg_renorm_min 0.0 \
  --rir_cfg_renorm_type global \
  --rir_beta 0.1 \
  --rir_dpo_weight 1.0 \
  --rir_grounding_weight 0.1
