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
model_path="${model_path:-/path/to/BAGEL-7B-MoT}"
wandb_project="${wandb_project:-bagel-finetune}"
wandb_offline="${wandb_offline:-True}"
shared_latent_teacher_ckpt="${shared_latent_teacher_ckpt:-None}"
checkpoint_dir="${checkpoint_dir:-/path/to/outputs/latentumm/checkpoints}"
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
total_steps="${total_steps:-500000}"
save_every="${save_every:-1000}"
save_final_checkpoint="${save_final_checkpoint:-True}"
expected_num_tokens="${expected_num_tokens:-10240}"
max_num_tokens="${max_num_tokens:-11520}"
max_num_tokens_per_sample="${max_num_tokens_per_sample:-10240}"
num_replicate="${num_replicate:-1}"
num_shard="${num_shard:-$nproc_per_node}"
sharding_strategy="${sharding_strategy:-HYBRID_SHARD}"

# Shared latent losses can over-regularize image generation when too strong.
shared_latent_weight="${shared_latent_weight:-0.0}"
shared_latent_teacher_weight="${shared_latent_teacher_weight:-0.0}"

# LatentUMM hidden-state auxiliary losses. Disabled by default; enable for the
# real shared-transformer alignment objective.
latentumm_enable="${latentumm_enable:-False}"
latentumm_embedding_root="${latentumm_embedding_root:-/path/to/dataset_embedding/t2i}"
latentumm_modal_weight="${latentumm_modal_weight:-0.0}"
latentumm_task_weight="${latentumm_task_weight:-0.0}"
latentumm_pref_weight="${latentumm_pref_weight:-0.0}"
latentumm_target_weight="${latentumm_target_weight:-0.0}"
latentumm_num_rollouts="${latentumm_num_rollouts:-4}"
latentumm_noise_std="${latentumm_noise_std:-0.05}"
latentumm_source="${latentumm_source:-text}"

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
echo "LatentUMM profile: enable=$latentumm_enable modal=$latentumm_modal_weight task=$latentumm_task_weight pref=$latentumm_pref_weight target=$latentumm_target_weight K=$latentumm_num_rollouts sigma=$latentumm_noise_std source=$latentumm_source"
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
  --wandb_offline $wandb_offline \
  --num_replicate $num_replicate \
  --num_shard $num_shard \
  --sharding_strategy $sharding_strategy \
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
  --save_every $save_every \
  --save_final_checkpoint $save_final_checkpoint \
  --total_steps $total_steps \
  --checkpoint_dir $checkpoint_dir \
  --expected_num_tokens $expected_num_tokens \
  --max_num_tokens $max_num_tokens \
  --max_num_tokens_per_sample $max_num_tokens_per_sample \
  --shared_latent_weight $shared_latent_weight \
  --shared_latent_teacher_ckpt $shared_latent_teacher_ckpt \
  --shared_latent_teacher_weight $shared_latent_teacher_weight \
  --latentumm_enable $latentumm_enable \
  --latentumm_embedding_root $latentumm_embedding_root \
  --latentumm_modal_weight $latentumm_modal_weight \
  --latentumm_task_weight $latentumm_task_weight \
  --latentumm_pref_weight $latentumm_pref_weight \
  --latentumm_target_weight $latentumm_target_weight \
  --latentumm_num_rollouts $latentumm_num_rollouts \
  --latentumm_noise_std $latentumm_noise_std \
  --latentumm_source $latentumm_source \
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
