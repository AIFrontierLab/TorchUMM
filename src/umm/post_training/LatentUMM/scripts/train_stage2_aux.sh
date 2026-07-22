#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
latentumm_root="$(cd "$script_dir/.." && pwd)"

export latentumm_enable="${latentumm_enable:-True}"
export latentumm_modal_weight="${latentumm_modal_weight:-0.01}"
export latentumm_task_weight="${latentumm_task_weight:-0.01}"
export latentumm_pref_weight="${latentumm_pref_weight:-0.001}"
export latentumm_target_weight="${latentumm_target_weight:-0.01}"
export latentumm_num_rollouts="${latentumm_num_rollouts:-2}"
export latentumm_noise_std="${latentumm_noise_std:-0.05}"

export shared_latent_teacher_ckpt="${shared_latent_teacher_ckpt:-/path/to/outputs/latentumm/stage1_alignment/stage1_shared_latent_model.pt}"
export shared_latent_weight="${shared_latent_weight:-0.001}"
export shared_latent_teacher_weight="${shared_latent_teacher_weight:-0.001}"

export lora_rank="${lora_rank:-64}"
export lora_alpha="${lora_alpha:-128}"
export checkpoint_dir="${checkpoint_dir:-/path/to/outputs/latentumm/stage2_aux}"

bash "$latentumm_root/train.sh"
