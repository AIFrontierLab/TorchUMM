#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
latentumm_root="$(cd "$script_dir/.." && pwd)"

output_dir="${output_dir:-/path/to/outputs/latentumm/stage1_alignment}"
prompts_path="${prompts_path:-/path/to/dataset/t2i/prompts.json}"
image_root="${image_root:-/path/to/dataset/t2i}"
embedding_root="${embedding_root:-/path/to/dataset_embedding/t2i}"
text_embedding_dir="${text_embedding_dir:-text_embedding}"
image_embedding_dir="${image_embedding_dir:-image_embedding}"
epochs="${epochs:-3}"
batch_size="${batch_size:-128}"
num_workers="${num_workers:-4}"
lr="${lr:-1e-4}"
lambda_modal="${lambda_modal:-1.0}"
lambda_task="${lambda_task:-1.0}"
lambda_gemini_target="${lambda_gemini_target:-0.1}"

python "$latentumm_root/train/stage1_shared_latent.py" \
  --prompts-path "$prompts_path" \
  --image-root "$image_root" \
  --embedding-root "$embedding_root" \
  --text-embedding-dir "$text_embedding_dir" \
  --image-embedding-dir "$image_embedding_dir" \
  --output-dir "$output_dir" \
  --epochs "$epochs" \
  --batch-size "$batch_size" \
  --num-workers "$num_workers" \
  --lr "$lr" \
  --lambda-modal "$lambda_modal" \
  --lambda-task "$lambda_task" \
  --lambda-gemini-target "$lambda_gemini_target" \
  --cache-embeddings
