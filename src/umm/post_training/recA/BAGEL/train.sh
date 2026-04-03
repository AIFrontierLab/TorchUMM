export master_addr=localhost
export master_port=12345
export output_path='./'
export ckpt_path='./checkpoints'
export PYTHONPATH=.

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=4 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/pretrain_unified_navit.py \
  --model_path '/sciclone/data10/yluo13/project/codebase/model_cache/bagel/models/BAGEL-7B-MoT' \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --freeze_vae True \
  --freeze_vit True \
  --freeze_und True \
  --finetune_from_ema True \
  --resume_from '/sciclone/data10/yluo13/project/codebase/model_cache/bagel/models/BAGEL-7B-MoT' \
  --results_dir $output_path \
  --checkpoint_dir $ckpt_path \
  --log_every 1 \
  --wandb_runid 1 \
  --use_flex \
  --num_replicate 1 \
  --num_shard 4 \
  --lr 0.00004 \
  --total_steps 1000