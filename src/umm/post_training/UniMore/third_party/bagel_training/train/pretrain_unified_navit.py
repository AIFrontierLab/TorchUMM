# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import re
import wandb
import yaml
from dataclasses import dataclass, field
from time import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)
from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger
from train.grouped_i2t_loss import reduce_ranking_violations
from train.fsdp_utils import FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper

DEFAULT_LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "q_proj_moe_gen",
    "k_proj_moe_gen",
    "v_proj_moe_gen",
    "o_proj_moe_gen",
)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _apply_lora_to_language_model(model, training_args, logger):
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "`use_lora=True` requires peft to be installed in the BAGEL environment."
        ) from exc
    for param in model.language_model.parameters():
        param.requires_grad = False
    target_modules = _split_csv(training_args.lora_target_modules)
    if not target_modules:
        target_modules = list(DEFAULT_LORA_TARGETS)
    lora_config = LoraConfig(
        r=int(training_args.lora_r),
        lora_alpha=int(training_args.lora_alpha),
        lora_dropout=float(training_args.lora_dropout),
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model.language_model = inject_adapter_in_model(lora_config, model.language_model)
    model.language_model.train()
    if hasattr(model.language_model, "model"):
        model.language_model.model.train()
    lora_params = 0
    for name, param in model.language_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params += param.numel()
    trainable = sum((p.numel() for p in model.language_model.parameters() if p.requires_grad))
    total = sum((p.numel() for p in model.language_model.parameters()))
    logger.info(
        f"Applied LoRA to language_model: r={training_args.lora_r}, alpha={training_args.lora_alpha}, dropout={training_args.lora_dropout}, targets={target_modules}, lora_params={lora_params}, trainable={trainable}/{total}"
    )
    return model


def _collect_lora_modules(model, logger):
    modules = []
    for name, module in model.named_modules():
        if re.search("(^|\\.)lora_[AB]$", name):
            modules.append(module)
    if dist.get_rank() == 0:
        logger.info(f"Ignoring {len(modules)} LoRA A/B modules in FSDP wrapping.")
    return modules


@dataclass
class ModelArguments:
    model_path: str = field(
        default="ckpt/BAGEL-7B-MoT",
        metadata={"help": "Path to the directory containing the pretrained model files."},
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."},
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."},
    )
    layer_module: str = field(
        default="Qwen2DecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."},
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."},
    )
    latent_patch_size: int = field(
        default=2, metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14, metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={
            "help": "Maximum number of ViT patches along one image side after cropping / resize."
        },
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."},
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={
            "help": "Interpolate positional embeddings when image resolution differs from pre-training."
        },
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={
            "help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."
        },
    )
    vit_rope: bool = field(
        default=False, metadata={"help": "Replace ViT positional encodings with RoPE."}
    )
    text_cond_dropout_prob: float = field(
        default=0.1, metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3, metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."},
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."},
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."},
    )
    num_workers: int = field(
        default=1, metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."},
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={
            "help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."
        },
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={
            "help": "While batch length is below this, pop from the overflow buffer before new sampling."
        },
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."},
    )
    data_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used when shuffling / sampling data shards to ensure reproducibility."
        },
    )


@dataclass
class TrainingArguments:
    visual_gen: bool = field(default=True, metadata={"help": "Train image generation branch."})
    visual_und: bool = field(default=True, metadata={"help": "Train image understanding branch."})
    results_dir: str = field(default="results", metadata={"help": "Root directory for logs."})
    checkpoint_dir: str = field(
        default="results/checkpoints", metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel_a800", metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run", metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."},
    )
    wandb_resume: str = field(
        default="allow", metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False, metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )
    save_after: int = field(
        default=1000, metadata={"help": "Save the checkpoint after N training steps."}
    )
    global_seed: int = field(
        default=4396, metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)."},
    )
    log_every: int = field(default=10, metadata={"help": "Print / log every N training steps."})
    save_every: int = field(
        default=250, metadata={"help": "Save a checkpoint every N training steps."}
    )
    save_final: bool = field(
        default=False,
        metadata={"help": "Save a checkpoint after the last completed training step."},
    )
    save_optimizer: bool = field(
        default=True,
        metadata={"help": "Also save optimizer, scheduler, and data-status state in checkpoints."},
    )
    total_steps: int = field(
        default=500000, metadata={"help": "Total number of optimizer steps to train for."}
    )
    warmup_steps: int = field(
        default=1000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."},
    )
    lr_scheduler: str = field(
        default="constant", metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(default=0.0001, metadata={"help": "Peak learning rate after warm-up."})
    min_lr: float = field(
        default=1e-07,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."},
    )
    beta1: float = field(default=0.9, metadata={"help": "AdamW β₁ coefficient."})
    beta2: float = field(default=0.95, metadata={"help": "AdamW β₂ coefficient."})
    eps: float = field(default=1e-15, metadata={"help": "AdamW ε for numerical stability."})
    lora_r: int = field(default=16, metadata={"help": "LoRA adapter rank for the language model."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha for the language model."})
    lora_dropout: float = field(
        default=0.05, metadata={"help": "LoRA dropout for the language model."}
    )
    fsdp_ignore_lora: bool = field(
        default=False,
        metadata={
            "help": "Keep LoRA A/B modules unsharded under FSDP; required for rank-wise gradient routing."
        },
    )
    lora_target_modules: str = field(
        default=",".join(DEFAULT_LORA_TARGETS),
        metadata={"help": "Comma-separated LLM module names to receive LoRA adapters."},
    )
    max_grad_norm: int = field(
        default=1.0, metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."},
    )
    mse_weight: float = field(
        default=1.0, metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0, metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ranking_margin: float = field(
        default=0.2,
        metadata={"help": "Margin for grouped i2t ranking loss over phrase candidates."},
    )
    expected_num_tokens: int = field(
        default=32768,
        metadata={
            "help": "Soft target token count; yield the batch once it reaches or exceeds this size."
        },
    )
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."},
    )
    num_shard: int = field(
        default=8, metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."},
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "Enable FSDP parameter offload to CPU."}
    )
    freeze_llm: bool = field(
        default=False, metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False, metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={
            "help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."
        },
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."},
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."},
    )


def main():
    assert torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    device = local_rank
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args, allow_val_change=True)
        wandb.config.update(data_args, allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f"Training arguments {training_args}")
    logger.info(f"Model arguments {model_args}")
    logger.info(f"Data arguments {data_args}")
    resume_from = training_args.resume_from
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)
    llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = getattr(llm_config, "eos_token_id", None)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = False
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()
    if training_args.visual_und:
        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_args.model_path, "vit_config.json")
        )
        vit_config.num_hidden_layers = (
            vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        )
        vit_config.rope = model_args.vit_rope
        vit_model = SiglipVisionModel(vit_config)
    if training_args.visual_gen:
        (vae_model, vae_config) = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
        )
    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    model = Bagel(language_model, vit_model if training_args.visual_und else None, config)
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    (tokenizer, new_token_ids, num_new_tokens) = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    ema_model = None
    (model, ema_model) = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, None, resume_from_ema=True
    )
    model = _apply_lora_to_language_model(model, training_args, logger)
    ignored_modules = _collect_lora_modules(model, logger) if training_args.fsdp_ignore_lora else []
    for module in ignored_modules:
        module.to(torch.device("cuda", device))
    fsdp_model = fsdp_wrapper(model, fsdp_config, ignored_modules=ignored_modules)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn,
    )
    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    optimizer_params = fsdp_model.parameters()
    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError
    train_step = 0
    data_status = None

    def build_train_loader(dataset_config_file, seed_offset=0, loader_data_status=None):
        with open(dataset_config_file, "r") as stream:
            dataset_meta = yaml.safe_load(stream)
        dataset_config = DataConfig(grouped_datasets=dataset_meta)
        if training_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if training_args.visual_gen:
            vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
            dataset_config.vae_image_downsample = vae_image_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
            dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
            dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
        train_dataset = PackedDataset(
            dataset_config,
            tokenizer=tokenizer,
            special_tokens=new_token_ids,
            local_rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            num_workers=data_args.num_workers,
            expected_num_tokens=training_args.expected_num_tokens,
            max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
            max_num_tokens=data_args.max_num_tokens,
            max_buffer_size=data_args.max_buffer_size,
            prefer_buffer_before=data_args.prefer_buffer_before,
            interpolate_pos=model_args.interpolate_pos,
            use_flex=training_args.use_flex,
            data_status=loader_data_status,
        )
        train_dataset.set_epoch(data_args.data_seed + seed_offset)
        return DataLoader(train_dataset, **loader_kwargs)

    loader_kwargs = dict(
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
    )
    if data_args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = data_args.prefetch_factor
    train_loader = build_train_loader(data_args.dataset_config_file, loader_data_status=data_status)
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    last_completed_step = train_step - 1

    def save_checkpoint(step):
        dist.barrier()
        torch.cuda.empty_cache()
        gather_list = None
        try:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            safe_data_status = dict(data_status) if data_status is not None else {}
            dist.gather_object(safe_data_status, gather_list, dst=0)
        except Exception as e:
            logger.warning(f"Failed to gather data_status: {e}. Continuing with checkpoint save...")
            if dist.get_rank() == 0:
                gather_list = [{}] * dist.get_world_size()
        FSDPCheckpoint.fsdp_save_ckpt(
            ckpt_dir=training_args.checkpoint_dir,
            train_steps=step,
            model=fsdp_model,
            ema_model=None,
            optimizer=optimizer,
            scheduler=scheduler,
            logger=logger,
            fsdp_config=fsdp_config,
            data_status=gather_list,
            save_optimizer=training_args.save_optimizer,
        )

    def compute_grouped_i2t_loss(token_ce, batch_data, batch_data_indexes):
        metadata_list = batch_data_indexes or []
        if not metadata_list:
            total_ce_tokens = torch.tensor(float(token_ce.numel()), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if total_ce_tokens.item() <= 0:
                return token_ce.sum() * 0.0
            return token_ce.sum() * dist.get_world_size() / total_ce_tokens
        sample_lens = [int(item) for item in batch_data["sample_lens"][: len(metadata_list)]]
        ce_positions = batch_data["ce_loss_indexes"]
        sample_items = []
        start = 0
        for sample_len in sample_lens:
            end = start + sample_len
            mask = (ce_positions >= start) & (ce_positions < end)
            if mask.any().item():
                sample_token_ce = token_ce[mask]
                sample_items.append(
                    {
                        "mean_loss": sample_token_ce.mean(),
                        "sum_loss": sample_token_ce.sum(),
                        "token_count": int(mask.sum().item()),
                    }
                )
            else:
                sample_items.append(None)
            start = end
        grouped = {}
        regular_token_loss = token_ce.sum() * 0.0
        regular_token_count = 0
        for metadata, sample_item in zip(metadata_list, sample_items):
            if sample_item is None:
                continue
            group_id = metadata.get("contrastive_group_id")
            if not group_id:
                regular_token_loss = regular_token_loss + sample_item["sum_loss"]
                regular_token_count += sample_item["token_count"]
                continue
            grouped.setdefault(group_id, []).append((metadata, sample_item["mean_loss"]))
        group_losses = []
        for group_items in grouped.values():
            positives = [
                sample_loss
                for (metadata, sample_loss) in group_items
                if int(metadata.get("contrastive_label", 0)) == 1
            ]
            negatives = [
                sample_loss
                for (metadata, sample_loss) in group_items
                if int(metadata.get("contrastive_label", 0)) == 0
            ]
            if not positives or not negatives:
                continue
            pos_loss = positives[0]
            neg_losses = torch.stack(negatives)
            pos_score = -pos_loss
            neg_scores = -neg_losses
            ranking_loss = reduce_ranking_violations(
                pos_score, neg_scores, training_args.ranking_margin, "mean"
            )
            group_losses.append(pos_loss + ranking_loss)
        regular_loss = token_ce.sum() * 0.0
        global_regular_token_count = torch.tensor(float(regular_token_count), device=device)
        dist.all_reduce(global_regular_token_count, op=dist.ReduceOp.SUM)
        if global_regular_token_count.item() > 0:
            regular_loss = regular_token_loss * dist.get_world_size() / global_regular_token_count
        local_group_count = torch.tensor(float(len(group_losses)), device=device)
        global_group_count = local_group_count.clone()
        dist.all_reduce(global_group_count, op=dist.ReduceOp.SUM)
        grouped_loss = token_ce.sum() * 0.0
        if global_group_count.item() > 0:
            local_group_loss = (
                torch.stack(group_losses).sum() if group_losses else token_ce.sum() * 0.0
            )
            grouped_loss = local_group_loss * dist.get_world_size() / global_group_count
        return regular_loss + grouped_loss

    train_iter = iter(train_loader)
    curr_step = train_step
    while curr_step < training_args.total_steps:
        if curr_step >= training_args.total_steps:
            logger.info(f"Reached total_steps={training_args.total_steps}; stopping training loop.")
            break
        data = next(train_iter)
        data = data.cuda(device).to_dict()
        data_indexes = data.pop("batch_data_indexes", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)
        if training_args.visual_gen and "padded_images" in data:
            with torch.no_grad():
                data["padded_latent"] = vae_model.encode(data.pop("padded_images"))

        def reduce_loss_dict(raw_loss_dict):
            reduced_loss_dict = raw_loss_dict
            reduced_loss = 0
            reduced_loss_terms = []
            ce = reduced_loss_dict["ce"]
            if ce is not None:
                ce_tokens = torch.tensor(len(data["ce_loss_indexes"]), device=device)
                dist.all_reduce(ce_tokens, op=dist.ReduceOp.SUM)
                ce = compute_grouped_i2t_loss(ce, data, data_indexes)
                reduced_loss_dict["ce"] = ce.detach()
                ce_term = ce * training_args.ce_weight
                reduced_loss = reduced_loss + ce_term
                reduced_loss_terms.append(("ce", ce_term))
            else:
                reduced_loss_dict["ce"] = torch.tensor(0, device=device)
                ce_tokens = torch.tensor(0, device=device)
            if training_args.visual_gen and reduced_loss_dict["mse"] is not None:
                mse = reduced_loss_dict["mse"]
                mse_tokens = torch.tensor(len(data["mse_loss_indexes"]), device=device)
                dist.all_reduce(mse_tokens, op=dist.ReduceOp.SUM)
                mse = mse.mean(dim=-1).sum() * dist.get_world_size() / mse_tokens
                reduced_loss_dict["mse"] = mse.detach()
                mse_term = mse * training_args.mse_weight
                reduced_loss = reduced_loss + mse_term
                reduced_loss_terms.append(("mse", mse_term))
            else:
                reduced_loss_dict["mse"] = torch.tensor(0, device=device)
                mse_tokens = torch.tensor(0, device=device)
            return (reduced_loss, reduced_loss_terms, reduced_loss_dict, ce_tokens, mse_tokens)

        def forward_loss_terms():
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                return reduce_loss_dict(fsdp_model(**data))

        (loss, loss_terms, loss_dict, total_ce_tokens, total_mse_tokens) = forward_loss_terms()
        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        last_completed_step = curr_step
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data["sample_lens"]), device=device)
            start_allreduce_time = time()
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            allreduce_time = time() - start_allreduce_time
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                avg_loss = torch.tensor(value.item(), device=device)
                start_time_reduce = time()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                reduce_time = time() - start_time_reduce
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            if curr_step % (training_args.log_every * 10) == 0 and dist.get_rank() == 0:
                message += f"AllReduce Time: {allreduce_time * 1000:.2f}ms, "
            logger.info(message)
            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["total_mse_tokens"] = total_mse_tokens.item()
            wandb_log["total_ce_tokens"] = total_ce_tokens.item()
            wandb_log["total_norm"] = total_norm.item()
            wandb_log["total_samples"] = total_samples.item()
            wandb_log["allreduce_time_ms"] = allreduce_time * 1000
            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log["mem_allocated"] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log["mem_cache"] = mem_cache
            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()
        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item["dataset_name"] not in data_status.keys():
                data_status[item["dataset_name"]] = {}
            data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]
        if (
            curr_step > 0
            and curr_step % training_args.save_every == 0
            and (curr_step >= training_args.save_after)
        ):
            save_checkpoint(curr_step)
        curr_step += 1
    if training_args.save_final and last_completed_step >= train_step:
        save_checkpoint(last_completed_step)
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
