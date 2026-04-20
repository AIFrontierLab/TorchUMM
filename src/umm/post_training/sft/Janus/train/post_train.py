#!/usr/bin/env python
# Copyright (c) 2024-2025 DeepSeek.
# SPDX-License-Identifier: MIT

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def get_cosine_with_min_lr_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, min_lr=0.0
):
    """Cosine schedule with warmup and optional min_lr floor.

    Returns a torch.optim.lr_scheduler.LambdaLR.
    """
    from torch.optim.lr_scheduler import LambdaLR

    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if min_lr > 0:
        # Wrap step() to enforce per-group min_lr floor after update.
        orig_step = scheduler.step

        def _step(*args, **kwargs):
            orig_step(*args, **kwargs)
            for i, group in enumerate(optimizer.param_groups):
                group["lr"] = max(group["lr"], min_lr)

        scheduler.step = _step  # type: ignore[attr-defined]

    return scheduler

from janus.models import MultiModalityCausalLM, VLChatProcessor

try:
    import torchvision.transforms as T
    from PIL import Image
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires torchvision and Pillow.") from exc
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


@dataclass
class ModelArguments:
    model_path: str = field(
        default="deepseek-ai/Janus-1.3B",
        metadata={"help": "HuggingFace repo ID or local path."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Enable trust_remote_code for AutoModelForCausalLM."},
    )
    torch_dtype: str = field(
        default="bf16",
        metadata={"help": "bf16|fp16|fp32"},
    )
    freeze_vision: bool = field(
        default=True,
        metadata={"help": "Freeze vision encoder and aligner (not used in t2i)."},
    )
    freeze_gen_vision: bool = field(
        default=True,
        metadata={"help": "Freeze VQ image encoder/decoder (gen_vision_model)."},
    )
    freeze_language: bool = field(
        default=False,
        metadata={"help": "Freeze the language model weights."},
    )


@dataclass
class DataArguments:
    data_json: str = field(
        default="/sciclone/data10/yluo13/project/umm_reasoning/dataset/t2i/prompts.json",
        metadata={"help": "Path to a JSON list with fields: image, prompt."},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Optional root to prefix image paths."},
    )
    image_size: int = field(
        default=384,
        metadata={"help": "Resize + center-crop size for VQ encoder input."},
    )
    vq_downsample: int = field(
        default=16,
        metadata={"help": "VQ downsample factor (image_size / vq_downsample = token side)."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit dataset size for debugging."},
    )
    num_workers: int = field(default=4)


@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs and checkpoints."},
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Checkpoint directory."},
    )
    global_seed: int = field(default=4396)
    auto_resume: bool = field(default=False)
    resume_from: Optional[str] = field(default=None)
    resume_model_only: bool = field(default=False)

    total_steps: int = field(
        default=1000,
        metadata={"help": "Stop after this many optimizer steps."},
    )
    epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs if total_steps is not reached."},
    )
    batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=1)

    lr: float = field(default=1e-5)
    min_lr: float = field(default=1e-7)
    warmup_steps: int = field(default=100)
    lr_scheduler: str = field(default="constant")

    log_every: int = field(default=10)
    save_every: int = field(default=200)

    wandb_project: str = field(default="janus-posttrain")
    wandb_name: str = field(default="run")
    wandb_offline: bool = field(default=True)


class T2IJsonDataset(Dataset):
    def __init__(self, data_json: str, image_root: Optional[str], image_size: int, max_samples: Optional[int]):
        with open(data_json, "r") as f:
            data = json.load(f)
        if max_samples is not None:
            data = data[:max_samples]
        self.data = data
        self.base_dir = os.path.dirname(data_json)
        self.image_root = image_root
        self.transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self):
        return len(self.data)

    def _resolve_image_path(self, p: str) -> str:
        if self.image_root is not None:
            return os.path.join(self.image_root, p)
        if os.path.isabs(p):
            return p
        return os.path.join(self.base_dir, p)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        img_path = self._resolve_image_path(item["image"])
        prompt = item["prompt"]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return {
            "image": img_tensor,
            "prompt": prompt,
        }


def left_pad(input_ids: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(x) for x in input_ids)
    out = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(input_ids):
        out[i, -len(ids):] = torch.tensor(ids, dtype=torch.long)
    return out


def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def get_latest_ckpt(ckpt_dir: str) -> Optional[str]:
    if not os.path.isdir(ckpt_dir):
        return None
    step_dirs = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
    if not step_dirs:
        return None
    step_dirs = sorted(step_dirs, key=lambda x: int(x))
    return os.path.join(ckpt_dir, step_dirs[-1])


def save_checkpoint(save_dir, step, model, optimizer, scheduler):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, f"{step:07d}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
    if scheduler is not None:
        torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    distributed, rank, world_size, local_rank = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if rank == 0:
        os.makedirs(train_args.output_dir, exist_ok=True)
        os.makedirs(train_args.checkpoint_dir, exist_ok=True)
        if wandb is None:
            print("wandb not available; proceeding without W&B logging.")
        else:
            wandb.init(
                project=train_args.wandb_project,
                name=train_args.wandb_name,
                mode="offline" if train_args.wandb_offline else "online",
            )
            wandb.config.update(model_args, allow_val_change=True)
            wandb.config.update(data_args, allow_val_change=True)
            wandb.config.update(train_args, allow_val_change=True)

    seed = train_args.global_seed * world_size + rank
    set_seed(seed)

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(model_args.torch_dtype, torch.bfloat16)

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    pad_id = vl_chat_processor.pad_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    if model_args.freeze_vision:
        for p in model.vision_model.parameters():
            p.requires_grad = False
        for p in model.aligner.parameters():
            p.requires_grad = False

    if model_args.freeze_gen_vision:
        for p in model.gen_vision_model.parameters():
            p.requires_grad = False

    if model_args.freeze_language:
        for p in model.language_model.parameters():
            p.requires_grad = False

    model.train()
    if model_args.freeze_gen_vision:
        model.gen_vision_model.eval()

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    raw_model = model.module if distributed else model

    dataset = T2IJsonDataset(
        data_args.data_json,
        data_args.image_root,
        data_args.image_size,
        data_args.max_samples,
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    def collate_fn(batch):
        prompts = [b["prompt"] for b in batch]
        images = torch.stack([b["image"] for b in batch], dim=0)
        input_ids = []
        for p in prompts:
            conv = [
                {"role": "User", "content": p},
                {"role": "Assistant", "content": ""},
            ]
            sft = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conv,
                sft_format=vl_chat_processor.sft_format,
                system_prompt="",
            )
            prompt = sft + vl_chat_processor.image_start_tag
            input_ids.append(tokenizer.encode(prompt))
        input_ids = left_pad(input_ids, pad_id)
        attention_mask = (input_ids != pad_id).long()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
        }

    loader = DataLoader(
        dataset,
        batch_size=train_args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
    )
    if train_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_args.warmup_steps,
            num_training_steps=train_args.total_steps,
            min_lr=train_args.min_lr,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=train_args.warmup_steps,
        )

    start_step = 0
    if train_args.auto_resume and train_args.resume_from is None:
        train_args.resume_from = get_latest_ckpt(train_args.checkpoint_dir)

    if train_args.resume_from is not None and os.path.exists(train_args.resume_from):
        if rank == 0:
            print(f"Resuming from {train_args.resume_from}")
        model_state = torch.load(os.path.join(train_args.resume_from, "model.pt"), map_location="cpu")
        model.load_state_dict(model_state, strict=False)
        if not train_args.resume_model_only:
            optimizer_state = torch.load(os.path.join(train_args.resume_from, "optimizer.pt"), map_location="cpu")
            optimizer.load_state_dict(optimizer_state)
            sched_path = os.path.join(train_args.resume_from, "scheduler.pt")
            if os.path.exists(sched_path):
                scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
            start_step = int(os.path.basename(train_args.resume_from))

    image_token_side = data_args.image_size // data_args.vq_downsample
    expected_num_tokens = image_token_side * image_token_side
    if rank == 0:
        print(f"Image tokens per image: {expected_num_tokens}")

    total_steps = train_args.total_steps
    global_step = start_step
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    for epoch in range(train_args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if global_step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            images = batch["images"].to(device, dtype=torch.float32, non_blocking=True)
            vq_dtype = raw_model.gen_vision_model.encoder.conv_in.weight.dtype
            images = images.to(dtype=vq_dtype)

            encode_no_grad = not any(p.requires_grad for p in raw_model.gen_vision_model.parameters())
            if encode_no_grad:
                with torch.no_grad():
                    quant, _, info = raw_model.gen_vision_model.encode(images)
            else:
                quant, _, info = raw_model.gen_vision_model.encode(images)
            min_encoding_indices = info[2]
            _, _, h, w = quant.shape
            image_tokens = min_encoding_indices.view(images.size(0), h * w)

            if image_tokens.size(1) != expected_num_tokens:
                raise ValueError(
                    f"Mismatch in image tokens: got {image_tokens.size(1)} expected {expected_num_tokens}. "
                    "Check image_size and vq_downsample."
                )

            lm = raw_model.language_model
            prompt_embeds = lm.get_input_embeddings()(input_ids)

            image_input_tokens = image_tokens[:, :-1]
            image_embeds = raw_model.prepare_gen_img_embeds(image_input_tokens.reshape(-1))
            image_embeds = image_embeds.view(image_tokens.size(0), -1, image_embeds.size(-1))

            inputs_embeds = torch.cat([prompt_embeds, image_embeds], dim=1)
            attn_image = torch.ones((image_tokens.size(0), image_embeds.size(1)), device=device, dtype=attention_mask.dtype)
            attention_mask_ext = torch.cat([attention_mask, attn_image], dim=1)

            with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch_dtype):
                outputs = lm.model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask_ext,
                    use_cache=False,
                )
            hidden = outputs.last_hidden_state

            last_prompt_hidden = hidden[:, input_ids.size(1) - 1, :]
            image_hidden = hidden[:, input_ids.size(1):, :]
            hidden_for_logits = torch.cat([last_prompt_hidden.unsqueeze(1), image_hidden], dim=1)

            logits = raw_model.gen_head(hidden_for_logits)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                image_tokens.reshape(-1),
                reduction="mean",
            )

            loss = loss / train_args.gradient_accumulation_steps
            loss.backward()

            if (global_step + 1) % train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if global_step % train_args.log_every == 0 and rank == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                steps_per_sec = train_args.log_every / elapsed
                lr = optimizer.param_groups[0]["lr"]
                mem_alloc = None
                mem_reserved = None
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                    mem_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
                print(
                    f"step={global_step:07d} loss={loss.item():.4f} "
                    f"lr={lr:.3e} steps/s={steps_per_sec:.2f}",
                    flush=True,
                )
                if wandb is not None:
                    log_dict = {
                        "loss": loss.item(),
                        "lr": lr,
                        "steps_per_sec": steps_per_sec,
                    }
                    if mem_alloc is not None:
                        log_dict["gpu_mem_allocated_mb"] = mem_alloc
                        log_dict["gpu_mem_reserved_mb"] = mem_reserved
                    wandb.log(log_dict, step=global_step)
                start_time = time.time()

            if global_step > 0 and global_step % train_args.save_every == 0:
                if rank == 0:
                    save_checkpoint(train_args.checkpoint_dir, global_step, model.module if distributed else model, optimizer, scheduler)

            global_step += 1

        if global_step >= total_steps:
            break

    if rank == 0:
        save_checkpoint(train_args.checkpoint_dir, global_step, model.module if distributed else model, optimizer, scheduler)
        if wandb is not None:
            wandb.finish()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
