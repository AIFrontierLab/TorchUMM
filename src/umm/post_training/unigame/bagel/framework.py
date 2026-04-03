from __future__ import annotations

import math
import random
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model

from umm.backbones.bagel.adapter import BagelBackbone


def _to_pil_batch(images: torch.Tensor | list[Any]) -> list[Image.Image]:
    if torch.is_tensor(images):
        x = images.detach().to(torch.float32).cpu().clamp(0, 1)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        out: list[Image.Image] = []
        for i in range(x.size(0)):
            arr = (x[i].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
            out.append(Image.fromarray(arr, "RGB"))
        return out

    out = []
    for item in images:
        if isinstance(item, Image.Image):
            out.append(item.convert("RGB"))
        elif torch.is_tensor(item):
            arr = (
                item.detach().to(torch.float32).cpu().clamp(0, 1).permute(1, 2, 0) * 255
            ).to(torch.uint8).numpy()
            out.append(Image.fromarray(arr, "RGB"))
        else:
            raise TypeError(f"Unsupported image item type: {type(item)}")
    return out


def _pad_1d(x: torch.Tensor, L: int, pad_val: int) -> torch.Tensor:
    return F.pad(x, (0, L - x.shape[0]), value=pad_val)


def _pad_2d(x: torch.Tensor, L: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, L - x.shape[0]), value=0.0)


def _prepare_attention_mask_per_sample(split_lens: list[int], attn_modes: list[str], device: torch.device) -> torch.Tensor:
    sample_len = sum(split_lens)
    attention_mask = torch.zeros((sample_len, sample_len), dtype=torch.bool, device=device)

    csum = 0
    for s, attn_mode in zip(split_lens, attn_modes):
        if attn_mode == "causal":
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s), device=device, dtype=torch.bool).tril()
            attention_mask[csum : csum + s, :csum] = 1
        else:
            attention_mask[csum : csum + s, csum : csum + s] = torch.ones((s, s), device=device, dtype=torch.bool)
            attention_mask[csum : csum + s, :csum] = 1
        csum += s

    attention_mask = torch.zeros_like(attention_mask, dtype=torch.float32).masked_fill_(~attention_mask, float("-inf"))
    return attention_mask


class PerturbNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int = 0,
        eps_max: float = 0.02,
        min_eps: float = 0.0,
        eps_init: float = 0.01,
    ):
        super().__init__()
        self.eps_max = float(eps_max)
        self.min_eps = float(min_eps)

        ratio = max(1e-6, min(1 - 1e-6, float(eps_init) / max(self.eps_max, 1e-12)))
        self.log_eps = nn.Parameter(torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32))

        hidden_dim = int(width) if width and width > 0 else in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, in_dim, bias=False),
        )
        nn.init.orthogonal_(self.mlp[-1].weight)

    def _current_eps(self) -> torch.Tensor:
        eps01 = torch.sigmoid(self.log_eps)
        return self.min_eps + (self.eps_max - self.min_eps) * eps01

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        direction = F.normalize(self.mlp(x32), dim=-1)
        eps = self._current_eps().to(x32.dtype)
        delta = eps * direction
        return delta.to(x.dtype)


def _guess_lora_targets(lm: nn.Module) -> List[str]:
    cands = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }
    found = set()
    for name, _ in lm.named_modules():
        base = name.split(".")[-1]
        if base in cands:
            found.add(base)
    if not found:
        for name, _ in lm.named_modules():
            if name.endswith("W_pack"):
                found.add("W_pack")
                break
    return sorted(found) if found else ["q_proj", "v_proj"]


def _iter_named_parameters(module: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return list(module.named_parameters())


def _infer_num_layers_from_names(named_params: list[tuple[str, nn.Parameter]]) -> int:
    max_idx = -1
    for name, _ in named_params:
        for match in re.finditer(r"(?:^|\.)layers\.(\d+)\.", name):
            max_idx = max(max_idx, int(match.group(1)))
    return max_idx + 1


class BagelUniGameFramework(nn.Module):
    def __init__(
        self,
        model_path: str,
        bagel_root: str | None = None,
        max_mem_per_gpu: str = "80GiB",
        offload_folder: str = "./tmp/offload",
        distributed_single_gpu: bool = False,
        eps_max: float = 0.02,
        finetune_last_k: int = 0,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        use_lora: bool = True,
        train_lm_head: bool = False,
        train_norms_only: bool = False,
        train_last_k_norms: int = 0,
    ) -> None:
        super().__init__()

        adapter = BagelBackbone(
            model_path=model_path,
            bagel_root=bagel_root,
            max_mem_per_gpu=max_mem_per_gpu,
            offload_folder=offload_folder,
            distributed_single_gpu=distributed_single_gpu,
        )
        adapter.load(
            {
                "model_path": model_path,
                "bagel_root": bagel_root,
                "max_mem_per_gpu": max_mem_per_gpu,
                "offload_folder": offload_folder,
                "distributed_single_gpu": distributed_single_gpu,
            }
        )

        self.adapter = adapter
        self.inferencer = adapter.inferencer
        self.mm = self.inferencer.model
        self.tokenizer = self.inferencer.tokenizer
        self.new_token_ids = self.inferencer.new_token_ids
        self.train_lm_head = bool(train_lm_head)
        self.train_norms_only = bool(train_norms_only)

        # Try to infer a stable device from language model embeddings.
        self.device = self.mm.language_model.model.embed_tokens.weight.device

        for p in self.mm.parameters():
            p.requires_grad = False

        if use_lora:
            target = _guess_lora_targets(self.mm.language_model)
            lcfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                bias="none",
                target_modules=target,
                task_type="CAUSAL_LM",
            )
            self.mm.language_model = get_peft_model(self.mm.language_model, lcfg)

        named_lm_params = _iter_named_parameters(self.mm.language_model)

        if self.train_lm_head:
            for name, p in named_lm_params:
                if name.endswith("lm_head.weight") or name.startswith("lm_head."):
                    p.requires_grad = True

        if finetune_last_k > 0:
            num_layers = _infer_num_layers_from_names(named_lm_params)
            keep_from = max(0, num_layers - int(finetune_last_k))
            for name, p in named_lm_params:
                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if match and int(match.group(1)) >= keep_from:
                    p.requires_grad = True

        if self.train_norms_only:
            num_layers = _infer_num_layers_from_names(named_lm_params)
            keep_from = max(0, num_layers - int(train_last_k_norms))
            norm_suffixes = (
                ".input_layernorm.",
                ".input_layernorm_moe_gen.",
                ".post_attention_layernorm.",
                ".post_attention_layernorm_moe_gen.",
            )

            for name, p in named_lm_params:
                if (
                    name.endswith(".norm.weight")
                    or name.endswith(".norm.bias")
                    or name.endswith(".norm_moe_gen.weight")
                    or name.endswith(".norm_moe_gen.bias")
                ):
                    p.requires_grad = True
                    continue

                match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
                if not match:
                    continue
                if int(match.group(1)) < keep_from:
                    continue
                if any(suffix in name for suffix in norm_suffixes):
                    p.requires_grad = True

        # Bagel decoder modules dispatch between train/inference paths based on
        # module.training, so force the language-model subtree back to train mode.
        self.mm.language_model.train()
        if hasattr(self.mm.language_model, "model"):
            self.mm.language_model.model.train()

        self.perturb = PerturbNet(self.mm.hidden_size, eps_max=eps_max)
        self.perturb = self.perturb.to(self.device, dtype=torch.float32)
        self.gen_params = list(self.perturb.parameters())

        self._adv_flag = False
        self.reg_enable = True
        self.reg_l2 = 0.0
        self.reg_tv = 0.0
        self.reg_eps = 0.0

        self.img_gen_size = 512
        self.img_patch_size = 16
        self.cfg_weight = 4.0
        self.gen_temperature = 1.0

    @staticmethod
    def _set_all_seeds(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    @contextmanager
    def _temporary_inference_mode(self):
        modules = [self.mm.language_model]
        if hasattr(self.mm.language_model, "model"):
            modules.append(self.mm.language_model.model)
        prev_states = [m.training for m in modules]
        try:
            for m in modules:
                m.eval()
            yield
        finally:
            for m, state in zip(modules, prev_states):
                m.train(state)

    def _build_vit_features(self, pil_batch: list[Image.Image]) -> list[torch.Tensor]:
        mm = self.mm
        img_inputs: list[Image.Image] = []
        for im in pil_batch:
            # Mirror inferencer path: resize through VAE transform before image updates.
            img_inputs.append(self.inferencer.vae_transform.resize_transform(im.convert("RGB")))

        generation_input, _, _ = mm.prepare_vit_images(
            curr_kvlens=[0 for _ in img_inputs],
            curr_rope=[0 for _ in img_inputs],
            images=img_inputs,
            transforms=self.inferencer.vit_transform,
            new_token_ids=self.new_token_ids,
        )

        vit_dtype = next(mm.vit_model.parameters()).dtype
        packed_vit_tokens = generation_input["packed_vit_tokens"].to(self.device, dtype=vit_dtype)
        packed_vit_position_ids = generation_input["packed_vit_position_ids"].to(self.device)
        vit_token_seqlens = generation_input["vit_token_seqlens"].to(self.device)

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(torch.max(vit_token_seqlens).item())

        packed_vit_token_embed = mm.vit_model(
            packed_pixel_values=packed_vit_tokens,
            packed_flattened_position_ids=packed_vit_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        packed_vit_token_embed = mm.connector(packed_vit_token_embed)
        pos_emb = mm.vit_pos_embed(packed_vit_position_ids)
        packed_vit_token_embed = packed_vit_token_embed + pos_emb

        splits = vit_token_seqlens.detach().cpu().tolist()
        return list(torch.split(packed_vit_token_embed, splits, dim=0))

    def _assemble_batch(
        self,
        image_features: list[torch.Tensor],
        questions: list[str],
        answers: list[str],
    ) -> tuple[torch.Tensor, list[int], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        tok = self.tokenizer
        lm = self.mm.language_model
        if hasattr(lm, "get_input_embeddings"):
            embed_tokens = lm.get_input_embeddings()
        elif hasattr(lm, "model") and hasattr(lm.model, "embed_tokens"):
            embed_tokens = lm.model.embed_tokens
        else:
            raise AttributeError(f"Cannot locate input embeddings on language model type={type(lm)}")
        dtype = embed_tokens.weight.dtype
        dev = embed_tokens.weight.device

        start_img_id = int(self.new_token_ids["start_of_image"])
        end_img_id = int(self.new_token_ids["end_of_image"])
        bos_id = int(self.new_token_ids.get("bos_token_id", tok.bos_token_id))
        eos_id = int(self.new_token_ids.get("eos_token_id", tok.eos_token_id))

        packed_sequences: list[torch.Tensor] = []
        sample_lens: list[int] = []
        attention_masks: list[torch.Tensor] = []
        packed_position_ids: list[torch.Tensor] = []
        packed_ce_indexes: list[torch.Tensor] = []
        packed_label_ids: list[torch.Tensor] = []

        curr = 0
        for img_feat, q, a in zip(image_features, questions, answers):
            q_ids = tok.encode(q, add_special_tokens=False)
            a_ids = tok.encode((a or "") + (tok.eos_token or ""), add_special_tokens=False)
            if len(a_ids) == 0:
                a_ids = [eos_id]
            text_input_ids = [bos_id] + q_ids + [eos_id] + a_ids[:-1]
            context_len = 1 + len(q_ids) + 1
            text_len = len(text_input_ids)

            text_t = torch.tensor(text_input_ids, dtype=torch.long, device=dev)
            si_t = torch.tensor([start_img_id], dtype=torch.long, device=dev)
            ei_t = torch.tensor([end_img_id], dtype=torch.long, device=dev)

            text_e = embed_tokens(text_t)
            si_e = embed_tokens(si_t)
            ei_e = embed_tokens(ei_t)

            img_e = img_feat.to(device=dev, dtype=dtype)
            seq = torch.cat([si_e, img_e, ei_e, text_e], dim=0)
            packed_sequences.append(seq)

            img_block_len = 1 + img_e.size(0) + 1
            total_len = img_block_len + text_len
            sample_lens.append(total_len)
            attention_masks.append(
                _prepare_attention_mask_per_sample(
                    split_lens=[img_block_len, text_len],
                    attn_modes=["full", "causal"],
                    device=dev,
                )
            )
            packed_position_ids.append(
                torch.tensor(
                    [0] * img_block_len + list(range(1, 1 + text_len)),
                    dtype=torch.long,
                    device=dev,
                )
            )
            packed_ce_indexes.append(
                curr + img_block_len + torch.arange(context_len - 1, context_len - 1 + len(a_ids), device=dev)
            )
            packed_label_ids.append(torch.tensor(a_ids, dtype=torch.long, device=dev))
            curr += total_len

        packed_sequence = torch.cat(packed_sequences, dim=0)
        packed_position_ids_t = torch.cat(packed_position_ids, dim=0)
        packed_ce_indexes_t = torch.cat(packed_ce_indexes, dim=0)
        packed_label_ids_t = torch.cat(packed_label_ids, dim=0)
        packed_und_token_indexes_t = torch.arange(packed_sequence.size(0), device=dev, dtype=torch.long)

        return (
            packed_sequence,
            sample_lens,
            attention_masks,
            packed_position_ids_t,
            packed_und_token_indexes_t,
            packed_ce_indexes_t,
            packed_label_ids_t,
        )

    def forward(self, batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        images = _to_pil_batch(batch["image"])
        questions = batch["question"]
        answers = batch.get("answer_text")
        if answers is None:
            raise ValueError("`answer_text` is required for Bagel UniGame forward.")

        img_feats_list = self._build_vit_features(images)

        aux: Dict[str, torch.Tensor] = {}
        if getattr(self, "_adv_flag", False):
            adv_feats = []
            deltas = []
            for feats in img_feats_list:
                delta = self.perturb(feats.detach().float())
                deltas.append(delta)
                adv_feats.append(feats + delta.to(feats.dtype))
            img_feats_list = adv_feats

            if getattr(self, "reg_enable", False):
                all_delta = torch.cat(deltas, dim=0)
                reg_l2 = all_delta.pow(2).mean()
                if all_delta.dim() == 2 and all_delta.size(0) > 1:
                    reg_tv = (all_delta[1:, :] - all_delta[:-1, :]).pow(2).mean()
                else:
                    reg_tv = all_delta.new_tensor(0.0)
                eps01 = torch.sigmoid(self.perturb.log_eps)
                reg_eps = eps01 * eps01
                reg_total = (
                    float(getattr(self, "reg_l2", 0.0)) * reg_l2
                    + float(getattr(self, "reg_tv", 0.0)) * reg_tv
                    + float(getattr(self, "reg_eps", 0.0)) * reg_eps
                )
                aux["reg"] = reg_total
                aux["reg_l2"] = reg_l2.detach()
                aux["reg_tv"] = reg_tv.detach()
                aux["reg_eps"] = reg_eps.detach()

        (
            packed_sequence,
            sample_lens,
            attention_masks,
            packed_position_ids,
            packed_und_token_indexes,
            packed_ce_indexes,
            packed_label_ids,
        ) = self._assemble_batch(
            image_features=img_feats_list,
            questions=questions,
            answers=answers,
        )

        lm = self.mm.language_model
        lm_core = lm
        if hasattr(lm, "base_model") and hasattr(lm.base_model, "model"):
            lm_core = lm.base_model.model
        elif hasattr(lm, "get_base_model"):
            lm_core = lm.get_base_model()
        if hasattr(lm_core, "model"):
            lm_core = lm_core.model

        if not hasattr(lm_core, "forward_train"):
            raise AttributeError(f"Expected Bagel Qwen2Model with forward_train, got {type(lm_core)}")

        last_hidden_state = lm_core.forward_train(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_masks,
            packed_position_ids=packed_position_ids,
            packed_und_token_indexes=packed_und_token_indexes,
        )
        logits = lm.lm_head(last_hidden_state[packed_ce_indexes])
        loss = F.cross_entropy(logits, packed_label_ids, reduction="mean")
        return loss, aux

    @torch.no_grad()
    def infer_answers_batch(
        self,
        images: torch.Tensor | list[Any],
        questions: list[str],
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> list[str]:
        pil_batch = _to_pil_batch(images)
        preds: list[str] = []

        with self._temporary_inference_mode():
            for im, q in zip(pil_batch, questions):
                out = self.inferencer(
                    image=im,
                    text=q,
                    understanding_output=True,
                    max_think_token_n=int(max_new_tokens),
                    do_sample=bool(temperature > 0),
                    text_temperature=max(1e-6, float(temperature)),
                )
                txt = out.get("text") if isinstance(out, dict) else None
                preds.append((txt or "").strip())

        return preds

    @torch.no_grad()
    def _official_generate_batch(
        self,
        prompts: list[str],
        inject_adv: bool = False,
        seed: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        if seed is not None:
            self._set_all_seeds(int(seed))

        outs: list[torch.Tensor] = []

        orig_forward_flow = None
        if inject_adv:
            orig_forward_flow = self.mm._forward_flow

            def _adv_forward_flow(*args: Any, **kwargs: Any):
                x_t = kwargs.get("x_t")
                timestep = kwargs.get("timestep")
                packed_pos_ids = kwargs.get("packed_vae_position_ids")
                if x_t is not None and timestep is not None and packed_pos_ids is not None:
                    x_embed = self.mm.vae2llm(x_t)
                    x_embed = x_embed + self.mm.time_embedder(timestep) + self.mm.latent_pos_embed(packed_pos_ids)
                    delta = self.perturb(x_embed.detach().float()).to(x_embed.dtype)
                    x_embed = x_embed + delta
                    kwargs["x_t"] = self.mm.llm2vae(x_embed).to(x_t.dtype)
                return orig_forward_flow(*args, **kwargs)

            self.mm._forward_flow = _adv_forward_flow

        try:
            with self._temporary_inference_mode():
                for p in prompts:
                    out = self.inferencer(
                        image=None,
                        text=p,
                        understanding_output=False,
                        do_sample=True,
                        text_temperature=max(1e-6, float(self.gen_temperature)),
                        cfg_text_scale=float(self.cfg_weight),
                    )
                    img = out.get("image") if isinstance(out, dict) else None
                    if not isinstance(img, Image.Image):
                        raise RuntimeError("Bagel generation did not return image output.")

                    arr = np.array(img.convert("RGB"), dtype=np.uint8, copy=True)
                    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    outs.append(t)
        finally:
            if inject_adv and orig_forward_flow is not None:
                self.mm._forward_flow = orig_forward_flow

        return torch.stack(outs, dim=0)

    def _current_eps(self) -> torch.Tensor:
        eps01 = torch.sigmoid(self.perturb.log_eps.detach().float())
        return self.perturb.eps_max * eps01

    @torch.no_grad()
    def generate_clean_and_adv(self, prompt_text: str, seed: int = 12345, **gen_kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self._official_generate_batch([prompt_text], inject_adv=False, seed=seed, **gen_kwargs)
        adv = self._official_generate_batch([prompt_text], inject_adv=True, seed=seed, **gen_kwargs)
        return clean, adv
