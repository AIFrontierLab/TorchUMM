import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Prefer UMM codebase Janus implementation over any environment package.
_SRC_ROOT = Path(__file__).resolve().parents[4]
_JANUS_ROOT = _SRC_ROOT / "umm" / "backbones" / "janus_pro" / "Janus"
if _JANUS_ROOT.exists():
    janus_root_str = str(_JANUS_ROOT)
    if janus_root_str not in sys.path:
        sys.path.insert(0, janus_root_str)

try:
    from janus.models import MultiModalityCausalLM, VLChatProcessor
    from janus.models.processing_vlm import (
        BatchedVLChatProcessorOutput,
        VLChatProcessorOutput,
    )
except ImportError:
    raise ImportError(
        "Failed to import Janus models from codebase. Expected in-repo Janus at "
        f"{_JANUS_ROOT}."
    )
from PIL import Image
import math
import random
import numpy as np


def _ensure_transformers_tied_weights_compat() -> None:
    """Compat shim for newer transformers expecting `all_tied_weights_keys`.

    Some Janus model classes only define `_tied_weights_keys`, while recent
    transformers code paths call `model.all_tied_weights_keys.keys()`.
    """

    if hasattr(MultiModalityCausalLM, "all_tied_weights_keys"):
        return

    def _all_tied_weights_keys(self):
        raw = getattr(self, "_tied_weights_keys", None)
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (list, tuple, set)):
            return {str(k): None for k in raw}
        return {}

    # Property keeps behavior dynamic per-instance and avoids touching backbone files.
    setattr(MultiModalityCausalLM, "all_tied_weights_keys", property(_all_tied_weights_keys))

def _pad_to_len(x: torch.Tensor, L: int, pad_val: int = 0) -> torch.Tensor:
    if x.dim() == 1:
        return F.pad(x, (0, L - x.shape[0]), value=pad_val)
    else:
        return F.pad(x, (0, 0, 0, L - x.shape[0]))

def _pack_prompt_and_labels(
    tokenizer,
    questions: List[str],
    answers: List[str],
    image_token: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    input_ids, attn_mask, labels = [], [], []
    for q, a in zip(questions, answers):
        prompt = f"{image_token}\n{q}\nAnswer:"
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(a + eos, add_special_tokens=False).input_ids
        ids = p_ids + a_ids
        mask = [1] * len(ids)
        lab = [-100] * len(p_ids) + a_ids
        input_ids.append(torch.tensor(ids, dtype=torch.long))
        attn_mask.append(torch.tensor(mask, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))
    T = max(len(x) for x in input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.stack([_pad_to_len(x, T, pad_val=pad_id) for x in input_ids])
    attn_mask = torch.stack([_pad_to_len(x, T, pad_val=0) for x in attn_mask])
    labels = torch.stack([_pad_to_len(x, T, pad_val=-100) for x in labels])
    return input_ids, attn_mask, labels

def _insert_image_features(
    text_embeds: torch.Tensor,
    attn_mask: torch.Tensor,
    labels: torch.Tensor,
    image_features: torch.Tensor,
    img_pos: int,
):
    T, H = text_embeds.shape
    Np = image_features.shape[0]
    e_new = torch.cat([text_embeds[:img_pos], image_features, text_embeds[img_pos + 1 :]], dim=0)
    m_new = torch.cat(
        [
            attn_mask[:img_pos],
            torch.ones(Np, dtype=attn_mask.dtype, device=attn_mask.device),
            attn_mask[img_pos + 1 :],
        ],
        dim=0,
    )
    l_new = torch.cat(
        [
            labels[:img_pos],
            torch.full((Np,), -100, dtype=labels.dtype, device=labels.device),
            labels[img_pos + 1 :],
        ],
        dim=0,
    )
    return e_new, m_new, l_new

class PerturbNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        width: int = 0,
        depth: int = 0,
        eps_max: float = 0.05,
        min_eps: float = 0.0,
        eps_init: float = 0.01,
    ):
        super().__init__()
        self.eps_max = float(eps_max)
        self.min_eps = float(min_eps)

        ratio = max(1e-6, min(1 - 1e-6, float(eps_init) / max(self.eps_max, 1e-12)))
        logit = math.log(ratio / (1.0 - ratio))
        self.log_eps = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

        hidden_dim = int(width) if (width is not None and width > 0) else in_dim

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
        dir_raw = self.mlp(x32)
        r = F.normalize(dir_raw, dim=-1)
        eps = self._current_eps().to(x32.dtype)
        delta32 = eps * r
        return delta32.to(x.dtype)

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
    for n, _ in lm.named_modules():
        base = n.split(".")[-1]
        if base in cands:
            found.add(base)
    if not found:
        for n, _ in lm.named_modules():
            if n.endswith("W_pack"):
                found.add("W_pack")
                break
    return sorted(found) if found else ["q_proj", "v_proj"]

class JanusProAdvFramework(nn.Module):
    def __init__(
        self,
        model_path: str = "deepseek-ai/Janus-Pro-7B",
        num_answers: int = 3129,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        finetune_last_k: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        eps_max: float = 0.02,
        low_cpu_mem_usage: bool = False,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path

        self.vl_chat: VLChatProcessor = VLChatProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.tokenizer = self.vl_chat.tokenizer

        _ensure_transformers_tied_weights_compat()
        print(
            "[umm.unigame.janus_pro] tied-weights compat:",
            hasattr(MultiModalityCausalLM, "all_tied_weights_keys"),
        )

        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": False,
        }
        print(
            "[umm.unigame.janus_pro] loading model",
            f"path={model_path}",
            f"torch={torch.__version__}",
            f"transformers={transformers.__version__}",
            f"python={sys.version.split()[0]}",
            f"dtype={dtype}",
            "low_cpu_mem_usage=False(force)",
        )
        # Force one deterministic loading path across ranks.
        orig_linspace = torch.linspace

        def _linspace_cpu_fallback(start, end, steps, *args, **kwargs):
            if kwargs.get("device") is None:
                kwargs["device"] = "cpu"
            return orig_linspace(start, end, steps, *args, **kwargs)

        torch.linspace = _linspace_cpu_fallback
        try:
            self.mm: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs,
            )
        finally:
            torch.linspace = orig_linspace

        self.mm = self.mm.to(device=self.device, dtype=dtype, non_blocking=True)
        self.mm.language_model.config.use_cache = False

        for p in self.mm.parameters():
            p.requires_grad = False

        target = _guess_lora_targets(self.mm.language_model)
        lcfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target,
            task_type="CAUSAL_LM",
        )
        self.mm.language_model = get_peft_model(self.mm.language_model, lcfg)

        trainable_tensors = sum(1 for p in self.mm.language_model.parameters() if p.requires_grad)
        trainable_elems = sum(int(p.numel()) for p in self.mm.language_model.parameters() if p.requires_grad)
        print(
            "[umm.unigame.janus_pro] lora trainable",
            f"tensors={trainable_tensors}",
            f"elements={trainable_elems}",
        )

        if bool(gradient_checkpointing):
            try:
                # Reduces activation memory; slower but usually necessary on 7B training.
                if hasattr(self.mm.language_model, "gradient_checkpointing_enable"):
                    self.mm.language_model.gradient_checkpointing_enable()
                if hasattr(self.mm.language_model, "enable_input_require_grads"):
                    self.mm.language_model.enable_input_require_grads()
                print("[umm.unigame.janus_pro] gradient checkpointing: enabled")
            except Exception as exc:
                print(f"[umm.unigame.janus_pro] gradient checkpointing setup skipped: {exc}")

        if finetune_last_k > 0:
            blocks = []
            for n, m in self.mm.language_model.named_modules():
                if hasattr(m, "self_attn") or "mlp" in n:
                    blocks.append((n, m))
            if len(blocks) > finetune_last_k:
                for n, m in blocks[:-finetune_last_k]:
                    for p in m.parameters():
                        p.requires_grad = False

        def _infer_proj_out_dim(proj_mod: nn.Module) -> int:
            if isinstance(proj_mod, nn.Linear):
                return proj_mod.out_features
            for m in proj_mod.modules():
                if isinstance(m, nn.Linear):
                    return m.out_features
            raise RuntimeError("Cannot infer projector output dim.")

        proj = (
            getattr(self.mm, "aligner", None)
            or getattr(self.mm, "visual_projector", None)
            or getattr(self.mm, "mm_projector", None)
            or getattr(self.mm, "projector", None)
        )
        assert proj is not None, "Cannot find aligner/mm_projector/visual_projector on model."
        H_llm = _infer_proj_out_dim(proj)
        self.perturb = PerturbNet(H_llm, eps_max=eps_max)

        self.gen_params = list(self.perturb.parameters())

        self._adv_flag = False

        self.to(self.device, dtype=self.mm.language_model.dtype)
        self.perturb = self.perturb.to(self.device, dtype=torch.float32)
        for p in self.perturb.parameters():
            p.data = p.data.float()

        self.reg_enable = True
        self.reg_l2 = 0.0
        self.reg_tv = 0.0
        self.reg_eps = 0.0
        self.max_answer_tokens = 0

    @staticmethod
    def _set_all_seeds(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _get_codebook_dim(self) -> int:
        gv = getattr(self.mm, "gen_vision_model", None)
        if gv is None:
            return 4
        if hasattr(gv, "quantize"):
            for attr in ("embedding_dim", "embed_dim", "e_dim", "dim"):
                if hasattr(gv.quantize, attr) and isinstance(getattr(gv.quantize, attr), (int,)):
                    return int(getattr(gv.quantize, attr))
        for attr in ("codebook_dim", "z_channels", "embed_dim", "code_ch"):
            if hasattr(gv, attr) and isinstance(getattr(gv, attr), (int,)):
                return int(getattr(gv, attr))
        return 4

    @torch.no_grad()
    def _official_generate_batch(
        self,
        prompts: List[str],
        *,
        inject_adv: bool = False,
        seed: Optional[int] = None,
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        if seed is not None:
            self._set_all_seeds(int(seed))

        mm, vlcp = self.mm, self.vl_chat
        img_size = img_size or getattr(self, "img_gen_size", 384)
        patch_size = patch_size or getattr(self, "img_patch_size", 16)
        cfg_weight = cfg_weight or getattr(self, "cfg_weight", 5.0)
        temperature = temperature or getattr(self, "gen_temperature", 1.0)

        cond_ids = []
        for pt in prompts:
            sft = vlcp.apply_sft_template_for_multi_turn_prompts(
                conversations=[
                    {"role": "<|User|>", "content": pt},
                    {"role": "<|Assistant|>", "content": ""},
                ],
                sft_format=vlcp.sft_format,
                system_prompt="",
            )
            ids = vlcp.tokenizer.encode(sft + vlcp.image_start_tag)
            cond_ids.append(torch.tensor(ids, dtype=torch.long))

        Lmax = max(x.numel() for x in cond_ids)
        pad_id = vlcp.pad_id
        B = len(cond_ids)
        tokens = torch.full((B * 2, Lmax), pad_id, dtype=torch.long, device=self.device)
        for i, ids in enumerate(cond_ids):
            L = ids.numel()
            tokens[2 * i, :L] = ids.to(self.device)
            tokens[2 * i + 1, :L] = ids.to(self.device)
            if L > 2:
                tokens[2 * i + 1, 1 : L - 1] = pad_id

        inputs_embeds = mm.language_model.get_input_embeddings()(tokens)

        Hc = Wc = img_size // patch_size
        T = Hc * Wc
        gen_tokens = torch.zeros((B, T), dtype=torch.int, device=self.device)

        pkv = None
        for t in range(T):
            outputs = mm.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=pkv,
                output_hidden_states=True,
            )
            hs = outputs.hidden_states[-1]
            pkv = outputs.past_key_values

            logits = mm.gen_head(hs[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + (cfg_weight * (logit_cond - logit_uncond))

            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            gen_tokens[:, t] = next_token.squeeze(-1)

            pair = torch.stack([next_token, next_token], dim=1).reshape(-1)
            img_embeds = mm.prepare_gen_img_embeds(pair)
            if inject_adv:
                delta = self.perturb(img_embeds.detach().float())
                img_embeds = img_embeds + delta.to(img_embeds.dtype)
            inputs_embeds = img_embeds.unsqueeze(1)

        code_ch = self._get_codebook_dim()
        dec = mm.gen_vision_model.decode_code(
            gen_tokens.to(torch.int), shape=[B, code_ch, Hc, Wc]
        )
        if isinstance(dec, torch.Tensor):
            img = (dec.clamp(-1, 1) + 1) / 2.0
        else:
            img = torch.from_numpy(np.clip((dec + 1) / 2.0, 0, 1)).to(self.device)

        if img.ndim == 4 and img.shape[1] in (1, 3):
            pass
        elif img.ndim == 4 and img.shape[-1] in (1, 3):
            img = img.permute(0, 3, 1, 2).contiguous()
        else:
            raise RuntimeError("Unexpected decoded image shape")

        return img.detach().to(torch.float32).clamp(0, 1)

    @torch.no_grad()
    def _official_generate_one(
        self,
        prompt_text: str,
        *,
        inject_adv: bool = False,
        seed: Optional[int] = None,
        parallel: int = 1,
        img_size: Optional[int] = None,
        patch_size: Optional[int] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        imgs = self._official_generate_batch(
            [prompt_text],
            inject_adv=inject_adv,
            seed=seed,
            img_size=img_size,
            patch_size=patch_size,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )
        return imgs[0:1]

    @torch.no_grad()
    def generate_clean_and_adv(
        self,
        prompt_text: str,
        seed: int = 12345,
        **gen_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        clean = self._official_generate_one(
            prompt_text, inject_adv=False, seed=seed, **gen_kwargs
        )
        adv = self._official_generate_one(
            prompt_text, inject_adv=True, seed=seed, **gen_kwargs
        )
        return clean, adv

    @torch.no_grad()
    def _make_decoded_images(self, batch: Dict[str, torch.Tensor], mode: str = "adv_only"):
        imgs_in = batch["image"]
        if torch.is_tensor(imgs_in):
            if imgs_in.dim() == 3:
                B, H, W = 1, imgs_in.shape[-2], imgs_in.shape[-1]
            else:
                B, H, W = imgs_in.shape[0], imgs_in.shape[-2], imgs_in.shape[-1]
        elif isinstance(imgs_in, list) and len(imgs_in) > 0:
            B = len(imgs_in)
            if isinstance(imgs_in[0], Image.Image):
                W, H = imgs_in[0].size
            elif torch.is_tensor(imgs_in[0]):
                H, W = imgs_in[0].shape[-2], imgs_in[0].shape[-1]
            else:
                raise TypeError(f"Unsupported image type inside list: {type(imgs_in[0])}")
        else:
            raise TypeError(f"Unsupported type for batch['image']: {type(imgs_in)}")

        qs = batch.get("question", None)
        if isinstance(qs, list):
            prompts = [str(s) if s is not None else "" for s in qs]
        elif qs is None:
            prompts = [""] * B
        else:
            prompts = [str(qs)] * B

        was_training = self.mm.training
        self.mm.eval()
        try:
            if mode == "adv_only":
                imgs_adv = self._official_generate_batch(
                    prompts,
                    inject_adv=True,
                    seed=114514 + random.randint(0, 1_000_000),
                    img_size=getattr(self, "img_gen_size", 256),
                    patch_size=getattr(self, "img_patch_size", 16),
                    cfg_weight=getattr(self, "cfg_weight", 5.0),
                    temperature=getattr(self, "gen_temperature", 1.0),
                )
                if imgs_adv.shape[-2:] != (H, W):
                    imgs_adv = F.interpolate(
                        imgs_adv, size=(H, W), mode="bilinear", align_corners=False
                    )
                return imgs_adv.clamp(0, 1)
            elif mode == "pair":
                imgs_clean = self._official_generate_batch(
                    prompts,
                    inject_adv=False,
                    seed=114514 + random.randint(0, 1_000_000),
                    img_size=getattr(self, "img_gen_size", 256),
                    patch_size=getattr(self, "img_patch_size", 16),
                    cfg_weight=getattr(self, "cfg_weight", 5.0),
                    temperature=getattr(self, "gen_temperature", 1.0),
                )
                imgs_adv = self._official_generate_batch(
                    prompts,
                    inject_adv=True,
                    seed=114514 + random.randint(0, 1_000_000),
                    img_size=getattr(self, "img_gen_size", 256),
                    patch_size=getattr(self, "img_patch_size", 16),
                    cfg_weight=getattr(self, "cfg_weight", 5.0),
                    temperature=getattr(self, "gen_temperature", 1.0),
                )
                if imgs_clean.shape[-2:] != (H, W):
                    imgs_clean = F.interpolate(
                        imgs_clean, size=(H, W), mode="bilinear", align_corners=False
                    )
                if imgs_adv.shape[-2:] != (H, W):
                    imgs_adv = F.interpolate(
                        imgs_adv, size=(H, W), mode="bilinear", align_corners=False
                    )
                return imgs_clean.clamp(0, 1), imgs_adv.clamp(0, 1)
            else:
                raise ValueError(f"Unknown mode={mode}")
        finally:
            if was_training:
                self.mm.train()

    @torch.no_grad()
    def _save_decoded_samples(self, clean, adv: torch.Tensor, tag: str):
        from torchvision.utils import make_grid, save_image
        import numpy as _np

        os.makedirs(getattr(self, "decoded_dir", "decoded_samples"), exist_ok=True)

        def _to_tensor_batch(imgs, size_hw):
            Ht, Wt = size_hw
            if torch.is_tensor(imgs):
                x = imgs
                if x.dim() == 3:
                    x = x.unsqueeze(0)
                x = x.clamp(0, 1).float().cpu()
                if x.shape[-2:] != (Ht, Wt):
                    x = F.interpolate(
                        x, size=(Ht, Wt), mode="bilinear", align_corners=False
                    )
                return x
            elif isinstance(imgs, list):
                lst = []
                for im in imgs[: adv.size(0)]:
                    if isinstance(im, Image.Image):
                        arr = _np.array(im.convert("RGB"), dtype=_np.uint8)
                        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                    elif torch.is_tensor(im):
                        t = im.detach().float().clamp(0, 1).cpu()
                        if t.dim() == 3 and t.size(0) == 1:
                            t = t.expand(3, -1, -1)
                        if t.dim() == 3 and t.size(0) == 4:
                            t = t[:3, ...]
                    else:
                        raise TypeError(f"Unsupported clean item: {type(im)}")
                    if t.shape[-2:] != (Ht, Wt):
                        t = F.interpolate(
                            t.unsqueeze(0),
                            size=(Ht, Wt),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0)
                    lst.append(t)
                return torch.stack(lst, dim=0)
            else:
                raise TypeError(f"Unsupported clean container: {type(imgs)}")

        adv_cpu = adv.clamp(0, 1).cpu()
        clean_cpu = _to_tensor_batch(clean, adv_cpu.shape[-2:])
        pairs = torch.cat([clean_cpu, adv_cpu], dim=-2)
        nrow = min(getattr(self, "save_decoded_n", 2), pairs.size(0))
        grid = make_grid(pairs[:nrow], nrow=nrow, padding=2)
        out_dir = getattr(self, "decoded_dir", "decoded_samples")
        save_image(grid, os.path.join(out_dir, f"{tag}.png"))

    @torch.no_grad()
    def _build_txt_embeds(self, texts: List[str]) -> torch.Tensor:
        ids = [self.tokenizer.encode(t, add_special_tokens=True) for t in texts]
        maxL = max(len(x) for x in ids)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        ids_ = torch.tensor(
            [x + [pad_id] * (maxL - len(x)) for x in ids],
            dtype=torch.long,
            device=self.device,
        )
        text_embeds = self.mm.language_model.get_input_embeddings()(ids_)
        return text_embeds, ids_

    @torch.no_grad()
    def infer_answers_batch(
        self,
        images,
        questions: List[str],
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        eos_token: str = None,
    ) -> List[str]:
        pil_batch = []
        if torch.is_tensor(images):
            x = images.detach().to(torch.float32).cpu().clamp(0, 1)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            for i in range(x.size(0)):
                arr = (x[i].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
                pil_batch.append(Image.fromarray(arr, "RGB"))
        else:
            for im in images:
                pil_batch.append(im.convert("RGB") if isinstance(im, Image.Image) else im)

        prepares = []
        for im, q in zip(pil_batch, questions):
            conv = [
                {"role": "<|User|>", "content": "<image_placeholder>\n" + q, "images": [im]},
                {"role": "<|Assistant|>", "content": ""},
            ]
            prep = self.vl_chat.process_one(conversations=conv, images=[im])
            prepares.append(prep)
        batched = self.vl_chat.batchify(prepares).to(self.device)
        tok = self.vl_chat.tokenizer
        lm = self.mm.language_model

        # Use Janus official multimodal embedding path to avoid image-token
        # misalignment from manual placeholder replacement.
        inputs_embeds = self.mm.prepare_inputs_embeds(**batched).to(lm.dtype)
        attention_mask = batched.attention_mask

        do_sample = bool(float(temperature) > 0)
        gen_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "max_new_tokens": int(max_new_tokens),
            "do_sample": do_sample,
            "use_cache": True,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(1e-6, float(temperature))

        gen_out = lm.generate(**gen_kwargs)

        preds = []
        for i in range(gen_out.size(0)):
            Lp = int(attention_mask[i].sum().item())
            seq = gen_out[i]
            # With `inputs_embeds`, some transformers versions return only newly
            # generated tokens (without prompt prefix). In that case, slicing by
            # prompt length would drop everything and produce empty predictions.
            if seq.size(0) > Lp:
                dec_ids = seq[Lp:]
            else:
                dec_ids = seq
            text = tok.decode(
                dec_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            # Some tokenizer/model combinations may emit visible BPE markers
            # (e.g. "Ġ"/"▁") in raw strings; normalize them to plain spaces.
            text = text.replace("Ġ", " ").replace("▁", " ")
            text = " ".join(text.split())
            preds.append(text.strip())
        return preds

    def forward(self, batch=None, image=None, question=None, answer=None):
        if batch is not None:
            imgs_in = batch["image"]
            questions = batch["question"]
            answers_txt = batch.get("answer_text", None)
        else:
            imgs_in = image
            questions = [question] if isinstance(question, str) else question
            answers_txt = [answer] if isinstance(answer, str) else answer
        assert answers_txt is not None, "forward 需要 answer_text（生成式标签）。"
        B = len(questions)

        pil_batch = []
        if torch.is_tensor(imgs_in):
            x = imgs_in.detach().to(torch.float32).cpu().clamp(0, 1)
            if x.dim() == 3:
                x = x.unsqueeze(0)
            for i in range(x.size(0)):
                arr = (x[i].permute(1, 2, 0) * 255).to(torch.uint8).numpy()
                pil_batch.append(Image.fromarray(arr, "RGB"))
        elif isinstance(imgs_in, list):
            for im in imgs_in:
                if isinstance(im, Image.Image):
                    pil_batch.append(im.convert("RGB"))
                elif torch.is_tensor(im):
                    arr = (
                        im.detach()
                        .to(torch.float32)
                        .cpu()
                        .clamp(0, 1)
                        .permute(1, 2, 0)
                        * 255
                    ).to(torch.uint8).numpy()
                    pil_batch.append(Image.fromarray(arr, "RGB"))
                else:
                    raise TypeError(f"Unsupported image type: {type(im)}")
        else:
            raise TypeError(f"Unsupported image container type: {type(imgs_in)}")

        prepares = []
        for im, q in zip(pil_batch, questions):
            conv = [
                {
                    "role": "<|User|>",
                    "content": "<image_placeholder>\n" + q,
                    "images": [im],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            prep = self.vl_chat.process_one(conversations=conv, images=[im])
            prepares.append(prep)
        batched = self.vl_chat.batchify(prepares).to(self.device)

        tok = self.vl_chat.tokenizer
        lm = self.mm.language_model

        txt_embeds_prompt = lm.get_input_embeddings()(batched.input_ids)
        attn_mask_prompt = batched.attention_mask

        proc = getattr(self.vl_chat, "image_processor", None) or getattr(
            self.mm, "image_processor", None
        )
        px = proc.preprocess(pil_batch, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=next(self.mm.parameters()).dtype
        )
        vt = getattr(self.mm, "vision_model", None)
        if vt is None and hasattr(self.mm, "get_vision_tower"):
            vt = self.mm.get_vision_tower()
        proj = (
            getattr(self.mm, "aligner", None)
            or getattr(self.mm, "visual_projector", None)
            or getattr(self.mm, "mm_projector", None)
            or getattr(self.mm, "projector", None)
        )
        assert vt is not None and proj is not None, "找不到 vision_tower 或 projector"

        with torch.no_grad():
            vt_feats = vt(images=px)
            img_feats = proj(vt_feats)

        aux: Dict[str, torch.Tensor] = {}
        if getattr(self, "_adv_flag", False):
            delta = self.perturb(img_feats.detach().float())

            if getattr(self, "reg_enable", False):
                reg_l2 = delta.pow(2).mean()
                if delta.dim() == 3 and delta.size(1) > 1:
                    reg_tv = (delta[:, 1:, :] - delta[:, :-1, :]).pow(2).mean()
                else:
                    reg_tv = delta.new_tensor(0.0)
                eps01 = torch.sigmoid(self.perturb.log_eps)
                reg_eps = eps01 * eps01

                reg_total = (
                    float(getattr(self, "reg_l2", 0.0)) * reg_l2
                    + float(getattr(self, "reg_tv", 0.0)) * reg_tv
                    + float(getattr(self, "reg_eps", 0.0)) * reg_eps
                )
                aux["reg"] = reg_total.to(img_feats.dtype)
                aux["reg_l2"] = reg_l2.detach()
                aux["reg_tv"] = reg_tv.detach()
                aux["reg_eps"] = reg_eps.detach()

            img_feats = img_feats + delta.to(img_feats.dtype)

        img_sub_ids = tok(self.vl_chat.image_start_tag, add_special_tokens=False).input_ids

        def _find_subseq_start(row: torch.Tensor, sub: List[int]) -> int:
            r = row.tolist()
            L, S = len(r), len(sub)
            for i in range(0, L - S + 1):
                if r[i : i + S] == sub:
                    return i
            return -1

        embeds_list, masks_list = [], []
        for b in range(txt_embeds_prompt.size(0)):
            ids_b = batched.input_ids[b]
            e_b = txt_embeds_prompt[b]
            m_b = attn_mask_prompt[b]
            p = _find_subseq_start(ids_b, img_sub_ids)
            if p < 0:
                embeds_list.append(e_b)
                masks_list.append(m_b)
                continue
            Np = img_feats.size(1)
            e_new = torch.cat([e_b[:p], img_feats[b], e_b[p + len(img_sub_ids) :]], dim=0)
            m_new = torch.cat(
                [
                    m_b[:p],
                    torch.ones(Np, dtype=m_b.dtype, device=m_b.device),
                    m_b[p + len(img_sub_ids) :],
                ],
                dim=0,
            )
            embeds_list.append(e_new)
            masks_list.append(m_new)

        eos = tok.eos_token or ""
        max_answer_tokens = int(getattr(self, "max_answer_tokens", 0) or 0)
        ans_embeds_list, labels_list = [], []
        for a in answers_txt:
            a_ids = tok(a + eos, add_special_tokens=False).input_ids
            a_ids = torch.tensor(a_ids, dtype=torch.long, device=self.device)
            if max_answer_tokens > 0 and a_ids.numel() > max_answer_tokens:
                a_ids = a_ids[:max_answer_tokens]
            a_emb = lm.get_input_embeddings()(a_ids)
            ans_embeds_list.append(a_emb)
            labels_list.append(a_ids)

        seqs, masks, labels = [], [], []
        maxL = 0
        for e_p, m_p, e_a, y_a in zip(embeds_list, masks_list, ans_embeds_list, labels_list):
            e = torch.cat([e_p, e_a], dim=0)
            m = torch.cat(
                [m_p, torch.ones(e_a.size(0), dtype=m_p.dtype, device=m_p.device)], dim=0
            )
            y = torch.cat(
                [torch.full((e_p.size(0),), -100, dtype=torch.long, device=self.device), y_a],
                dim=0,
            )
            seqs.append(e)
            masks.append(m)
            labels.append(y)
            maxL = max(maxL, e.size(0))

        def _pad(t, L, padv=0):
            if t.dim() == 1:
                return F.pad(t, (0, L - t.size(0)), value=padv)
            else:
                return F.pad(t, (0, 0, 0, L - t.size(0)), value=0)

        inputs_embeds = torch.stack([_pad(s, maxL, 0) for s in seqs])
        attention_mask = torch.stack([_pad(m, maxL, 0) for m in masks])
        labels = torch.stack([_pad(y, maxL, -100) for y in labels])

        out = lm(
            inputs_embeds=inputs_embeds.to(lm.dtype),
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return out.loss, aux
