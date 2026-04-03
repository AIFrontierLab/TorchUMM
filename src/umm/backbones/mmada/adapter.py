from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def _img_format(path: str | Path) -> str:
    """Infer PIL save format from file extension, defaulting to PNG."""
    ext = Path(path).suffix.lower()
    return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext.lstrip("."), "PNG")


def _image_transform(image: Image.Image, resolution: int = 512) -> torch.Tensor:
    """Resize, center-crop, normalize to [-1, 1]."""
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    return image


# Llama3-style chat template used by MMaDA
_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n'"
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}"
)


class MMaDABackbone:
    name = "mmada"

    def __init__(
        self,
        model_path: Optional[str] = None,
        mmada_root: Optional[str] = None,
        vq_model_path: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.mmada_root = (
            Path(mmada_root).expanduser()
            if mmada_root
            else repo_root / "model" / "MMaDA"
        )
        self.model_path = (
            str(Path(model_path).expanduser())
            if model_path
            else str(repo_root / "model_cache" / "mmada" / "MMaDA-8B-Base")
        )
        self.vq_model_path = (
            str(Path(vq_model_path).expanduser())
            if vq_model_path
            else str(repo_root / "model_cache" / "mmada" / "magvitv2")
        )
        self.seed = seed
        self.resolution = 512

        self.default_generation_cfg: dict[str, Any] = {
            "guidance_scale": 1.5,
            "temperature": 1.0,
            "timesteps": 12,
            "num_vq_tokens": 1024,
            "codebook_size": 8192,
            "resolution": 512,
            "mask_schedule": "cosine",
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 512,
            "steps": 256,
            "block_length": 128,
            "temperature": 0.0,
            "remasking": "low_confidence",
        }

        # Populated by load()
        self.model: Any = None
        self.vq_model: Any = None
        self.tokenizer: Any = None
        self.uni_prompting: Any = None
        self.mask_token_id: int = 126336
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._get_mask_schedule: Any = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, cfg: dict[str, Any]) -> None:
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(Path(cfg_model_path).expanduser())
        cfg_mmada_root = cfg.get("mmada_root")
        if cfg_mmada_root:
            self.mmada_root = Path(cfg_mmada_root).expanduser()
        cfg_vq_path = cfg.get("vq_model_path")
        if cfg_vq_path:
            self.vq_model_path = str(Path(cfg_vq_path).expanduser())
        self.seed = int(cfg.get("seed", self.seed))
        self.resolution = int(cfg.get("resolution", self.resolution))

        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

        self._set_seed(self.seed)
        self._load_models()

    def _load_models(self) -> None:
        # Add MMaDA repo to sys.path for model imports
        mmada_root_str = str(self.mmada_root.resolve())
        if mmada_root_str not in sys.path:
            sys.path.insert(0, mmada_root_str)

        from models import MAGVITv2, MMadaModelLM, get_mask_schedule
        from training.prompting_utils import UniversalPrompting
        from transformers import AutoTokenizer

        self._get_mask_schedule = get_mask_schedule

        # Tokenizer
        print(f"[mmada] Loading tokenizer from {self.model_path} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        self.tokenizer.chat_template = _CHAT_TEMPLATE
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Universal prompting helper
        self.uni_prompting = UniversalPrompting(
            self.tokenizer,
            max_text_len=512,
            special_tokens=(
                "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>",
            ),
            ignore_id=-100,
            cond_dropout_prob=0.1,
            use_reserved_token=True,
        )

        # VQ model (MAGVITv2)
        print(f"[mmada] Loading VQ model from {self.vq_model_path} ...", flush=True)
        self.vq_model = MAGVITv2.from_pretrained(self.vq_model_path).to(self.device)
        self.vq_model.eval()
        self.vq_model.requires_grad_(False)

        # Main model
        print(f"[mmada] Loading MMaDA model from {self.model_path} ...", flush=True)
        self.model = MMadaModelLM.from_pretrained(
            self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()

        # Mask token ID
        if hasattr(self.model.config, "mask_token_id") and self.model.config.mask_token_id is not None:
            self.mask_token_id = self.model.config.mask_token_id
        else:
            self.mask_token_id = 126336
        print(f"[mmada] Model loaded. mask_token_id={self.mask_token_id}", flush=True)

    # ------------------------------------------------------------------
    # Generation (text-to-image)
    # ------------------------------------------------------------------

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected batch['prompt'] as a non-empty string.")
        return self.generation(prompt=prompt, output_path=batch.get("output_path"), generation_cfg=gen_cfg)

    def generation(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load({})

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        guidance_scale = float(config.get("guidance_scale", 1.5))
        temperature = float(config.get("temperature", 1.0))
        timesteps = int(config.get("timesteps", 12))
        num_vq_tokens = int(config.get("num_vq_tokens", 1024))
        codebook_size = int(config.get("codebook_size", 8192))
        mask_schedule_name = str(config.get("mask_schedule", "cosine"))

        mask_schedule = self._get_mask_schedule(mask_schedule_name)

        # Build input tokens
        prompts = [prompt]
        image_tokens = torch.ones(
            (len(prompts), num_vq_tokens), dtype=torch.long, device=self.device
        ) * self.mask_token_id

        input_ids, attention_mask = self.uni_prompting((prompts, image_tokens), "t2i_gen")

        # Classifier-free guidance
        if guidance_scale > 0:
            uncond_input_ids, uncond_attention_mask = self.uni_prompting(
                ([""] * len(prompts), image_tokens), "t2i_gen"
            )
        else:
            uncond_input_ids = None
            uncond_attention_mask = None

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen_token_ids = self.model.t2i_generate(
                input_ids=input_ids,
                uncond_input_ids=uncond_input_ids,
                attention_mask=attention_mask,
                uncond_attention_mask=uncond_attention_mask,
                guidance_scale=guidance_scale,
                temperature=temperature,
                timesteps=timesteps,
                noise_schedule=mask_schedule,
                seq_len=num_vq_tokens,
                mask_token_id=self.mask_token_id,
                codebook_size=codebook_size,
                uni_prompting=self.uni_prompting,
                config=None,
            )

        # Decode VQ tokens to image
        gen_token_ids = torch.clamp(gen_token_ids, max=codebook_size - 1, min=0)
        images = self.vq_model.decode_code(gen_token_ids)
        images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
        images *= 255.0
        images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        pil_image = Image.fromarray(images[0])

        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            pil_image.save(output_path, format=_img_format(output_path))

        return {"image": pil_image}

    # ------------------------------------------------------------------
    # Understanding (image + text -> text)
    # ------------------------------------------------------------------

    def understand(self, batch: dict[str, Any], understanding_cfg: dict[str, Any]) -> Any:
        return self.understanding(
            prompt=batch.get("prompt"),
            images=batch.get("images", []),
            videos=batch.get("videos", []),
            understanding_cfg=understanding_cfg,
        )

    def understanding(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if self.model is None:
            self.load({})

        if videos:
            raise NotImplementedError("MMaDA does not support video understanding.")

        image_list = images or []
        if prompt is None and not image_list:
            raise ValueError("Understanding requires at least a prompt or images.")

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        max_new_tokens = int(config.get("max_new_tokens", 512))
        steps = int(config.get("steps", max(1, max_new_tokens // 2)))
        block_length = int(config.get("block_length", max(1, max_new_tokens // 4)))
        temperature = float(config.get("temperature", 0.0))
        remasking = str(config.get("remasking", "low_confidence"))

        # Build chat messages
        question = prompt if isinstance(prompt, str) else "Please describe this image in detail."
        messages = [{"role": "user", "content": question}]

        if image_list:
            # Image understanding
            image_path = Path(image_list[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            image_pil = Image.open(image_path).convert("RGB")
            image_tensor = _image_transform(image_pil, resolution=self.resolution).to(self.device)
            image_tensor = image_tensor.unsqueeze(0)

            # VQ encode — offset by tokenizer vocab size
            image_tokens = self.vq_model.get_code(image_tensor) + len(self.tokenizer)

            # Chat template tokens
            text_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)

            # Build input: [mmu] [soi] [image_tokens] [eoi] [text_tokens]
            batch_size = image_tokens.shape[0]
            input_ids = torch.cat([
                (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict["<|mmu|>"]).to(self.device),
                (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict["<|soi|>"]).to(self.device),
                image_tokens,
                (torch.ones(batch_size, 1) * self.uni_prompting.sptids_dict["<|eoi|>"]).to(self.device),
                text_token_ids,
            ], dim=1).long()
        else:
            # Text-only understanding (no image)
            text_token_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.device)
            input_ids = text_token_ids

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = self.model.mmu_generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                steps=steps,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                mask_id=self.mask_token_id,
            )

        generated_ids = output_ids[:, input_ids.shape[1]:]
        response_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {"text": response_text}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
