from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


def _img_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext.lstrip("."), "PNG")


class OvisU1Backbone:
    """Adapter for AIDC-AI Ovis-U1-3B unified multimodal model.

    Supports image understanding, text-to-image generation, and image editing.
    """

    name = "ovis_u1"

    def __init__(
        self,
        model_path: Optional[str] = None,
        seed: int = 42,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> None:
        self.model_path = model_path
        self.seed = seed
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)

        self.default_generation_cfg: dict[str, Any] = {
            "height": 1024,
            "width": 1024,
            "num_steps": 50,
            "txt_cfg": 5.0,
            "img_cfg": 0,
            "seed": 42,
        }
        self.default_editing_cfg: dict[str, Any] = {
            "height": 1024,
            "width": 1024,
            "num_steps": 50,
            "txt_cfg": 6.0,
            "img_cfg": 1.5,
            "seed": 42,
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 4096,
            "do_sample": False,
        }

        self.model: Any = None
        self.text_tokenizer: Any = None
        self.visual_tokenizer: Any = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, cfg: dict[str, Any]) -> None:
        if cfg.get("model_path"):
            self.model_path = cfg["model_path"]
        if cfg.get("seed") is not None:
            self.seed = int(cfg["seed"])
        if cfg.get("device"):
            self.device = cfg["device"]
        if cfg.get("torch_dtype"):
            self.torch_dtype = getattr(torch, cfg["torch_dtype"], torch.bfloat16)

        gen_cfg = cfg.get("generation_cfg")
        if isinstance(gen_cfg, dict):
            self.default_generation_cfg.update(gen_cfg)
        edit_cfg = cfg.get("editing_cfg")
        if isinstance(edit_cfg, dict):
            self.default_editing_cfg.update(edit_cfg)
        und_cfg = cfg.get("understanding_cfg")
        if isinstance(und_cfg, dict):
            self.default_understanding_cfg.update(und_cfg)

        self._set_seed(self.seed)
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM

        print(f"[ovis_u1] loading model from {self.model_path} ...", flush=True)
        model, _ = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            output_loading_info=True,
            trust_remote_code=True,
        )
        self.model = model.eval().to(self.device).to(self.torch_dtype)
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        print("[ovis_u1] model loaded", flush=True)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load({})

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def _build_understanding_inputs(
        self,
        prompt: str,
        pil_images: list[Image.Image] | None,
        multimodal_type: str = "single_image",
    ) -> tuple:
        """Build inputs for understanding (text generation) tasks."""
        images_arg = pil_images if multimodal_type == "multiple_image" else (pil_images[:1] if pil_images else [None])

        _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            prompt,
            images_arg,
            generation_preface="",
            return_labels=False,
            propagate_exception=False,
            multimodal_type=multimodal_type,
            fix_sample_overall_length_navit=False,
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = torch.cat(
                [pixel_values.to(device=self.visual_tokenizer.device, dtype=torch.bfloat16)], dim=0
            )
        if grid_thws is not None:
            grid_thws = torch.cat(
                [grid_thws.to(device=self.visual_tokenizer.device)], dim=0
            )
        return input_ids, pixel_values, attention_mask, grid_thws

    def _build_generation_inputs(
        self,
        prompt: str,
        pil_image: Image.Image,
        target_width: int,
        target_height: int,
    ) -> tuple:
        """Build inputs for generation/editing (image output) tasks."""
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = (
            self.model.visual_generator.process_image_aspectratio(pil_image, target_size)
        )
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=self.model.device)

        width, height = pil_image.width, pil_image.height
        resized_h, resized_w = self.visual_tokenizer.smart_resize(
            height, width, max_pixels=self.visual_tokenizer.image_processor.min_pixels
        )
        pil_image = pil_image.resize((resized_w, resized_h))

        _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            prompt,
            [pil_image],
            generation_preface=None,
            return_labels=False,
            propagate_exception=False,
            multimodal_type="single_image",
            fix_sample_overall_length_navit=False,
        )
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
        if pixel_values is not None:
            pixel_values = torch.cat(
                [pixel_values.to(device=self.visual_tokenizer.device, dtype=torch.bfloat16)], dim=0
            )
        if grid_thws is not None:
            grid_thws = torch.cat(
                [grid_thws.to(device=self.visual_tokenizer.device)], dim=0
            )
        return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------

    def understanding(
        self,
        prompt: Optional[str] = None,
        images: Optional[list[str]] = None,
        videos: Optional[list[str]] = None,
        understanding_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        image_list = images or []
        if videos:
            raise NotImplementedError("OvisU1Backbone does not support video understanding.")

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        # Open images
        pil_images: list[Image.Image] = []
        for img in image_list:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                pil_images.append(Image.open(img).convert("RGB"))

        # Determine multimodal type and format prompt
        if not pil_images:
            multimodal_type = "text"
        elif len(pil_images) == 1:
            multimodal_type = "single_image"
            prompt = "<image>\n" + (prompt or "")
        else:
            multimodal_type = "multiple_image"
            prefix = "\n".join(f"Image {i + 1}: <image>" for i in range(len(pil_images)))
            prompt = prefix + "\n" + (prompt or "")

        # Handle text-only: need mock visual input
        if multimodal_type == "text":
            from torchvision import transforms as T

            mock_pv, _ = self.visual_tokenizer.mock_input()
            mock_img = T.ToPILImage()(mock_pv[0])
            pil_images = [mock_img]
            # For text-only, preprocess with multimodal_type='text'
            _, input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                prompt,
                None,
                generation_preface="",
                return_labels=False,
                propagate_exception=False,
                multimodal_type="text",
                fix_sample_overall_length_navit=False,
            )
            attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
            input_ids = input_ids.unsqueeze(0).to(device=self.model.device)
            attention_mask = attention_mask.unsqueeze(0).to(device=self.model.device)
            # Process mock image for pixel_values
            pixel_values, grid_thws, _ = self.visual_tokenizer.preprocess_image(mock_img)
            if pixel_values is not None:
                pixel_values = torch.cat(
                    [pixel_values.to(device=self.visual_tokenizer.device, dtype=torch.bfloat16)], dim=0
                )
            if grid_thws is not None:
                grid_thws = torch.cat(
                    [grid_thws.to(device=self.visual_tokenizer.device)], dim=0
                )
        else:
            input_ids, pixel_values, attention_mask, grid_thws = self._build_understanding_inputs(
                prompt, pil_images, multimodal_type
            )

        gen_kwargs = dict(
            max_new_tokens=config.get("max_new_tokens", 4096),
            do_sample=config.get("do_sample", False),
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.text_tokenizer.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True,
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                grid_thws=grid_thws,
                **gen_kwargs,
            )[0]
            text = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return {"text": text}

    # ------------------------------------------------------------------
    # Generation (text-to-image)
    # ------------------------------------------------------------------

    def generation(
        self,
        prompt: str,
        output_path: Optional[str] = None,
        generation_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        height = int(config.pop("height", 1024))
        width = int(config.pop("width", 1024))
        num_steps = int(config.pop("num_steps", 50))
        txt_cfg = float(config.pop("txt_cfg", 5.0))
        img_cfg = float(config.pop("img_cfg", 0))
        seed = int(config.pop("seed", self.seed))

        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.text_tokenizer.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True,
            height=height,
            width=width,
            num_steps=num_steps,
            seed=seed,
            img_cfg=img_cfg,
            txt_cfg=txt_cfg,
        )

        blank_image = Image.new("RGB", (width, height), (255, 255, 255))
        uncond_prompt = "<image>\nGenerate an image."

        # Stage 1: unconditional (no text, no image)
        ids, pv, am, gt, _ = self._build_generation_inputs(uncond_prompt, blank_image, width, height)
        with torch.inference_mode():
            no_both_cond = self.model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs
            )

        # Stage 2: no text condition for T2I
        no_txt_cond = None

        # Stage 3: full condition
        full_prompt = (
            "<image>\nDescribe the image by detailing the color, shape, size, "
            "texture, quantity, text, and spatial relationships of the objects:" + prompt
        )
        ids, pv, am, gt, vae_pv = self._build_generation_inputs(full_prompt, blank_image, width, height)
        with torch.inference_mode():
            cond = self.model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs
            )
            cond["vae_pixel_values"] = vae_pv
            images = self.model.generate_img(
                cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs
            )

        pil_image = images[0]
        saved_paths: list[str] = []
        if output_path is not None:
            pil_image.save(output_path, format=_img_format(output_path))
            saved_paths.append(output_path)

        return {"image": pil_image, "images": images, "saved_paths": saved_paths}

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def editing(
        self,
        prompt: str,
        images: list[str],
        output_path: Optional[str] = None,
        editing_cfg: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        if not images:
            raise ValueError("Editing requires at least one input image.")

        # Open source image
        if isinstance(images[0], Image.Image):
            input_img = images[0].convert("RGB")
        else:
            input_img = Image.open(images[0]).convert("RGB")

        config = dict(self.default_editing_cfg)
        if editing_cfg:
            config.update(editing_cfg)

        num_steps = int(config.pop("num_steps", 50))
        txt_cfg = float(config.pop("txt_cfg", 6.0))
        img_cfg = float(config.pop("img_cfg", 1.5))
        seed = int(config.pop("seed", self.seed))

        # Resize source image dimensions to factor of 32
        orig_w, orig_h = input_img.size
        height, width = self.visual_tokenizer.smart_resize(orig_h, orig_w, factor=32)

        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.text_tokenizer.eos_token_id,
            pad_token_id=self.text_tokenizer.pad_token_id,
            use_cache=True,
            height=height,
            width=width,
            num_steps=num_steps,
            seed=seed,
            img_cfg=img_cfg,
            txt_cfg=txt_cfg,
        )

        uncond_prompt = "<image>\nGenerate an image."
        blank_image = Image.new("RGB", (width, height), (255, 255, 255))

        # Stage 1: unconditional
        ids, pv, am, gt, _ = self._build_generation_inputs(uncond_prompt, blank_image, width, height)
        with torch.inference_mode():
            no_both_cond = self.model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs
            )

        # Stage 2: image-only condition (source image, uncond prompt)
        resized_input = input_img.resize((width, height))
        ids, pv, am, gt, _ = self._build_generation_inputs(uncond_prompt, resized_input, width, height)
        with torch.inference_mode():
            no_txt_cond = self.model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs
            )

        # Stage 3: full condition (source image + edit prompt)
        edit_prompt = "<image>\n" + prompt.strip()
        ids, pv, am, gt, vae_pv = self._build_generation_inputs(edit_prompt, resized_input, width, height)
        with torch.inference_mode():
            cond = self.model.generate_condition(
                ids, pixel_values=pv, attention_mask=am, grid_thws=gt, **gen_kwargs
            )
            cond["vae_pixel_values"] = vae_pv
            images_out = self.model.generate_img(
                cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs
            )

        pil_image = images_out[0]
        saved_paths: list[str] = []
        if output_path is not None:
            pil_image.save(output_path, format=_img_format(output_path))
            saved_paths.append(output_path)

        return {"image": pil_image, "images": images_out, "saved_paths": saved_paths}

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
