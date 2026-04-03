from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image


class JanusProBackbone:
    name = "janus_pro"

    def __init__(
        self,
        model_path: Optional[str] = None,
        janus_root: Optional[str] = None,
        seed: int = 42,
        torch_dtype: str = "bfloat16",
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        self.janus_root = (
            Path(janus_root).expanduser()
            if janus_root
            else repo_root / "src" / "umm" / "backbones" / "janus_pro" / "Janus"
        )
        self.model_path = (
            str(Path(model_path).expanduser())
            if model_path
            else str(repo_root / "model_cache" / "janus" / "models" / "Janus_pro_7B")
        )
        self.seed = seed
        self.torch_dtype = torch_dtype

        self.default_generation_cfg: dict[str, Any] = {
            "temperature": 1.0,
            "parallel_size": 4,
            "cfg_weight": 5.0,
            "image_token_num_per_image": 576,
            "img_size": 384,
            "patch_size": 16,
        }
        self.default_understanding_cfg: dict[str, Any] = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True,
        }

        self.vl_chat_processor: Any = None
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self, cfg: dict[str, Any]) -> None:
        cfg_model_path = cfg.get("model_path")
        if cfg_model_path:
            self.model_path = str(Path(cfg_model_path).expanduser())
        cfg_janus_root = cfg.get("janus_root")
        if cfg_janus_root:
            self.janus_root = Path(cfg_janus_root).expanduser()
        self.seed = int(cfg.get("seed", self.seed))
        self.torch_dtype = str(cfg.get("torch_dtype", self.torch_dtype))
        generation_cfg = cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        understanding_cfg = cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

        self._set_seed(self.seed)
        self.vl_chat_processor, self.tokenizer, self.model = self._build_model()

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
        if self.model is None or self.vl_chat_processor is None:
            self.load({})
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Expected `prompt` as a non-empty string.")

        config = dict(self.default_generation_cfg)
        if generation_cfg:
            config.update(generation_cfg)

        full_prompt = self._build_generation_prompt(prompt)
        images = self._generate_images(full_prompt, config)
        saved_paths = self._save_images(images, output_path)
        return {
            "image": images[0] if images else None,
            "images": images,
            "saved_paths": saved_paths,
        }

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
        if self.model is None or self.vl_chat_processor is None:
            self.load({})

        image_paths = images or []
        video_paths = videos or []
        if video_paths:
            raise NotImplementedError("JanusProBackbone.understanding currently does not support videos.")

        if not prompt and not image_paths:
            raise ValueError("Understanding requires at least one input: prompt or images.")

        pil_images: list[Image.Image] = []
        if image_paths:
            image_path = Path(image_paths[0]).expanduser()
            if not image_path.exists():
                raise FileNotFoundError(f"Understanding image not found: {image_path}")
            pil_images.append(Image.open(image_path).convert("RGB"))

        user_prompt = prompt if isinstance(prompt, str) and prompt else "Describe the image."
        content = f"<image_placeholder>\n{user_prompt}" if pil_images else user_prompt
        conversation = [
            {
                "role": "<|User|>",
                "content": content,
                "images": image_paths[:1] if image_paths else [],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        config = dict(self.default_understanding_cfg)
        if understanding_cfg:
            config.update(understanding_cfg)

        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=int(config.get("max_new_tokens", 512)),
            do_sample=bool(config.get("do_sample", False)),
            use_cache=bool(config.get("use_cache", True)),
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return {"text": answer, "sft_format": prepare_inputs["sft_format"][0]}


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

    def _build_model(self) -> tuple[Any, Any, Any]:
        if not self.janus_root.exists():
            raise FileNotFoundError(f"Janus root not found: {self.janus_root}")

        janus_root_str = str(self.janus_root.resolve())
        if janus_root_str not in sys.path:
            sys.path.insert(0, janus_root_str)

        if not torch.cuda.is_available():
            raise RuntimeError("JanusProBackbone requires at least one CUDA device.")

        transformers_module = importlib.import_module("transformers")
        janus_models = importlib.import_module("janus.models")

        model_path = str(Path(self.model_path).expanduser())
        vl_chat_processor = janus_models.VLChatProcessor.from_pretrained(model_path)
        tokenizer = vl_chat_processor.tokenizer

        # Force CPU as default device during model loading to prevent
        # transformers 5.x from using meta tensors internally, which
        # breaks torch.linspace(...).item() in siglip_vit.py.
        default_device_before = None
        if hasattr(torch, "set_default_device"):
            if hasattr(torch, "get_default_device"):
                default_device_before = torch.get_default_device()
            torch.set_default_device("cpu")

        try:
            model = transformers_module.AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                device_map=None,
            )
        finally:
            if hasattr(torch, "set_default_device"):
                torch.set_default_device(default_device_before)

        model = model.to(self._resolve_torch_dtype(self.torch_dtype)).cuda().eval()
        return vl_chat_processor, tokenizer, model

    def _build_generation_prompt(self, prompt: str) -> str:
        conversation = [
            {"role": "<|User|>", "content": prompt},
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.vl_chat_processor.sft_format,
            system_prompt="",
        )
        return sft_format + self.vl_chat_processor.image_start_tag

    def _generate_images(self, prompt: str, cfg: dict[str, Any]) -> list[Image.Image]:
        temperature = float(cfg.get("temperature", 1.0))
        parallel_size = int(cfg.get("parallel_size", 4))
        cfg_weight = float(cfg.get("cfg_weight", 5.0))
        image_token_num_per_image = int(cfg.get("image_token_num_per_image", 576))
        img_size = int(cfg.get("img_size", 384))
        patch_size = int(cfg.get("patch_size", 16))

        input_ids = self.vl_chat_processor.tokenizer.encode(prompt)
        input_ids = torch.LongTensor(input_ids).cuda()

        tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).cuda()
        for i in range(parallel_size * 2):
            tokens[i, :] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = self.vl_chat_processor.pad_id

        inputs_embeds = self.model.language_model.get_input_embeddings()(tokens)
        generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

        with torch.inference_mode():
            for i in range(image_token_num_per_image):
                outputs = self.model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=outputs.past_key_values if i != 0 else None,
                )
                hidden_states = outputs.last_hidden_state
                logits = self.model.gen_head(hidden_states[:, -1, :])
                logit_cond = logits[0::2, :]
                logit_uncond = logits[1::2, :]
                logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
                probs = torch.softmax(logits / temperature, dim=-1)

                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens[:, i] = next_token.squeeze(dim=-1)
                next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
                img_embeds = self.model.prepare_gen_img_embeds(next_token)
                inputs_embeds = img_embeds.unsqueeze(dim=1)

            decoded = self.model.gen_vision_model.decode_code(
                generated_tokens.to(dtype=torch.int),
                shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
            )
            decoded = decoded.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            decoded = np.clip((decoded + 1) / 2 * 255, 0, 255).astype(np.uint8)

        return [Image.fromarray(decoded[i]) for i in range(parallel_size)]

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")

    def _save_images(self, images: list[Image.Image], output_path: Optional[str]) -> list[str]:
        if not output_path:
            return []

        out_path = Path(output_path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix if out_path.suffix else ".png"
        stem = out_path.stem if out_path.stem else "output"

        saved_paths: list[str] = []
        if len(images) == 1:
            target = out_path if out_path.suffix else out_path.with_suffix(suffix)
            images[0].save(target, format=self._fmt(target))
            saved_paths.append(str(target))
            return saved_paths

        for idx, image in enumerate(images):
            target = out_path.with_name(f"{stem}_{idx}{suffix}")
            image.save(target, format=self._fmt(target))
            saved_paths.append(str(target))
        return saved_paths

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        lowered = dtype_name.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
