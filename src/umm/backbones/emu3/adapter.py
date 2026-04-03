from __future__ import annotations

import importlib
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image


def _timeout_handler(signum, frame):
    raise TimeoutError("Image generation timed out")


class Emu3Backbone:
    name = "emu3"

    def __init__(
        self,
        model_path: Optional[str] = None,
        vq_hub: Optional[str] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "Emu3"
        self.emu_root = packaged if packaged.exists() else repo_root / "model" / "emu3" / "Emu3"
        self.model_path = model_path or str(repo_root / "model_cache" / "emu3" / "models" / "emu3_gen")
        self.vq_hub = vq_hub or str(repo_root / "model_cache" / "emu3" / "models" / "emu3_vision_tokenizer")
        self.device = device or "cuda:0"
        self.device_map = device_map or "cuda:0"
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation

        # Lazily populated by load()
        self.model: Any = None
        self.tokenizer: Any = None
        self.processor: Any = None

    def load(self, cfg: dict[str, Any]) -> None:
        if isinstance(cfg.get("emu_root"), str):
            self.emu_root = Path(cfg["emu_root"]).expanduser()
        if isinstance(cfg.get("model_path"), str):
            self.model_path = cfg["model_path"]
        if isinstance(cfg.get("vq_hub"), str):
            self.vq_hub = cfg["vq_hub"]
        if isinstance(cfg.get("device"), str):
            self.device = cfg["device"]
        if isinstance(cfg.get("device_map"), str):
            self.device_map = cfg["device_map"]
        if isinstance(cfg.get("torch_dtype"), str):
            self.torch_dtype = cfg["torch_dtype"]
        if isinstance(cfg.get("attn_implementation"), str):
            self.attn_implementation = cfg["attn_implementation"]

        # Add Emu3 repo to sys.path so `emu3.*` is importable
        emu_root_str = str(self.emu_root.resolve())
        if emu_root_str not in sys.path:
            sys.path.insert(0, emu_root_str)

        torch_dtype = self._resolve_torch_dtype(self.torch_dtype)

        # Import Emu3-specific modules
        processing_mod = importlib.import_module("emu3.mllm.processing_emu3")
        Emu3Processor = processing_mod.Emu3Processor

        from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM

        print(f"[emu3] Loading model from {self.model_path} ...", flush=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device_map,
                torch_dtype=torch_dtype,
                attn_implementation=self.attn_implementation,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[emu3] Warning: failed to load with device_map/attn_impl: {e}", flush=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True,
            )
            self.model = self.model.to(self.device)

        self.model.eval()

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, padding_side="left",
            )
        except (OSError, TypeError) as e:
            print(f"[emu3] Warning: failed to load tokenizer from local path: {e}", flush=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                "BAAI/Emu3-Gen", trust_remote_code=True, padding_side="left",
            )

        image_processor = AutoImageProcessor.from_pretrained(self.vq_hub, trust_remote_code=True)
        image_tokenizer = AutoModel.from_pretrained(
            self.vq_hub, device_map=self.device_map, trust_remote_code=True,
        ).eval()

        self.processor = Emu3Processor(image_processor, image_tokenizer, self.tokenizer)
        print("[emu3] Model loaded.", flush=True)

    def _ensure_loaded(self) -> None:
        if self.model is None:
            self.load({})

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        self._ensure_loaded()

        prompt = batch.get("prompt") or gen_cfg.get("prompt")
        if prompt is None:
            raise ValueError("Generation requires a prompt.")
        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        output_path = batch.get("output_path")

        saved_paths = []
        with torch.inference_mode():
            for i, p in enumerate(prompts):
                img = self._generate_one(p, gen_cfg)
                if img is not None and output_path and i == 0:
                    dst = Path(output_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(dst), format=self._fmt(dst))
                    saved_paths.append(str(dst))

        return {
            "images": saved_paths,
            "saved_paths": saved_paths,
            "output_path": output_path or "",
        }

    def generate_batch(
        self,
        prompt_items: list[dict[str, Any]],
        gen_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not prompt_items:
            return []
        self._ensure_loaded()

        timeout_sec = int(gen_cfg.get("timeout_per_image", 900))

        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for i, item in enumerate(prompt_items):
                prompt = item["prompt"]
                output_path = item.get("output_path", "")
                print(f"[emu3] [{i + 1}/{len(prompt_items)}] {prompt[:80]} ...", flush=True)

                try:
                    # Set per-image timeout via SIGALRM (Unix only)
                    prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(timeout_sec)
                    try:
                        img = self._generate_one(prompt, gen_cfg)
                    finally:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, prev_handler)
                except TimeoutError:
                    print(f"[emu3]   Timeout after {timeout_sec}s, skipping", flush=True)
                    results.append({"images": [], "ok": False})
                    continue
                except Exception as e:
                    print(f"[emu3]   Error: {e}", flush=True)
                    results.append({"images": [], "ok": False})
                    continue

                ok = False
                if img is not None and output_path:
                    dst = Path(output_path)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    img.save(str(dst), format=self._fmt(dst))
                    ok = dst.is_file()
                    print(f"[emu3]   -> saved to {dst}", flush=True)

                results.append({"images": [output_path] if ok else [], "output_path": output_path, "ok": ok})

        return results

    def _generate_one(self, prompt: str, gen_cfg: dict[str, Any]) -> Optional[Image.Image]:
        """Run a single image generation and return the PIL Image (or None)."""
        from transformers.generation.configuration_utils import GenerationConfig
        from transformers.generation import (
            LogitsProcessorList,
            PrefixConstrainedLogitsProcessor,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
        )

        positive_prompt = gen_cfg.get("positive_prompt", " masterpiece, film grained, best quality.")
        negative_prompt = gen_cfg.get(
            "negative_prompt",
            "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
            "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
            "signature, watermark, username, blurry.",
        )
        cfg_scale = float(gen_cfg.get("classifier_free_guidance", 3.0))

        generation_config = GenerationConfig(
            use_cache=True,
            eos_token_id=self.model.config.eos_token_id,
            pad_token_id=self.model.config.pad_token_id,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 40960)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
            top_k=int(gen_cfg.get("top_k", 2048)),
        )

        full_prompt = prompt + positive_prompt
        kwargs = dict(
            mode="G",
            ratio=gen_cfg.get("ratio", "1:1"),
            image_area=int(gen_cfg.get("image_area", self.model.config.image_area)),
            return_tensors="pt",
            padding="longest",
        )

        pos_inputs = self.processor(text=full_prompt, **kwargs)
        neg_inputs = self.processor(text=negative_prompt, **kwargs)

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = self.processor.build_prefix_constrained_fn(h, w)

        logits_processor = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(
                cfg_scale,
                self.model,
                unconditional_ids=neg_inputs.input_ids.to(self.device),
            ),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
        ])

        outputs = self.model.generate(
            pos_inputs.input_ids.to(self.device),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(self.device),
        )

        mm_list = self.processor.decode(outputs[0])
        for item in mm_list:
            if isinstance(item, Image.Image):
                return item
        return None

    # ------------------------------------------------------------------
    # Understanding
    # ------------------------------------------------------------------

    def understand(self, batch: dict[str, Any], understanding_cfg: dict[str, Any]) -> Any:
        self._ensure_loaded()

        images = batch.get("images", [])
        prompt = batch.get("prompt") or understanding_cfg.get("prompt")

        results = []
        with torch.inference_mode():
            for img_path in images:
                text = self._understand_one(prompt, img_path, understanding_cfg)
                results.append({"image": str(img_path), "text": text})

        return {"results": results, "text": results[0]["text"] if results else ""}

    def understand_batch(
        self,
        items: list[dict[str, Any]],
        understanding_cfg: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not items:
            return []
        self._ensure_loaded()

        results: list[dict[str, Any]] = []
        with torch.inference_mode():
            for i, item in enumerate(items):
                prompt = item.get("prompt", "")
                images = item.get("images", [])
                img_path = images[0] if images else None
                print(f"[emu3] understand [{i + 1}/{len(items)}] {prompt[:80]} ...", flush=True)

                try:
                    text = self._understand_one(prompt, img_path, understanding_cfg)
                except Exception as e:
                    print(f"[emu3]   Error: {e}", flush=True)
                    text = ""

                results.append({"text": text})
                print(f"[emu3]   -> {len(text)} chars", flush=True)

        return results

    def _understand_one(
        self, prompt: Optional[str], img_path: Optional[str], cfg: dict[str, Any],
    ) -> str:
        """Run understanding on a single image+prompt and return the text response."""
        from transformers.generation.configuration_utils import GenerationConfig

        img = None
        if img_path:
            img = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            text=prompt, image=img, mode="U",
            return_tensors="pt", padding="longest",
        )

        gen_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=int(cfg.get("max_new_tokens", 1024)),
        )

        outputs = self.model.generate(
            inputs.input_ids.to(self.device),
            gen_config,
            attention_mask=inputs.attention_mask.to(self.device),
        )
        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        text_out = self.processor.batch_decode(outputs, skip_special_tokens=True)
        return text_out[0] if text_out else ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt(path: Path) -> str:
        """Infer PIL save format from extension, defaulting to PNG."""
        ext = path.suffix.lower().lstrip(".")
        return {"jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(ext, "PNG")

    @staticmethod
    def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
        lowered = dtype_name.lower()
        if lowered in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if lowered in {"fp16", "float16", "half"}:
            return torch.float16
        if lowered in {"fp32", "float32"}:
            return torch.float32
        return getattr(torch, dtype_name, torch.bfloat16)
