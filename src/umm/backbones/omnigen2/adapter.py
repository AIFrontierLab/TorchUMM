"""OmniGen2 backbone adapter for text-to-image, understanding, and editing."""

import sys
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image


class OmniGen2Backbone:
    """OmniGen2 backbone adapter supporting text-to-image, understanding, and editing.

    Uses in-process pipeline calls (no subprocess isolation). Custom omnigen2
    modules are registered in sys.modules once so diffusers can resolve the
    relative names in model_index.json.

    Both OmniGen2Pipeline and OmniGen2ChatPipeline share the same underlying
    components (mllm, transformer, vae, scheduler, processor) to avoid
    duplicating large model weights in GPU memory.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        **kwargs
    ):
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "OmniGen2"
        self.omnigen2_root = packaged if packaged.exists() else repo_root / "model" / "OmniGen2"
        self.model_path = model_path or str(repo_root / "model_cache" / "omnigen2" / "models" / "omnigen2")
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_map = kwargs.get("device_map", device)
        self.enable_cpu_offload = kwargs.get("enable_cpu_offload", False)
        self.enable_sequential_cpu_offload = kwargs.get("enable_sequential_cpu_offload", False)

        # Output directories
        self.out_dir_t2i = Path(kwargs.get("output_dir_t2i", "output/omnigen2_images"))
        self.out_dir_understanding = Path(kwargs.get("output_dir_understanding", "output/omnigen2_understanding"))
        self.out_dir_editing = Path(kwargs.get("output_dir_editing", "output/omnigen2_editing"))

        self.out_dir_t2i.mkdir(parents=True, exist_ok=True)
        self.out_dir_understanding.mkdir(parents=True, exist_ok=True)
        self.out_dir_editing.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded pipelines (share components to avoid OOM)
        self.pipeline = None       # OmniGen2Pipeline (t2i + edit)
        self.chat_pipeline = None  # OmniGen2ChatPipeline (understand)
        self._modules_registered = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_output_dir(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path

    def _resolve_dtype(self) -> torch.dtype:
        return {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(self.torch_dtype, torch.bfloat16)

    @staticmethod
    def _to_items(batch: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if batch is None:
            return []
        if isinstance(batch, dict):
            return [batch]
        if isinstance(batch, list):
            return [item for item in batch if isinstance(item, dict)]
        return []

    def _register_modules(self) -> None:
        """Add omnigen2 repo to sys.path and register custom modules so
        diffusers can resolve relative names from model_index.json."""
        if self._modules_registered:
            return
        omnigen2_root_str = str(self.omnigen2_root.resolve())
        if omnigen2_root_str not in sys.path:
            sys.path.insert(0, omnigen2_root_str)
        import omnigen2.models.transformers.transformer_omnigen2 as _tf_mod
        import omnigen2.schedulers.scheduling_flow_match_euler_discrete as _sched_mod
        sys.modules["transformer_omnigen2"] = _tf_mod
        sys.modules["scheduling_flow_match_euler_discrete"] = _sched_mod
        self._modules_registered = True

    def _load_primary_pipeline(self) -> None:
        """Load the main OmniGen2Pipeline from disk (only called once)."""
        self._register_modules()
        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline

        torch_dtype = self._resolve_dtype()
        dm = self.device_map
        if isinstance(dm, str) and dm.startswith("cuda:"):
            dm = "cuda"

        print(f"Loading OmniGen2 model from {self.model_path}...")
        if dm in ("cuda", "balanced"):
            self.pipeline = OmniGen2Pipeline.from_pretrained(
                self.model_path, torch_dtype=torch_dtype, device_map=dm,
            )
        else:
            self.pipeline = OmniGen2Pipeline.from_pretrained(
                self.model_path, torch_dtype=torch_dtype,
            )
            self.pipeline.to(self.device)

        if not hasattr(self.pipeline.transformer, "enable_teacache"):
            self.pipeline.transformer.enable_teacache = False
        if self.enable_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        if self.enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()

    def _ensure_pipeline(self) -> None:
        """Lazy-load OmniGen2Pipeline (used for t2i and editing)."""
        if self.pipeline is not None:
            return

        if self.chat_pipeline is not None:
            # Build from shared components to avoid loading weights twice
            self._register_modules()
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
            print("Building OmniGen2Pipeline from shared chat pipeline components...")
            self.pipeline = OmniGen2Pipeline(
                transformer=self.chat_pipeline.transformer,
                vae=self.chat_pipeline.vae,
                scheduler=self.chat_pipeline.scheduler,
                mllm=self.chat_pipeline.mllm,
                processor=self.chat_pipeline.processor,
            )
            if not hasattr(self.pipeline.transformer, "enable_teacache"):
                self.pipeline.transformer.enable_teacache = False
        else:
            self._load_primary_pipeline()

    def _ensure_chat_pipeline(self) -> None:
        """Lazy-load OmniGen2ChatPipeline (used for understanding)."""
        if self.chat_pipeline is not None:
            return

        if self.pipeline is not None:
            # Build from shared components to avoid loading weights twice
            self._register_modules()
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline
            print("Building OmniGen2ChatPipeline from shared pipeline components...")
            self.chat_pipeline = OmniGen2ChatPipeline(
                transformer=self.pipeline.transformer,
                vae=self.pipeline.vae,
                scheduler=self.pipeline.scheduler,
                mllm=self.pipeline.mllm,
                processor=self.pipeline.processor,
            )
            if hasattr(self.chat_pipeline, "transformer") and not hasattr(self.chat_pipeline.transformer, "enable_teacache"):
                self.chat_pipeline.transformer.enable_teacache = False
        else:
            # Load chat pipeline from disk
            self._register_modules()
            from omnigen2.pipelines.omnigen2.pipeline_omnigen2_chat import OmniGen2ChatPipeline

            torch_dtype = self._resolve_dtype()
            dm = self.device_map
            if isinstance(dm, str) and dm.startswith("cuda:"):
                dm = "cuda"

            print(f"Loading OmniGen2 chat model from {self.model_path}...")
            if dm in ("cuda", "balanced"):
                self.chat_pipeline = OmniGen2ChatPipeline.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype, device_map=dm,
                )
            else:
                self.chat_pipeline = OmniGen2ChatPipeline.from_pretrained(
                    self.model_path, torch_dtype=torch_dtype,
                )
                self.chat_pipeline.to(self.device)

            if hasattr(self.chat_pipeline, "transformer") and not hasattr(self.chat_pipeline.transformer, "enable_teacache"):
                self.chat_pipeline.transformer.enable_teacache = False
            if self.enable_cpu_offload:
                self.chat_pipeline.enable_model_cpu_offload()
            if self.enable_sequential_cpu_offload:
                self.chat_pipeline.enable_sequential_cpu_offload()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self, cfg: dict[str, Any]) -> None:
        """Load or update configuration."""
        changed = False
        for key, attr in [("model_path", "model_path"), ("device", "device"),
                          ("torch_dtype", "torch_dtype"), ("device_map", "device_map")]:
            if key in cfg:
                if cfg[key] != getattr(self, attr):
                    changed = True
                setattr(self, attr, cfg[key])
        if "output_dir_t2i" in cfg:
            self.out_dir_t2i = self._resolve_output_dir(str(cfg["output_dir_t2i"]))
            self.out_dir_t2i.mkdir(parents=True, exist_ok=True)
        if "output_dir_understanding" in cfg:
            self.out_dir_understanding = self._resolve_output_dir(str(cfg["output_dir_understanding"]))
            self.out_dir_understanding.mkdir(parents=True, exist_ok=True)
        if "output_dir_editing" in cfg:
            self.out_dir_editing = self._resolve_output_dir(str(cfg["output_dir_editing"]))
            self.out_dir_editing.mkdir(parents=True, exist_ok=True)
        if "omnigen2_root" in cfg:
            self.omnigen2_root = Path(cfg["omnigen2_root"]).expanduser()
            self._modules_registered = False
            changed = True
        # Reset pipelines if model config changed
        if changed:
            self.pipeline = None
            self.chat_pipeline = None

    def generate(self, batch: dict[str, Any] | list[dict[str, Any]], gen_cfg: dict[str, Any]) -> dict[str, Any]:
        """Generate images from text prompts (text-to-image).

        Returns:
            {"images": [PIL.Image], "image_paths": [str], ...} on success.
            {"error": str, "images": [], "image_paths": [], "stderr": str} on failure.
        """
        items = self._to_items(batch)
        prompts = [item.get("prompt", "") for item in items if item.get("prompt")]
        if not prompts:
            raise ValueError("Generation requires a non-empty prompt.")

        output_path = None
        if isinstance(batch, dict):
            output_path = batch.get("output_path")
        if not output_path and items:
            output_path = items[0].get("output_path")
        if output_path:
            output_path = str(Path(output_path).resolve())

        self._ensure_pipeline()

        height = gen_cfg.get("height", 1024)
        width = gen_cfg.get("width", 1024)
        num_inference_steps = gen_cfg.get("num_inference_steps", 50)
        text_guidance_scale = gen_cfg.get("text_guidance_scale", 4.0)
        negative_prompt = gen_cfg.get("negative_prompt", "")
        num_images_per_prompt = gen_cfg.get("num_images_per_prompt", 1)
        seed = gen_cfg.get("seed", 0)

        results: dict[str, Any] = {"image_paths": []}
        images_pil: list[Image.Image] = []
        out_dir = self.out_dir_t2i.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        for prompt_idx, prompt in enumerate(prompts):
            print(f"Generating image {prompt_idx + 1}/{len(prompts)}: {prompt[:80]}...")
            try:
                generator = torch.Generator(device="cpu").manual_seed(seed + prompt_idx)
                output = self.pipeline(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    text_guidance_scale=text_guidance_scale,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    generator=generator,
                    output_type="pil",
                )
                imgs = output.images if hasattr(output, "images") else output
                for img_idx, img in enumerate(imgs):
                    if isinstance(img, Image.Image):
                        if output_path and prompt_idx == 0 and img_idx == 0:
                            save_path = Path(output_path)
                            save_path.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            save_path = out_dir / f"omnigen2_generated_{prompt_idx}_{img_idx}.png"
                        img.save(str(save_path), format="PNG")
                        results["image_paths"].append(str(save_path.resolve()))
                        images_pil.append(img)
                        print(f"  Saved: {save_path}")
            except Exception as e:
                print(f"  Error generating image: {e}")
                results[f"error_prompt_{prompt_idx}"] = str(e)

        if not results["image_paths"]:
            err_details = "\n".join(
                v for k, v in results.items() if isinstance(v, str) and k.startswith("error_prompt_")
            )
            return {
                "error": "No images were generated.",
                "images": [],
                "image_paths": [],
                "stderr": err_details,
                **{k: v for k, v in results.items() if k.startswith("error_prompt_")},
            }
        return {"images": images_pil, **results}

    # Tokens to strip from understanding responses
    _SPECIAL_TOKENS = ("<|img|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>")

    def _generate_text_only(
        self,
        instruction: str,
        input_images: list[Image.Image],
        max_new_tokens: int = 512,
    ) -> str:
        """Call mllm.generate directly with a text-focused system prompt.

        Bypasses chat_pipeline.__call__ which uses an image-generation-biased
        system prompt ("generates high-quality images") that causes the model
        to output <|img|> instead of text reasoning for complex tasks.
        """
        # Build image tags matching _apply_chat_template format
        img_tags = "".join(
            f"<img{i}>: <|vision_start|><|image_pad|><|vision_end|>"
            for i in range(1, len(input_images) + 1)
        )
        # Use a text-analysis-focused system prompt instead of image-generation one
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant that analyzes images and provides "
            "detailed text descriptions and reasoning. "
            "Respond with text only. Do not generate images.<|im_end|>\n"
            f"<|im_start|>user\n{img_tags}{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        cp = self.chat_pipeline
        inputs = cp.prepare_inputs_for_text_generation(
            prompt, input_images, cp.mllm.device
        )
        generated_ids = cp.mllm.generate(
            **inputs,
            tokenizer=cp.processor.tokenizer,
            max_new_tokens=max_new_tokens,
            # Do NOT include <|img|> as stop string — if the model outputs it,
            # subsequent tokens may still contain useful reasoning text.
            stop_strings=["<|im_end|>", "<|endoftext|>"],
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text = cp.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]

        # Clean special tokens from output
        for tok in self._SPECIAL_TOKENS:
            text = text.replace(tok, "")
        return text.strip()

    def understand(self, batch: dict[str, Any] | list[dict[str, Any]], understanding_cfg: dict[str, Any]) -> dict[str, Any]:
        """Understand/describe images using OmniGen2's mllm (Qwen2.5-VL).

        Calls mllm.generate directly with a text-focused system prompt to
        ensure the model reasons in text instead of emitting <|img|> tokens.
        Supports multiple images in a single request.

        Returns:
            {"understandings": [{"image_path": str, "instruction": str, "response": str}, ...]}
        """
        items = self._to_items(batch)

        # Extract all images from the batch (supports both paths and PIL objects)
        input_images_pil: list[Image.Image] = []
        image_path_labels: list[str] = []
        instruction: str = ""
        seen_paths: set[str] = set()

        if items:
            first = items[0]
            if first.get("image_path"):
                p = str(first["image_path"])
                input_images_pil.append(Image.open(p).convert("RGB"))
                image_path_labels.append(p)
                seen_paths.add(p)
            img_list = first.get("images", [])
            if isinstance(img_list, list):
                for img in img_list:
                    if isinstance(img, Image.Image):
                        input_images_pil.append(img.convert("RGB"))
                        image_path_labels.append("<pil_image>")
                    elif isinstance(img, str) and img and img not in seen_paths:
                        input_images_pil.append(Image.open(img).convert("RGB"))
                        image_path_labels.append(img)
                        seen_paths.add(img)
            instruction = first.get("prompt", "")

        # Fallback to understanding_cfg
        if not input_images_pil:
            cfg_path = understanding_cfg.get("image_path")
            if cfg_path:
                input_images_pil.append(Image.open(cfg_path).convert("RGB"))
                image_path_labels.append(cfg_path)
        if not instruction:
            instruction = understanding_cfg.get("instruction") or "Please describe this image briefly."

        if not input_images_pil:
            raise ValueError("Understanding requires at least one image (in request.images or params.image_path).")

        self._ensure_chat_pipeline()

        max_new_tokens = understanding_cfg.get("max_new_tokens", 512)
        seed = understanding_cfg.get("seed", 0)

        results: dict[str, Any] = {"understandings": []}

        print(f"Understanding with {len(input_images_pil)} image(s): {instruction[:80]}...")
        try:
            torch.manual_seed(seed)
            response = self._generate_text_only(
                instruction, input_images_pil, max_new_tokens=max_new_tokens,
            )
            results["understandings"].append({
                "image_path": image_path_labels[0],
                "all_image_paths": image_path_labels,
                "instruction": instruction,
                "response": response,
            })
            print(f"  Response: {response[:100]}...")
        except Exception as e:
            print(f"  Error: {e}")
            results["understandings"].append({
                "image_path": image_path_labels[0],
                "all_image_paths": image_path_labels,
                "instruction": instruction,
                "error": str(e),
            })

        return results

    def edit(self, batch: dict[str, Any] | list[dict[str, Any]], edit_cfg: dict[str, Any]) -> dict[str, Any]:
        """Edit images using OmniGen2.

        Supports multiple input images as references (OmniGen2 can condition
        on up to 5 reference images).

        Args:
            batch: Dict with 'prompt', 'images', 'output_path' keys
                   (standard format from run_editing), or list of dicts
                   with 'image_path', 'instruction' keys (legacy format).
            edit_cfg: Editing configuration.

        Returns:
            {"images": [PIL.Image], "image_paths": [str], ...} on success.
            {"error": str, "images": [], "image_paths": [], "stderr": str} on failure.
        """
        # Normalize batch into (input_images_pil, prompt, output_path)
        # Supports both string paths and PIL Image objects in images list.
        if isinstance(batch, dict):
            prompt = batch.get("prompt", "")
            images = batch.get("images", [])
            output_path = batch.get("output_path")
            image_path = batch.get("image_path")
        else:
            item = self._to_items(batch)[0] if self._to_items(batch) else {}
            prompt = item.get("instruction") or item.get("prompt", "")
            image_path = item.get("image_path")
            images = item.get("images", [])
            output_path = item.get("output_path")

        # Collect inputs as PIL Images, accepting both paths and PIL objects
        input_images_pil: list[Image.Image] = []
        seen_paths: set[str] = set()
        if image_path:
            p = str(image_path)
            input_images_pil.append(Image.open(p).convert("RGB"))
            seen_paths.add(p)
        if isinstance(images, list):
            for img in images:
                if isinstance(img, Image.Image):
                    input_images_pil.append(img.convert("RGB"))
                elif isinstance(img, str) and img and img not in seen_paths:
                    input_images_pil.append(Image.open(img).convert("RGB"))
                    seen_paths.add(img)

        if not input_images_pil:
            raise ValueError("Editing requires at least one image (batch['images'] or batch['image_path']).")
        if not prompt:
            raise ValueError("Editing requires a non-empty prompt.")

        if output_path:
            output_path = str(Path(output_path).resolve())

        self._ensure_pipeline()

        height = edit_cfg.get("height", 1024)
        width = edit_cfg.get("width", 1024)
        num_inference_steps = edit_cfg.get("num_inference_steps", 50)
        text_guidance_scale = edit_cfg.get("text_guidance_scale", 5.0)
        image_guidance_scale = edit_cfg.get("image_guidance_scale", 2.0)
        negative_prompt = edit_cfg.get("negative_prompt", "")
        num_images_per_prompt = edit_cfg.get("num_images_per_prompt", 1)
        seed = edit_cfg.get("seed", 0)

        results: dict[str, Any] = {"image_paths": []}
        images_pil: list[Image.Image] = []
        out_dir = self.out_dir_editing.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Do NOT catch pipeline exceptions here — let them propagate so that
        # generate_image_from_context in run_generation.py can fall back to
        # pure text-to-image generation when editing fails.
        print(f"Editing with {len(input_images_pil)} input image(s): {prompt[:80]}...")
        torch.manual_seed(seed)
        output = self.pipeline(
            prompt=prompt,
            input_images=input_images_pil,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            image_guidance_scale=image_guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
        )
        imgs = output.images if hasattr(output, "images") else output
        for img_idx, img in enumerate(imgs):
            if isinstance(img, Image.Image):
                if output_path and img_idx == 0:
                    save_path = Path(output_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    save_path = out_dir / f"omnigen2_edited_0_{img_idx}.png"
                img.save(str(save_path), format="PNG")
                results["image_paths"].append(str(save_path.resolve()))
                images_pil.append(img)
                print(f"  Saved: {save_path}")

        return {"images": images_pil, **results}
