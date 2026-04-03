from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional


class TokenFlowBackbone:
    name = "tokenflow"

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        cfg: float = 7.5,
        loop: int = 1,
        mixed_precision: str = "bf16",
        batch_size: int = 1,
        output_dir: str = "output/tokenflow_images",
        tokenflow_root: Optional[str] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "TokenFlow"
        self.tokenflow_root = (
            Path(tokenflow_root).expanduser()
            if tokenflow_root
            else (packaged if packaged.exists() else repo_root / "model" / "tokenflow" / "TokenFlow")
        )
        self.t2i_root = self.tokenflow_root / "t2i"
        self.model_path = model_path or str(repo_root / "model_cache" / "tokenflow" / "models" / "tokenflow-t2i")
        self.tokenizer_path = tokenizer_path or str(
            repo_root / "model_cache" / "tokenflow" / "models" / "tokenflow-t2i-tokenizer" / "tokenflow_clipb_32k_enhanced.pt"
        )
        self.cfg = cfg
        self.loop = loop
        self.mixed_precision = mixed_precision
        self.batch_size = batch_size
        out = Path(output_dir).expanduser()
        self.output_dir = out if out.is_absolute() else (repo_root / out)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, cfg: dict[str, Any]) -> None:
        if isinstance(cfg.get("model_path"), str):
            self.model_path = cfg["model_path"]
        if isinstance(cfg.get("tokenizer_path"), str):
            self.tokenizer_path = cfg["tokenizer_path"]
        if isinstance(cfg.get("cfg"), (float, int)):
            self.cfg = float(cfg["cfg"])
        if isinstance(cfg.get("loop"), int):
            self.loop = int(cfg["loop"])
        if isinstance(cfg.get("mixed_precision"), str):
            self.mixed_precision = cfg["mixed_precision"]
        if isinstance(cfg.get("batch_size"), int):
            self.batch_size = int(cfg["batch_size"])
        if isinstance(cfg.get("tokenflow_root"), str):
            self.tokenflow_root = Path(cfg["tokenflow_root"]).expanduser()
            self.t2i_root = self.tokenflow_root / "t2i"
        if isinstance(cfg.get("output_dir"), str):
            out = Path(cfg["output_dir"]).expanduser()
            if not out.is_absolute():
                out = (Path.cwd() / out).resolve()
            self.output_dir = out
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_prompts(batch: dict[str, Any], gen_cfg: dict[str, Any]) -> list[str]:
        prompt_value = batch.get("prompt", gen_cfg.get("prompt"))
        if prompt_value is None:
            return []
        if isinstance(prompt_value, str):
            return [prompt_value]
        if isinstance(prompt_value, list):
            return [str(item) for item in prompt_value if isinstance(item, (str, int, float))]
        return [str(prompt_value)]

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> dict[str, Any]:
        prompts = self._normalize_prompts(batch, gen_cfg)
        if not prompts:
            raise ValueError("TokenFlow generation requires a non-empty prompt.")

        out_dir = self.output_dir
        output_path = batch.get("output_path") or gen_cfg.get("output_path")
        if isinstance(output_path, str) and output_path:
            output_path_obj = Path(output_path).expanduser()
            if not output_path_obj.is_absolute():
                output_path_obj = (Path.cwd() / output_path_obj).resolve()
            out_dir = output_path_obj.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "cfg": float(gen_cfg.get("cfg", self.cfg)),
            "loop": int(gen_cfg.get("loop", self.loop)),
            "mixed_precision": str(gen_cfg.get("mixed_precision", self.mixed_precision)),
            "batch_size": int(gen_cfg.get("batch_size", self.batch_size)),
            "g_seed": gen_cfg.get("g_seed", None),
            "prompts": prompts,
            "output_dir": str(out_dir),
            "output_path": str(output_path_obj) if isinstance(output_path, str) and output_path else None,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            params_path = tmp_path / "params.json"
            runner_path = tmp_path / "runner_tokenflow.py"
            results_path = tmp_path / "results.json"
            params_path.write_text(json.dumps(params), encoding="utf-8")
            runner_path.write_text(self._runner_code(), encoding="utf-8")

            env = dict(os.environ)
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(self.t2i_root) + (f":{existing}" if existing else "")

            proc = subprocess.run(
                [sys.executable, str(runner_path), str(params_path), str(results_path)],
                cwd=str(self.t2i_root),
                capture_output=True,
                text=True,
                env=env,
            )

            if proc.returncode != 0:
                return {
                    "error": f"TokenFlow generation failed with return code {proc.returncode}",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

            if results_path.exists():
                payload = json.loads(results_path.read_text(encoding="utf-8"))
                return {
                    "images": payload.get("image_paths", []),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    **payload,
                }
            return {"error": "No results file produced", "stdout": proc.stdout, "stderr": proc.stderr}

    @staticmethod
    def _runner_code() -> str:
        return textwrap.dedent(
            """\
            import argparse
            import json
            import os
            from pathlib import Path

            import numpy as np
            import torch
            import transformers
            from PIL import Image

            from llava_t2i.dataset.process import crop_and_encode_text_and_img
            from llava_t2i.model import LlavaLlamaForCausalLM
            from llava_t2i.utils import disable_torch_init


            MULTI_STEP_INFER_STRATEGY = {
                1: {"topk_list": [600], "topp_list": [0.6]},
                2: {"topk_list": [1200, 1], "topp_list": [0.8, 0]},
                3: {"topk_list": [1200, 100, 1], "topp_list": [0.8, 0.8, 0]},
            }


            def parse_args():
                p = argparse.ArgumentParser()
                p.add_argument("params_path")
                p.add_argument("results_path")
                return p.parse_args()


            def main():
                args = parse_args()
                params = json.loads(Path(args.params_path).read_text(encoding="utf-8"))

                model_path = params["model_path"]
                tokenizer_path = params["tokenizer_path"]
                cfg = float(params.get("cfg", 7.5))
                loop = int(params.get("loop", 1))
                mixed_precision = params.get("mixed_precision", "bf16")
                batch_size = int(params.get("batch_size", 1))
                g_seed = params.get("g_seed")
                prompts = [str(x).strip() for x in params.get("prompts", []) if str(x).strip()]
                output_dir = Path(params["output_dir"])
                output_path = params.get("output_path")

                if not prompts:
                    raise ValueError("No prompts provided.")
                if loop not in MULTI_STEP_INFER_STRATEGY:
                    raise ValueError(f"Unsupported loop={loop}. Supported: {sorted(MULTI_STEP_INFER_STRATEGY.keys())}")

                output_dir.mkdir(parents=True, exist_ok=True)
                disable_torch_init()

                if not torch.cuda.is_available():
                    raise RuntimeError("TokenFlow t2i currently requires CUDA.")
                torch.set_default_tensor_type("torch.cuda.FloatTensor")

                ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[mixed_precision]
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    attn_implementation="eager",
                    mm_vision_tower=tokenizer_path,
                )
                model = model.eval().to(ptdtype).cuda()
                model.get_vision_tower().to(ptdtype)
                model.config.mm_vision_vq_type = str(model.config.mm_vision_vq_type)
                model.config.use_cache = False

                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    model_path,
                    model_max_length=model.config.tokenizer_model_max_length,
                    padding_side="right",
                    use_fast=False,
                )
                model.reinit_image_token_start_end(tokenizer)

                negative_prompt = (
                    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, "
                    "cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, "
                    "username, blurry."
                )
                topk_list = MULTI_STEP_INFER_STRATEGY[loop]["topk_list"]
                topp_list = MULTI_STEP_INFER_STRATEGY[loop]["topp_list"]

                image_paths = []
                total = len(prompts)
                current = 0
                while current < total:
                    cur_prompts = prompts[current : current + batch_size]
                    prefix_text_codes = []
                    for p in cur_prompts:
                        input_id, _ = crop_and_encode_text_and_img(tokenizer, p, image=None, max_text_token_num=128)
                        prefix_text_codes.append(input_id)

                    uncondition_input_id, _ = crop_and_encode_text_and_img(
                        tokenizer, negative_prompt, image=None, max_text_token_num=128
                    )
                    prefix_text_codes += [uncondition_input_id] * len(cur_prompts)

                    with torch.inference_mode():
                        samples = model.autoregressive_infer_cfg(
                            B=len(cur_prompts),
                            prefix_text_codes=prefix_text_codes,
                            cfg=cfg,
                            topk_list=topk_list,
                            topp_list=topp_list,
                            g_seed=g_seed,
                        )

                    for i, img in enumerate(samples):
                        idx = current + i
                        if output_path and total == 1:
                            out_path = Path(output_path).expanduser()
                        elif output_path:
                            base = Path(output_path).expanduser()
                            out_path = base.with_name(f"{base.stem}_{idx}{base.suffix or '.png'}")
                        else:
                            out_path = output_dir / f"tokenflow_generated_{idx}.png"
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(np.asarray(img).astype(np.uint8)).save(str(out_path), format="PNG")
                        image_paths.append(str(out_path))

                    current += len(cur_prompts)
                    torch.cuda.empty_cache()

                result = {"image_paths": image_paths}
                Path(args.results_path).write_text(json.dumps(result, indent=2), encoding="utf-8")


            if __name__ == "__main__":
                main()
            """
        )
