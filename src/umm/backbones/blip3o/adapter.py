from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Optional


class Blip3oBackbone:
    name = "blip3o"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        scale: int = 0,
        seq_len: int = 729,
        top_p: float = 0.95,
        top_k: int = 1200,
        blip3o_root: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        packaged = Path(__file__).resolve().parent / "BLIP3o"
        self.blip3o_root = (
            Path(blip3o_root).expanduser()
            if blip3o_root
            else (packaged if packaged.exists() else repo_root / "model" / "blip3o" / "BLIP3o")
        )
        self.model_path = model_path or str(repo_root / "model_cache" / "blip3o" / "models" / "blip3o_next_sft_3b")
        self.device = device
        self.torch_dtype = torch_dtype
        self.scale = scale
        self.seq_len = seq_len
        self.top_p = top_p
        self.top_k = top_k
        out = Path(output_dir).expanduser() if output_dir else (repo_root / "output" / "blip3o_images")
        self.output_dir = out if out.is_absolute() else (repo_root / out)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load(self, cfg: dict[str, Any]) -> None:
        if isinstance(cfg.get("model_path"), str):
            self.model_path = cfg["model_path"]
        if isinstance(cfg.get("device"), str):
            self.device = cfg["device"]
        if isinstance(cfg.get("torch_dtype"), str):
            self.torch_dtype = cfg["torch_dtype"]
        if isinstance(cfg.get("scale"), int):
            self.scale = cfg["scale"]
        if isinstance(cfg.get("seq_len"), int):
            self.seq_len = cfg["seq_len"]
        if isinstance(cfg.get("top_p"), (int, float)):
            self.top_p = float(cfg["top_p"])
        if isinstance(cfg.get("top_k"), int):
            self.top_k = cfg["top_k"]
        if isinstance(cfg.get("blip3o_root"), str):
            self.blip3o_root = Path(cfg["blip3o_root"]).expanduser()
        if isinstance(cfg.get("output_dir"), str):
            out = Path(cfg["output_dir"]).expanduser()
            self.output_dir = out if out.is_absolute() else (Path(__file__).resolve().parents[4] / out)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, batch: dict[str, Any], gen_cfg: dict[str, Any]) -> Any:
        prompt = batch.get("prompt") or gen_cfg.get("prompt")
        if prompt is None:
            raise ValueError("Generation requires a prompt.")
        prompts = prompt if isinstance(prompt, list) else [str(prompt)]
        explicit_output_path = batch.get("output_path") or gen_cfg.get("output_path")
        explicit_output_path_resolved: str | None = None
        if isinstance(explicit_output_path, str) and explicit_output_path:
            explicit = Path(explicit_output_path).expanduser()
            if not explicit.is_absolute():
                explicit = (Path.cwd() / explicit).resolve()
            explicit_output_path_resolved = str(explicit)

        out_dir_cfg = gen_cfg.get("output_dir") or gen_cfg.get("image_save_dir")
        if isinstance(out_dir_cfg, str) and out_dir_cfg:
            out_dir = Path(out_dir_cfg).expanduser()
            if not out_dir.is_absolute():
                out_dir = Path.cwd() / out_dir
        else:
            out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "model_path": self.model_path,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "scale": int(gen_cfg.get("scale", self.scale)),
            "seq_len": int(gen_cfg.get("seq_len", self.seq_len)),
            "top_p": float(gen_cfg.get("top_p", self.top_p)),
            "top_k": int(gen_cfg.get("top_k", self.top_k)),
            "prompts": prompts,
            "output_dir": str(out_dir),
            "output_path": explicit_output_path_resolved,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            params_file = tmppath / "params.json"
            runner_file = tmppath / "runner_blip3o.py"
            results_file = tmppath / "results.json"
            params_file.write_text(json.dumps(params), encoding="utf-8")
            runner_file.write_text(self._runner_code(), encoding="utf-8")

            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{self.blip3o_root}:{existing_pythonpath}" if existing_pythonpath else str(self.blip3o_root)
            )

            proc = subprocess.run(
                [sys.executable, str(runner_file), str(params_file), str(results_file)],
                cwd=str(self.blip3o_root),
                env=env,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                return {
                    "error": f"BLIP3o generation failed with return code {proc.returncode}",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }

            if results_file.exists():
                payload = json.loads(results_file.read_text(encoding="utf-8"))
                return {
                    "images": payload.get("image_paths", []),
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    **payload,
                }

            return {
                "error": "No results file produced",
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }

    @staticmethod
    def _runner_code() -> str:
        return textwrap.dedent(
            """\
            import json
            import sys
            from pathlib import Path

            import torch
            from transformers import AutoTokenizer
            from blip3o.model import blip3oQwenForInferenceLM

            def _dtype_from_string(name: str):
                mapping = {
                    "float32": torch.float32,
                    "fp32": torch.float32,
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                }
                return mapping.get(str(name).lower(), torch.bfloat16)

            def main():
                params_path = Path(sys.argv[1])
                results_path = Path(sys.argv[2])
                params = json.loads(params_path.read_text(encoding="utf-8"))

                model_path = params["model_path"]
                device = params.get("device", "cuda:0")
                dtype = _dtype_from_string(params.get("torch_dtype", "bfloat16"))
                scale = int(params.get("scale", 0))
                seq_len = int(params.get("seq_len", 729))
                top_p = float(params.get("top_p", 0.95))
                top_k = int(params.get("top_k", 1200))
                prompts = params.get("prompts", [])
                output_dir = Path(params["output_dir"])
                output_path = params.get("output_path")
                output_dir.mkdir(parents=True, exist_ok=True)

                model = blip3oQwenForInferenceLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
                tokenizer = AutoTokenizer.from_pretrained(model_path)

                results = {"image_paths": []}
                for i, prompt in enumerate(prompts):
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Please generate image based on the following caption: {prompt}"},
                    ]
                    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    input_text += f"<im_start><S{scale}>"
                    inputs = tokenizer(
                        [input_text],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        padding_side="left",
                    )

                    _, output_image = model.generate_images(
                        inputs.input_ids.to(device),
                        inputs.attention_mask.to(device),
                        max_new_tokens=seq_len,
                        do_sample=True,
                        top_p=top_p,
                        top_k=top_k,
                    )
                    image = output_image[0]
                    if output_path and len(prompts) == 1:
                        out_path = Path(output_path).expanduser()
                    elif output_path and len(prompts) > 1:
                        base = Path(output_path).expanduser()
                        out_path = base.with_name(f"{base.stem}_{i}{base.suffix or '.png'}")
                    else:
                        out_path = output_dir / f"blip3o_generated_{i}.png"
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(str(out_path), format="PNG")
                    results["image_paths"].append(str(out_path))

                results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

            if __name__ == "__main__":
                main()
            """
        )
