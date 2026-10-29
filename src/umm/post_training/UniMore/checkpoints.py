"""Export independent branches and combine their task vectors."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

SIDE_FILES = (
    "ae.safetensors",
    "config.json",
    "llm_config.json",
    "vit_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
    "generation_config.json",
    "vocab.json",
    "merges.txt",
)
DIRECT_PREFIXES = {"t2i": ("time_embedder.", "vae2llm.", "llm2vae."), "i2t": ("connector.",)}
RELEASE_REPO = "HuggingSelf/test0519"
RELEASE_REVISION = "34a5e9d9b847f092f1c72f815ecc986c787f9ed6"
BASE_REPO = "ByteDance-Seed/BAGEL-7B-MoT"
BASE_REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
MODEL_PATHS = {
    "unimore": "checkpoints/n12_l40t2i_qdw_tv_i2t002",
    "t2i": "checkpoints/v2_semantic_ground_implicit_t2i",
    "i2t": "checkpoints/qv3_i2t_top4k_detailwrong_lr18e5_s600_step0599",
}


def weights(path):
    path = Path(path).expanduser().resolve()
    return path / "ema.safetensors" if path.is_dir() else path


def create_output(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Output must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def support_files(base, output):
    for name in SIDE_FILES:
        source = Path(base) / name
        if source.is_file():
            if name == "ae.safetensors":
                (output / name).symlink_to(source.resolve())
            else:
                shutil.copy2(source, output / name)


def export_branch(base, checkpoint, route, output):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if route not in DIRECT_PREFIXES:
        raise ValueError("route must be t2i or i2t")
    output = create_output(output)
    with (
        safe_open(str(weights(base)), framework="pt", device="cpu") as raw,
        safe_open(str(checkpoint), framework="pt", device="cpu") as trained,
    ):
        a_keys = {
            key.removesuffix(".lora_A.default.weight"): key
            for key in trained.keys()
            if key.endswith(".lora_A.default.weight")
        }
        b_keys = {
            key.removesuffix(".lora_B.default.weight"): key
            for key in trained.keys()
            if key.endswith(".lora_B.default.weight")
        }
        if not a_keys or a_keys.keys() != b_keys.keys():
            raise ValueError("A complete set of LoRA A/B pairs is required")
        raw_keys = set(raw.keys())
        required = {key for key in raw_keys if key.startswith(DIRECT_PREFIXES[route])}
        if not required.issubset(trained.keys()):
            raise ValueError("Checkpoint is missing the trained branch projections")
        if any(f"{module}.weight" not in raw_keys for module in a_keys):
            raise ValueError("LoRA modules do not match the base model")
        values = {}
        for key in raw.keys():
            value = raw.get_tensor(key)
            module = key.removesuffix(".weight")
            if key.endswith(".weight") and module in a_keys:
                a, b = trained.get_tensor(a_keys[module]), trained.get_tensor(b_keys[module])
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != 16 or b.shape[1] != 16:
                    raise ValueError(f"Expected rank-16 LoRA: {module}")
                value = value.float().addmm(b.float(), a.float(), beta=1.0, alpha=2.0)
            if key.startswith(DIRECT_PREFIXES[route]) and key in trained.keys():
                updated = trained.get_tensor(key)
                if updated.shape != value.shape:
                    raise ValueError(f"Direct projection shape mismatch: {key}")
                value = updated
            values[key] = value.to(torch.bfloat16)
    temporary = output / "ema.safetensors.tmp"
    save_file(values, str(temporary))
    temporary.replace(output / "ema.safetensors")
    support_files(Path(base), output)


def merge_tensor(base, t2i, i2t):
    import torch

    if base.shape != t2i.shape or base.shape != i2t.shape:
        raise ValueError("Branch tensors must match the base shape")
    if base.dtype != t2i.dtype or base.dtype != i2t.dtype:
        raise ValueError("Base and branch tensor dtypes must match")
    if not base.is_floating_point():
        if not torch.equal(base, t2i) or not torch.equal(base, i2t):
            raise ValueError("Non-floating model state must not change")
        return base
    if base.dtype != torch.bfloat16:
        raise ValueError("The released merge requires BF16 base and branch weights")
    # Match the released checkpoint: round each delta and their sum in BF16.
    value = base + (1.0 * (t2i - base) + 0.02 * (i2t - base))
    if not torch.isfinite(value).all():
        raise ValueError("Non-finite merged tensor")
    return value


def merge_branches(base, t2i, i2t, output):
    from safetensors import safe_open
    from safetensors.torch import save_file

    output = create_output(output)
    with (
        safe_open(str(weights(base)), framework="pt", device="cpu") as raw,
        safe_open(str(weights(t2i)), framework="pt", device="cpu") as gen,
        safe_open(str(weights(i2t)), framework="pt", device="cpu") as und,
    ):
        if set(raw.keys()) != set(gen.keys()) or set(raw.keys()) != set(und.keys()):
            raise ValueError("Base and branch tensor names must match exactly")
        values = {
            key: merge_tensor(raw.get_tensor(key), gen.get_tensor(key), und.get_tensor(key))
            for key in raw.keys()
        }
    temporary = output / "ema.safetensors.tmp"
    save_file(values, str(temporary))
    temporary.replace(output / "ema.safetensors")
    support_files(Path(base), output)
    (output / "merge.json").write_text(
        json.dumps(
            {
                "t2i_weight": 1.0,
                "i2t_weight": 0.02,
                "base": str(base),
                "t2i": str(t2i),
                "i2t": str(i2t),
                "dtype": "bfloat16",
            },
            indent=2,
        )
        + "\n"
    )


def download_model(name, output):
    from huggingface_hub import HfApi, hf_hub_download

    output = create_output(output)
    repo = BASE_REPO if name == "base" else RELEASE_REPO
    revision = BASE_REVISION if name == "base" else RELEASE_REVISION
    prefix = "" if name == "base" else MODEL_PATHS[name] + "/"
    files = set(HfApi().list_repo_files(repo, revision=revision))
    for filename in ("ema.safetensors", *SIDE_FILES):
        source_repo, source_revision, source_file = repo, revision, prefix + filename
        if source_file not in files:
            if filename == "ema.safetensors":
                raise FileNotFoundError(source_file)
            source_repo, source_revision, source_file = BASE_REPO, BASE_REVISION, filename
        try:
            path = hf_hub_download(source_repo, source_file, revision=source_revision)
        except Exception as exc:
            from huggingface_hub.errors import EntryNotFoundError

            if isinstance(exc, EntryNotFoundError) and filename == "special_tokens_map.json":
                continue
            raise
        (output / filename).symlink_to(Path(path).resolve())
