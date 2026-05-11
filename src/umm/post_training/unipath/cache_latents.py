from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

VC_RE = re.compile(r"<VC>(.*?)</VC>", re.DOTALL)
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def _ensure_bagel_code_on_path() -> None:
    from umm.post_training.unipath.bagel_paths import add_bagel_code_to_sys_path

    add_bagel_code_to_sys_path(REPO_ROOT)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_image_reference(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    lower = text.lower()
    return lower.endswith(IMAGE_SUFFIXES) or lower.startswith("images/") or lower.startswith("got/images/")


@lru_cache(maxsize=32)
def _image_path_maps(base_dir: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    map_path = Path(base_dir) / "image_path_map.json"
    if not map_path.exists():
        return {}, {}, {}
    with map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    forward = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    inverse = {value: key for key, value in forward.items()}
    basename_counts: dict[str, int] = {}
    basename_inverse: dict[str, str] = {}
    for key, value in forward.items():
        name = Path(value).name
        basename_counts[name] = basename_counts.get(name, 0) + 1
        basename_inverse[name] = key
    basename_inverse = {name: key for name, key in basename_inverse.items() if basename_counts.get(name) == 1}
    return forward, inverse, basename_inverse


def _candidate_paths(base_dir: Path, value: str) -> list[Path]:
    text = value.strip()
    path = Path(text).expanduser()
    name = path.name
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                base_dir / text,
                base_dir / "images" / name,
                base_dir / "got" / "images" / name,
            ]
        )

    forward, inverse, basename_inverse = _image_path_maps(str(base_dir.resolve()))
    for key in (text, inverse.get(text), basename_inverse.get(name)):
        if key:
            candidates.append(base_dir / key)
    mapped = forward.get(text)
    if mapped:
        candidates.append(Path(mapped).expanduser())

    repo_like_root = base_dir.parents[1] if len(base_dir.parents) >= 2 else base_dir
    if text.startswith("generated_data/l1/scienceqa/images/"):
        candidates.append(repo_like_root / "extracted" / "l1" / "scienceqa" / "images" / name)
    if text.startswith("generated_data/l2/images/"):
        candidates.append(repo_like_root / "extracted" / "l2" / "images" / name)
    if text.startswith("images/"):
        candidates.extend(
            [
                repo_like_root / "extracted" / "data" / "l2" / "images" / name,
                repo_like_root / "extracted" / "l2" / "images" / name,
                repo_like_root / "extracted" / "data" / "l3" / "images" / name,
            ]
        )
    return candidates


def resolve_image_path(base_dir: Path, value: str) -> Path:
    candidates = _candidate_paths(base_dir, value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else (base_dir / value).resolve()


def _latent_index_items(row: dict[str, Any], role: str) -> list[dict[str, Any]]:
    latent_index = row.get("latent_index")
    if not isinstance(latent_index, dict):
        return []
    items = latent_index.get(role)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _refs_from_latent_index(row: dict[str, Any], role: str) -> list[str]:
    refs: list[str] = []
    for item in _latent_index_items(row, role):
        ref = item.get("image_path")
        if is_image_reference(ref):
            refs.append(str(ref))
    return refs


def role_to_references(row: dict[str, Any]) -> list[tuple[str, list[str]]]:
    refs: list[tuple[str, list[str]]] = []

    vc_refs = _refs_from_latent_index(row, "vc")
    if not vc_refs:
        vc_path = row.get("vc_image_path")
        if not is_image_reference(vc_path):
            vc_path = row.get("vc")
        if not is_image_reference(vc_path):
            match = VC_RE.search(str(row.get("trajectory") or ""))
            if match and is_image_reference(match.group(1).strip()):
                vc_path = match.group(1).strip()
        if is_image_reference(vc_path):
            vc_refs = [str(vc_path)]
    if vc_refs:
        refs.append(("vc", vc_refs))

    vh_refs = _refs_from_latent_index(row, "vh")
    if not vh_refs:
        vh_paths = row.get("vh_image_paths")
        if isinstance(vh_paths, list):
            vh_refs = [str(item) for item in vh_paths if is_image_reference(item)]
    if vh_refs:
        refs.append(("vh", vh_refs))

    answer_refs = _refs_from_latent_index(row, "answer")
    if not answer_refs:
        answer_image_path = row.get("answer_image_path")
        if is_image_reference(answer_image_path):
            answer_refs = [str(answer_image_path)]
        elif is_image_reference(row.get("answer")):
            answer_refs = [str(row["answer"])]
    if answer_refs:
        refs.append(("answer", answer_refs))
    return refs


def _existing_latent_path(row: dict[str, Any], role: str, idx: int, cache_dir: Path) -> str | None:
    items = _latent_index_items(row, role)
    if idx >= len(items):
        return None
    rel = items[idx].get("latent_path")
    if not isinstance(rel, str) or not rel:
        return None
    return rel if (cache_dir / rel).exists() else None


def row_needs_latent_cache(row: dict[str, Any], cache_dir: Path) -> bool:
    for role, refs in role_to_references(row):
        for idx, _ in enumerate(refs):
            if _existing_latent_path(row, role, idx, cache_dir) is None:
                return True
    return False


def jsonl_needs_latent_cache(path: Path, cache_dir: Path, max_rows: int | None = None) -> bool:
    rows = read_jsonl(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    return any(row_needs_latent_cache(row, cache_dir) for row in rows)


def jsonl_has_latent_references(path: Path, max_rows: int | None = None) -> bool:
    rows = read_jsonl(path)
    if max_rows is not None:
        rows = rows[:max_rows]
    return any(role_to_references(row) for row in rows)


def encode_image(vae_model: torch.nn.Module, transform: ImageTransform, image_path: Path, device: torch.device) -> dict[str, Any]:
    import torch
    from PIL import Image
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    image = Image.open(image_path).convert("RGB")
    orig_size = [image.height, image.width]
    tensor = transform(image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        latent = vae_model.encode(tensor).to(dtype=torch.float16).cpu()[0].contiguous()
    return {
        "latent": latent,
        "orig_size": orig_size,
        "transformed_size": list(tensor.shape[-2:]),
    }


def cache_row_latents(
    row: dict[str, Any],
    base_dir: Path,
    cache_dir: Path,
    vae_model: torch.nn.Module,
    transform: ImageTransform,
    device: torch.device,
    overwrite: bool,
) -> tuple[dict[str, Any], int, int]:
    updated = dict(row)
    sample_id = str(row.get("id") or row.get("sample_id") or "unknown")
    latent_index: dict[str, Any] = dict(row.get("latent_index") or {})
    written = 0
    missing = 0

    for role, refs in role_to_references(row):
        old_items = _latent_index_items(row, role)
        items: list[dict[str, Any]] = []
        for idx, ref in enumerate(refs):
            old_item = dict(old_items[idx]) if idx < len(old_items) else {}
            rel_cache = old_item.get("latent_path")
            if not isinstance(rel_cache, str) or not rel_cache:
                rel_cache = str(Path(sample_id) / f"{role}_{idx:02d}.pt")
            out_path = cache_dir / rel_cache

            if overwrite or not out_path.exists():
                image_path = resolve_image_path(base_dir, ref)
                if not image_path.exists():
                    missing += 1
                    items.append({"image_path": ref, "missing": True})
                    continue
                try:
                    payload = encode_image(vae_model, transform, image_path, device=device)
                except OSError as exc:
                    missing += 1
                    items.append({"image_path": ref, "broken": True, "error": str(exc)})
                    continue
                out_path.parent.mkdir(parents=True, exist_ok=True)
                import torch

                torch.save(payload, out_path)
                written += 1

            item = {**old_item, "image_path": ref, "latent_path": rel_cache}
            items.append(item)
        if items:
            latent_index[role] = items

    if latent_index:
        updated["latent_index"] = latent_index
    return updated, written, missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache UniPath image latents with BAGEL's VAE encoder.")
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", action="append", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if len(args.input_jsonl) != len(args.output_jsonl):
        raise ValueError("--input-jsonl and --output-jsonl must have the same count.")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Latent caching requires CUDA.")

    device = torch.device("cuda")
    model_path = Path(args.model_path).expanduser()
    cache_dir = Path(args.cache_dir).expanduser()
    ae_path = model_path / "ae.safetensors"
    _ensure_bagel_code_on_path()
    from data.transforms import ImageTransform
    from modeling.autoencoder import load_ae

    vae_model, _ = load_ae(str(ae_path))
    vae_model = vae_model.to(device=device, dtype=torch.bfloat16).eval()
    transform = ImageTransform(1024, 512, 16)

    total_written = 0
    total_missing = 0
    for input_path_str, output_path_str in zip(args.input_jsonl, args.output_jsonl):
        input_path = Path(input_path_str).expanduser()
        output_path = Path(output_path_str).expanduser()
        rows = read_jsonl(input_path)
        if args.max_rows is not None:
            rows = rows[: args.max_rows]
        updated_rows: list[dict[str, Any]] = []
        file_written = 0
        file_missing = 0
        for idx, row in enumerate(rows, start=1):
            updated, written, missing = cache_row_latents(
                row=row,
                base_dir=input_path.parent,
                cache_dir=cache_dir,
                vae_model=vae_model,
                transform=transform,
                device=device,
                overwrite=args.overwrite,
            )
            updated_rows.append(updated)
            file_written += written
            file_missing += missing
            if idx == 1 or (args.progress_every > 0 and idx % args.progress_every == 0):
                print(
                    f"[unipath_cache] file={input_path.name} processed_rows={idx} "
                    f"written={file_written} missing={file_missing}",
                    flush=True,
                )
        write_jsonl(output_path, updated_rows)
        total_written += file_written
        total_missing += file_missing
        print(
            f"[unipath_cache] wrote_jsonl={output_path} rows={len(updated_rows)} "
            f"written={file_written} missing={file_missing}",
            flush=True,
        )
    print(f"[unipath_cache] complete written={total_written} missing={total_missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
