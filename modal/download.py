"""
Download model weights and datasets to Modal Volumes, and sync codebase.

Run once per model to populate the volume. After that, every inference/training
run mounts the same volume — no re-downloading.

Usage:
    # Sync local codebase to Modal Volume (run after code changes)
    modal run modal/download.py --sync

    # Download a specific model's weights
    modal run modal/download.py --model bagel
    modal run modal/download.py --model blip3o
    modal run modal/download.py --model emu3
    modal run modal/download.py --model omnigen2
    modal run modal/download.py --model show_o2
    modal run modal/download.py --model tokenflow

    # Download a specific dataset
    modal run modal/download.py --dataset blip3o
    modal run modal/download.py --dataset ueval

    # Download everything (models + datasets)
    modal run modal/download.py --all

    # List what's already cached
    modal run modal/download.py --ls
"""

import os

import modal

from config import (
    DATASET_CACHE_PATH,
    DATASET_URL_ARCHIVES,
    HF_DATASETS,
    HF_MODELS,
    HF_SINGLE_FILES,
    MODEL_CACHE_PATH,
    POST_TRAIN_MODEL_CACHE_PATH,
    URL_DOWNLOADS,
    VLU_DATASETS,
    WORKSPACE_PATH,
)
from volumes import codebase_volume, dataset_volume, model_volume, post_train_model_volume

# Lightweight image for downloading — only needs huggingface_hub
download_image = (
    modal.Image.debian_slim(python_version="3.10")
    # no extra apt packages needed — tar is built-in
    .pip_install(["huggingface_hub>=0.29.0", "datasets", "pandas", "Pillow", "openpyxl"])
    .add_local_python_source("config", "volumes")
)

# Image for syncing codebase — includes rsync and the local codebase
sync_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(["rsync"])
    .add_local_python_source("config", "volumes")
    .add_local_dir(".", remote_path="/tmp/codebase", ignore=[
        "model_cache/", ".git/", "**/.git/", "__pycache__/", ".venv/", "*.egg-info",
        "CLAUDE.md", "modal_setup_readme.md", "README2ME.md", "output/",
    ])
)

app = modal.App("umm-download")


# ---------------------------------------------------------------------------
# Sync codebase to Volume
# ---------------------------------------------------------------------------

@app.function(
    image=sync_image,
    volumes={WORKSPACE_PATH: codebase_volume},
    timeout=3600,
)
def sync_codebase(force: bool = False) -> None:
    """Sync local codebase from /tmp/codebase to the codebase Volume."""
    import subprocess

    if force:
        import shutil
        print("[sync] Force mode: clearing volume before sync...")
        if os.path.isdir(WORKSPACE_PATH):
            for entry in os.listdir(WORKSPACE_PATH):
                path = os.path.join(WORKSPACE_PATH, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
        codebase_volume.commit()
        print("[sync] Volume cleared.")

    print("[sync] Syncing codebase to Volume...")
    subprocess.run(
        ["rsync", "-a", "--delete", "/tmp/codebase/", f"{WORKSPACE_PATH}/"],
        check=True,
    )
    codebase_volume.commit()
    print("[sync] Codebase synced and committed to Volume.")


# ---------------------------------------------------------------------------
# Core download functions
# ---------------------------------------------------------------------------

@app.function(
    image=download_image,
    volumes={MODEL_CACHE_PATH: model_volume},
    timeout=14400,  # 4 hours max
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_model(repo_name: str) -> None:
    """Download model weights to the model-cache Volume (excludes post_train)."""
    import os
    from huggingface_hub import snapshot_download

    if repo_name not in HF_MODELS and repo_name not in URL_DOWNLOADS:
        available = ", ".join(set(HF_MODELS.keys()) | set(URL_DOWNLOADS.keys()))
        raise ValueError(f"Unknown model repo '{repo_name}'. Available: {available}")
    if repo_name == "post_train":
        raise ValueError("post_train uses a separate volume. Use download_post_train instead.")

    hf_token = os.environ.get("HF_TOKEN")

    for hf_repo_id, local_subdir in HF_MODELS.get(repo_name, []):
        local_dir = f"{MODEL_CACHE_PATH}/{local_subdir}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"[download] {hf_repo_id} → {local_dir}")
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
        )
        print(f"[download] done: {hf_repo_id}")

    # Single-file downloads (e.g. VAE weights from large repos)
    if repo_name in HF_SINGLE_FILES:
        from huggingface_hub import hf_hub_download

        for hf_repo_id, filename, local_path in HF_SINGLE_FILES[repo_name]:
            dest = f"{MODEL_CACHE_PATH}/{local_path}"
            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"[download] single file: {hf_repo_id}/{filename} → {dest}")
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=filename,
                local_dir=dest_dir,
                token=hf_token,
            )
            # hf_hub_download saves as {local_dir}/{filename}, rename if needed
            downloaded = f"{dest_dir}/{filename}"
            if downloaded != dest:
                os.rename(downloaded, dest)
            print(f"[download] done: {filename}")

    # Direct URL downloads (non-HuggingFace files)
    if repo_name in URL_DOWNLOADS:
        from urllib.request import urlretrieve

        for url, local_path in URL_DOWNLOADS[repo_name]:
            dest = f"{MODEL_CACHE_PATH}/{local_path}"
            if os.path.exists(dest):
                print(f"[download] already exists, skipping: {dest}")
                continue
            dest_dir = os.path.dirname(dest)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"[download] URL: {url} → {dest}")
            urlretrieve(url, dest)
            print(f"[download] done: {local_path}")

    model_volume.commit()
    print(f"[download] committed Volume for {repo_name}")


@app.function(
    image=download_image,
    volumes={POST_TRAIN_MODEL_CACHE_PATH: post_train_model_volume},
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_post_train(folders: str = "") -> None:
    """Download post_train model weights to its dedicated Volume.

    Args:
        folders: comma-separated list of subdirectories to download (e.g. "Janus_pro,tokenflow").
                 If empty, downloads the entire repo.
    """
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    # Build allow_patterns if specific folders requested
    allow_patterns = None
    if folders:
        folder_list = [f.strip() for f in folders.split(",") if f.strip()]
        allow_patterns = [f"{folder}/*" for folder in folder_list]
        print(f"[download] filtering to folders: {folder_list}")

    for hf_repo_id, local_subdir in HF_MODELS["post_train"]:
        local_dir = f"{POST_TRAIN_MODEL_CACHE_PATH}/{local_subdir}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"[download] {hf_repo_id} → {local_dir}")
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
            token=hf_token,
            ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            allow_patterns=allow_patterns,
        )
        print(f"[download] done: {hf_repo_id}")

    post_train_model_volume.commit()
    print("[download] committed post_train Volume")


@app.function(
    image=download_image,
    volumes={DATASET_CACHE_PATH: dataset_volume},

    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_dataset(repo_name: str) -> None:
    """Download datasets for a given repo to the datasets-cache Volume."""
    import os
    from huggingface_hub import snapshot_download

    if repo_name not in HF_DATASETS:
        available = ", ".join(HF_DATASETS.keys())
        raise ValueError(f"Unknown dataset repo '{repo_name}'. Available: {available}")

    hf_token = os.environ.get("HF_TOKEN")

    # imgedit: download only Benchmark.tar, extract, and remap directory names
    if repo_name == "imgedit":
        _download_imgedit(hf_token)
        dataset_volume.commit()
        print(f"[download] committed dataset Volume for {repo_name}")
        return

    # unified_bench: download Image.zip from GitHub release, extract ref images
    if repo_name == "unified_bench":
        _download_unified_bench()
        dataset_volume.commit()
        print(f"[download] committed dataset Volume for {repo_name}")
        return

    for hf_repo_id, local_subdir in HF_DATASETS[repo_name]:
        local_dir = f"{DATASET_CACHE_PATH}/{local_subdir}"
        os.makedirs(local_dir, exist_ok=True)
        print(f"[download] dataset {hf_repo_id} → {local_dir}")
        snapshot_download(
            repo_id=hf_repo_id,
            local_dir=local_dir,
            token=hf_token,
            repo_type="dataset",
        )
        print(f"[download] done: {hf_repo_id}")

        # Auto-extract archives (e.g. Uni-MMMU ships data as data.tar)
        import glob
        import subprocess
        for arc in glob.glob(f"{local_dir}/*.tar") + glob.glob(f"{local_dir}/*.tar.gz") + glob.glob(f"{local_dir}/*.tgz"):
            print(f"[download] extracting {arc} → {local_dir}")
            subprocess.run(["tar", "xf", arc, "-C", local_dir], check=True)
            print(f"[download] extracted: {arc}")

    dataset_volume.commit()
    print(f"[download] committed dataset Volume for {repo_name}")


@app.function(
    image=download_image,
    volumes={DATASET_CACHE_PATH: dataset_volume},
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_vlu_dataset(name: str) -> None:
    """Download and prepare a VLU benchmark dataset to the datasets-cache Volume.

    Supports two modes:
    - "cache": pre-cache via datasets.load_dataset() (MMMU, MathVista)
    - "extract": download from HF and extract to raw format (MME, MMBench, MM-Vet)
    """
    import base64
    import json
    import os
    from io import BytesIO
    from pathlib import Path

    from datasets import load_dataset

    if name not in VLU_DATASETS:
        available = ", ".join(VLU_DATASETS.keys())
        raise ValueError(f"Unknown VLU dataset '{name}'. Available: {available}")

    entry = VLU_DATASETS[name]
    hf_repo = entry["hf_repo"]
    local_dir = f"{DATASET_CACHE_PATH}/{entry['local_dir']}"
    fmt = entry["format"]
    os.makedirs(local_dir, exist_ok=True)

    hf_token = os.environ.get("HF_TOKEN")

    if fmt == "cache":
        # Pre-cache via load_dataset — data will be in HF cache format
        print(f"[download] pre-caching {hf_repo} → {local_dir}")
        if name == "mmmu":
            # MMMU has per-subject configs; cache all subjects
            from datasets import get_dataset_config_names
            subjects = get_dataset_config_names(hf_repo, token=hf_token)
            for subj in subjects:
                print(f"[download]   subject: {subj}")
                load_dataset(hf_repo, subj, split="validation", cache_dir=local_dir, token=hf_token)
        elif name == "mathvista":
            load_dataset(hf_repo, cache_dir=local_dir, token=hf_token)
        else:
            load_dataset(hf_repo, cache_dir=local_dir, token=hf_token)
        print(f"[download] done: {hf_repo}")

    elif fmt == "extract":
        print(f"[download] extracting {hf_repo} → {local_dir}")

        if name == "mme":
            _extract_mme(hf_repo, local_dir, hf_token)
        elif name == "mmbench":
            _extract_mmbench(hf_repo, local_dir, hf_token)
        elif name == "mmvet":
            _extract_mmvet(hf_repo, local_dir, hf_token)
        else:
            raise ValueError(f"No extraction logic for '{name}'")
        print(f"[download] done: {hf_repo}")

    dataset_volume.commit()
    print(f"[download] committed dataset Volume for {name}")


def _extract_mme(hf_repo: str, local_dir: str, hf_token: str | None) -> None:
    """Extract lmms-lab/MME parquet → txt question files + image directories."""
    import os
    from pathlib import Path

    from datasets import load_dataset

    ds = load_dataset(hf_repo, split="test", token=hf_token)

    image_root = Path(local_dir) / "MME_Benchmark_release_version"
    question_root = Path(local_dir) / "Your_Results"

    # Group samples by category
    categories: dict[str, list] = {}
    for idx, sample in enumerate(ds):
        cat = sample.get("category", sample.get("subcategory", f"unknown"))
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((idx, sample))

    for cat, samples in categories.items():
        cat_image_dir = image_root / cat
        cat_image_dir.mkdir(parents=True, exist_ok=True)
        question_root.mkdir(parents=True, exist_ok=True)

        lines = []
        for idx, sample in samples:
            image = sample.get("image")
            question = sample.get("question", "")
            answer = sample.get("answer", "")
            img_name = f"{idx:06d}.jpg"

            if image is not None:
                img_path = cat_image_dir / img_name
                if not img_path.exists():
                    image.save(str(img_path))

            lines.append(f"{img_name}\t{question}\t{answer}")

        txt_path = question_root / f"{cat}.txt"
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[download]   {cat}: {len(samples)} samples")


def _extract_mmbench(hf_repo: str, local_dir: str, hf_token: str | None) -> None:
    """Extract lmms-lab/MMBench parquet → TSV, plus fetch V11 test TSVs for local scoring.

    Online MMBench evaluation was shut down on 2026-03-31; OpenCompass responded by
    publishing the full TEST splits (EN/CN, V11) with ground-truth labels so we can
    now score TEST entirely locally. See open-compass/MMBench#61.
    """
    import base64
    import os
    import urllib.request
    from io import BytesIO
    from pathlib import Path

    import pandas as pd
    from datasets import load_dataset

    # Load the English dev split
    try:
        ds = load_dataset(hf_repo, "en", split="dev", token=hf_token)
    except Exception:
        # Fallback: try without config name
        ds = load_dataset(hf_repo, split="dev", token=hf_token)

    rows = []
    for sample in ds:
        # Convert PIL image to base64
        image = sample.get("image")
        img_b64 = ""
        if image is not None:
            buf = BytesIO()
            image.save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        row = {
            "index": sample.get("index", 0),
            "question": sample.get("question", ""),
            "hint": sample.get("hint", ""),
            "A": sample.get("A", ""),
            "B": sample.get("B", ""),
            "C": sample.get("C", ""),
            "D": sample.get("D", ""),
            "answer": sample.get("answer", ""),
            "category": sample.get("category", ""),
            "source": sample.get("source", ""),
            "l2-category": sample.get("l2-category", sample.get("l2_category", "")),
            "comment": sample.get("comment", ""),
            "image": img_b64,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = Path(local_dir) / "mmbench_dev_20230712.tsv"
    df.to_csv(str(out_path), sep="\t", index=False)
    print(f"[download]   wrote {len(rows)} rows to {out_path}")

    v11_tsvs = [
        (
            "MMBench_TEST_EN_V11.tsv",
            "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_EN_V11.tsv",
        ),
        (
            "MMBench_TEST_CN_V11.tsv",
            "https://opencompass.openxlab.space/utils/benchmarks/MMBench/MMBench_TEST_CN_V11.tsv",
        ),
    ]
    for fname, url in v11_tsvs:
        target = Path(local_dir) / fname
        if target.exists() and target.stat().st_size > 10_000_000:
            print(f"[download]   {fname} already present ({target.stat().st_size:,} bytes), skipping")
            continue
        print(f"[download]   fetching {fname} ← {url}")
        urllib.request.urlretrieve(url, str(target))
        print(f"[download]   wrote {target.stat().st_size:,} bytes to {target}")


def _extract_mmvet(hf_repo: str, local_dir: str, hf_token: str | None) -> None:
    """Extract lmms-lab/MMVet parquet → JSONL + images."""
    import json
    import os
    from pathlib import Path

    from datasets import load_dataset

    ds = load_dataset(hf_repo, split="test", token=hf_token)

    image_dir = Path(local_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    jsonl_lines = []
    for sample in ds:
        qid = sample.get("question_id", sample.get("id", ""))
        question = sample.get("question", "")
        image = sample.get("image")
        img_name = f"{qid}.jpg"

        if image is not None:
            img_path = image_dir / img_name
            if not img_path.exists():
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(str(img_path))

        jsonl_lines.append(json.dumps({
            "image": img_name,
            "text": question,
            "question_id": str(qid),
        }))

    jsonl_path = Path(local_dir) / "llava-mm-vet.jsonl"
    jsonl_path.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8")
    print(f"[download]   wrote {len(jsonl_lines)} entries to {jsonl_path}")


def _download_imgedit(hf_token: str | None) -> None:
    """Download ImgEdit Benchmark.tar from HuggingFace, extract, and remap dirs.

    The HuggingFace tar contains:
        Benchmark/Basic/      -> singleturn images (animal/*.jpg, style/*.jpg, ...)
        Benchmark/UGE/        -> hard editing images + UGE_edit.json
        Benchmark/Multiturn/  -> multiturn images by category

    Our eval code expects (under origin_img_root = /datasets/imgedit/Benchmark):
        Benchmark/singleturn/ -> origin images for basic edits
        Benchmark/hard/       -> UGE images + annotation.jsonl
        Benchmark/multiturn/  -> multiturn images by category
    """
    import json
    import os
    import subprocess
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    imgedit_dir = f"{DATASET_CACHE_PATH}/imgedit"
    benchmark_dir = Path(imgedit_dir) / "Benchmark"

    # Skip if already extracted and remapped (all three suites must exist)
    if (
        (benchmark_dir / "singleturn").is_dir()
        and (benchmark_dir / "hard").is_dir()
        and (benchmark_dir / "multiturn").is_dir()
    ):
        print("[download] imgedit Benchmark already extracted and remapped, skipping.")
        return

    # Step 1: Download Benchmark.tar
    print("[download] downloading sysuyy/ImgEdit Benchmark.tar ...")
    os.makedirs(imgedit_dir, exist_ok=True)
    tar_path = hf_hub_download(
        repo_id="sysuyy/ImgEdit",
        filename="Benchmark.tar",
        repo_type="dataset",
        local_dir=imgedit_dir,
        token=hf_token,
    )
    print(f"[download] downloaded: {tar_path}")

    # Step 2: Extract tar
    print(f"[download] extracting Benchmark.tar -> {imgedit_dir}")
    subprocess.run(["tar", "xf", tar_path, "-C", imgedit_dir], check=True)
    print("[download] extracted.")

    # Step 3: Remap directory names to match eval code expectations
    #   Basic/    -> singleturn/   (contains category subdirs with images)
    #   UGE/      -> hard/         (contains UGE images + annotation)
    #   Multiturn -> multiturn     (case normalization)
    basic_dir = benchmark_dir / "Basic"
    uge_dir = benchmark_dir / "UGE"
    multiturn_orig = benchmark_dir / "Multiturn"

    singleturn_dir = benchmark_dir / "singleturn"
    hard_dir = benchmark_dir / "hard"
    multiturn_dir = benchmark_dir / "multiturn"

    if basic_dir.is_dir() and not singleturn_dir.exists():
        basic_dir.rename(singleturn_dir)
        print("[download] renamed Basic/ -> singleturn/")

    if uge_dir.is_dir() and not hard_dir.exists():
        uge_dir.rename(hard_dir)
        print("[download] renamed UGE/ -> hard/")

    if multiturn_orig.is_dir() and not multiturn_dir.exists():
        multiturn_orig.rename(multiturn_dir)
        print("[download] renamed Multiturn/ -> multiturn/")

    # Step 4: Convert UGE_edit.json to annotation.jsonl (code expects JSONL format)
    uge_json_path = hard_dir / "UGE_edit.json"
    ann_jsonl_path = hard_dir / "annotation.jsonl"
    if uge_json_path.is_file() and not ann_jsonl_path.exists():
        print("[download] converting UGE_edit.json -> annotation.jsonl")
        with open(uge_json_path, "r", encoding="utf-8") as f:
            uge_data = json.load(f)
        # UGE_edit.json is a dict: {"1": {"id": "xxx.jpg", "prompt": "..."}, ...}
        lines = []
        for key in sorted(uge_data.keys(), key=lambda k: int(k)):
            item = uge_data[key]
            item["key"] = key
            lines.append(json.dumps(item, ensure_ascii=False))
        ann_jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[download] wrote {len(lines)} entries to annotation.jsonl")

    # Clean up tar file to save volume space
    tar_file = Path(imgedit_dir) / "Benchmark.tar"
    if tar_file.is_file():
        tar_file.unlink()
        print("[download] removed Benchmark.tar to save space")

    print("[download] imgedit Benchmark ready.")
    print(f"[download]   singleturn: {singleturn_dir}")
    print(f"[download]   hard (UGE): {hard_dir}")
    print(f"[download]   multiturn:  {multiturn_dir}")


def _download_unified_bench() -> None:
    """Download Unified-Bench (UAE) reference images from GitHub release.

    The GitHub release zip contains only 100 reference images in an `Image/`
    folder — no prompts/metadata, because Unified-Bench measures round-trip
    reconstruction (the backbone generates its own caption from the reference
    image, then regenerates an image from that caption).

    Final layout:
        /datasets/unified_bench/Image/0.jpg ... 99.jpg
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path
    from urllib.request import urlretrieve
    from zipfile import ZipFile

    url, subdir = DATASET_URL_ARCHIVES["unified_bench"]
    target_root = Path(f"{DATASET_CACHE_PATH}/{subdir}")
    image_dir = target_root / "Image"

    # Skip if already populated.
    if image_dir.is_dir() and any(image_dir.iterdir()):
        print(f"[download] unified_bench Image/ already populated at {image_dir}, skipping.")
        return

    target_root.mkdir(parents=True, exist_ok=True)
    zip_path = target_root / "Image.zip"
    print(f"[download] fetching {url} -> {zip_path}")
    urlretrieve(url, str(zip_path))

    print(f"[download] extracting {zip_path} -> {target_root}")
    with ZipFile(str(zip_path)) as zf:
        zf.extractall(str(target_root))

    # The zip may extract as Image/ directly, or nest under a top-level dir.
    if not image_dir.is_dir():
        for child in target_root.iterdir():
            if child.is_dir():
                nested = child / "Image"
                if nested.is_dir():
                    nested.rename(image_dir)
                    break
                # If the zip put images directly in a top-level dir other than Image/,
                # rename that dir to Image/.
                has_images = any(
                    p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for p in child.iterdir() if p.is_file()
                )
                if has_images:
                    child.rename(image_dir)
                    break

    if not image_dir.is_dir():
        raise RuntimeError(
            f"unified_bench extraction did not produce {image_dir}. "
            f"Check zip layout at {target_root}"
        )

    count = sum(1 for p in image_dir.iterdir() if p.is_file())
    print(f"[download] unified_bench ready: {count} reference images at {image_dir}")

    # Clean up the zip to save volume space.
    try:
        zip_path.unlink()
    except Exception:
        pass


@app.function(
    image=download_image,
    volumes={
        MODEL_CACHE_PATH: model_volume,
        DATASET_CACHE_PATH: dataset_volume,
    },

    timeout=36000,  # 10 hours — downloading everything takes a while
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def download_everything() -> None:
    """Download ALL models and datasets. Takes hours."""
    import os
    from huggingface_hub import snapshot_download

    hf_token = os.environ.get("HF_TOKEN")

    print("=" * 60)
    print("[download] Downloading ALL models...")
    print("=" * 60)
    for repo_name, entries in HF_MODELS.items():
        for hf_repo_id, local_subdir in entries:
            local_dir = f"{MODEL_CACHE_PATH}/{local_subdir}"
            os.makedirs(local_dir, exist_ok=True)
            print(f"[download] {hf_repo_id} → {local_dir}")
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=local_dir,
                token=hf_token,
                ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
            )
    model_volume.commit()

    print("=" * 60)
    print("[download] Downloading ALL datasets...")
    print("=" * 60)
    for repo_name, entries in HF_DATASETS.items():
        for hf_repo_id, local_subdir in entries:
            local_dir = f"{DATASET_CACHE_PATH}/{local_subdir}"
            os.makedirs(local_dir, exist_ok=True)
            print(f"[download] dataset {hf_repo_id} → {local_dir}")
            snapshot_download(
                repo_id=hf_repo_id,
                local_dir=local_dir,
                token=hf_token,
                repo_type="dataset",
            )
    dataset_volume.commit()

    print("=" * 60)
    print("[download] ALL downloads complete!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Inspect what's cached
# ---------------------------------------------------------------------------

@app.function(
    image=download_image,
    volumes={
        MODEL_CACHE_PATH: model_volume,
        DATASET_CACHE_PATH: dataset_volume,
    },

)
def list_cache() -> None:
    """Print top-level contents of both Volumes."""
    import os

    print("=== Model Cache ===")
    for item in sorted(os.listdir(MODEL_CACHE_PATH)):
        path = f"{MODEL_CACHE_PATH}/{item}"
        if os.path.isdir(path):
            sub = os.listdir(path)
            print(f"  {item}/ ({len(sub)} items)")
        else:
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {item} ({size_mb:.1f} MB)")

    print("\n=== Dataset Cache ===")
    for item in sorted(os.listdir(DATASET_CACHE_PATH)):
        path = f"{DATASET_CACHE_PATH}/{item}"
        if os.path.isdir(path):
            sub = os.listdir(path)
            print(f"  {item}/ ({len(sub)} items)")
            # Show deeper structure for datasets
            for sub_item in sorted(sub):
                sub_path = f"{path}/{sub_item}"
                if os.path.isdir(sub_path):
                    sub_sub = os.listdir(sub_path)
                    print(f"    {sub_item}/ ({len(sub_sub)} items)")
                    for s in sorted(sub_sub)[:20]:
                        ss_path = f"{sub_path}/{s}"
                        if os.path.isdir(ss_path):
                            print(f"      {s}/")
                        else:
                            sz = os.path.getsize(ss_path) / 1e6
                            print(f"      {s} ({sz:.1f} MB)")
                else:
                    sz = os.path.getsize(sub_path) / 1e6
                    print(f"    {sub_item} ({sz:.1f} MB)")
        else:
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {item} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = "",
    dataset: str = "",
    all: bool = False,
    ls: bool = False,
    sync: bool = False,
    force: bool = False,
    folders: str = "",
) -> None:
    """
    Download models/datasets to Modal Volumes, or sync codebase.

    Examples:
        modal run modal/download.py --sync
        modal run modal/download.py --sync --force
        modal run modal/download.py --model bagel
        modal run modal/download.py --model post_train --folders "Janus_pro,tokenflow,omnigen2,show_o2"
        modal run modal/download.py --dataset ueval
        modal run modal/download.py --all
        modal run modal/download.py --ls
    """
    if sync:
        sync_codebase.remote(force=force)
    elif ls:
        list_cache.remote()
    elif all:
        download_everything.remote()
    elif model:
        if model == "post_train":
            download_post_train.remote(folders=folders)
        else:
            download_model.remote(repo_name=model)
    elif dataset:
        if dataset in HF_DATASETS:
            download_dataset.remote(repo_name=dataset)
        elif dataset in VLU_DATASETS:
            download_vlu_dataset.remote(name=dataset)
        else:
            all_datasets = sorted(set(HF_DATASETS.keys()) | set(VLU_DATASETS.keys()))
            print(f"Unknown dataset '{dataset}'. Available: {', '.join(all_datasets)}")
    else:
        all_datasets = sorted(set(HF_DATASETS.keys()) | set(VLU_DATASETS.keys()))
        print("Usage: modal run modal/download.py --sync | --model <name> | --dataset <name> | --all | --ls")
        print(f"  Models:   {', '.join(HF_MODELS.keys())}")
        print(f"  Datasets: {', '.join(all_datasets)}")
