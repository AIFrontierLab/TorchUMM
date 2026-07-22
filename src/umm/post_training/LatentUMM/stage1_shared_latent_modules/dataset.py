import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        return list(data.values())
    raise ValueError(f"Unsupported prompt record format: {path}")


def _load_embedding(path: Path) -> torch.Tensor:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    values = data.get("values", data.get("embedding", data.get("embeddings")))
    if values is None:
        raise KeyError(f"No embedding vector found in {path}")
    return torch.tensor(values, dtype=torch.float32)


class Stage1SharedLatentDataset(Dataset):
    """
    Paired text/image embedding dataset for Stage 1 dual alignment.

    Expected layout under embedding_root:
      text_embedding/{idx}.json
      image_embedding/{idx}.json

    The JSON files are expected to contain a `values` array. This matches the
    preprocessed Gemini embedding files in umm_reasoning/dataset_embedding/t2i.
    """

    def __init__(
        self,
        prompts_path: str = "/path/to/dataset/t2i/prompts.json",
        image_root: str = "/path/to/dataset/t2i",
        embedding_root: str = "/path/to/dataset_embedding/t2i",
        text_embedding_dir: str = "text_embedding",
        image_embedding_dir: str = "image_embedding",
        load_image: bool = False,
        cache_embeddings: bool = False,
    ) -> None:
        self.prompts_path = Path(prompts_path)
        self.image_root = Path(image_root)
        self.embedding_root = Path(embedding_root)
        self.text_embedding_dir = text_embedding_dir
        self.image_embedding_dir = image_embedding_dir
        self.load_image = load_image
        self.cache_embeddings = cache_embeddings
        self._embedding_cache: Dict[Path, torch.Tensor] = {}

        if self.prompts_path.exists():
            self.records = _load_records(self.prompts_path)
        else:
            text_dir = self.embedding_root / self.text_embedding_dir
            self.records = [{"index": int(p.stem)} for p in sorted(text_dir.glob("*.json"), key=lambda x: int(x.stem))]

        self.indices = self._filter_available_indices()
        if not self.indices:
            raise ValueError(
                "No paired embeddings found. Checked "
                f"{self.embedding_root / self.text_embedding_dir} and "
                f"{self.embedding_root / self.image_embedding_dir}."
            )

    def _filter_available_indices(self) -> List[int]:
        indices = []
        text_dir = self.embedding_root / self.text_embedding_dir
        image_dir = self.embedding_root / self.image_embedding_dir
        for idx, _ in enumerate(self.records):
            text_path = text_dir / f"{idx}.json"
            image_path = image_dir / f"{idx}.json"
            if text_path.exists() and image_path.exists():
                indices.append(idx)
        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def _read_embedding(self, path: Path) -> torch.Tensor:
        if not self.cache_embeddings:
            return _load_embedding(path)
        cached = self._embedding_cache.get(path)
        if cached is None:
            cached = _load_embedding(path)
            self._embedding_cache[path] = cached
        return cached

    def __getitem__(self, item: int) -> Dict[str, Any]:
        idx = self.indices[item]
        record = self.records[idx]
        text_path = self.embedding_root / self.text_embedding_dir / f"{idx}.json"
        image_embedding_path = self.embedding_root / self.image_embedding_dir / f"{idx}.json"

        image_value: Optional[str] = record.get("image") or record.get("path")
        image_path = None
        if image_value is not None:
            candidate = Path(image_value)
            image_path = candidate if candidate.is_absolute() else self.image_root / candidate

        sample = {
            "index": idx,
            "prompt": record.get("prompt", ""),
            "image_path": str(image_path) if image_path is not None else "",
            "text_embedding": self._read_embedding(text_path),
            "image_embedding": self._read_embedding(image_embedding_path),
        }

        if self.load_image:
            if image_path is None:
                raise ValueError(f"Record {idx} does not contain an image path")
            sample["image"] = Image.open(image_path).convert("RGB")

        return sample


def collate_stage1(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "prompt": [item["prompt"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
        "text_embedding": torch.stack([item["text_embedding"] for item in batch], dim=0),
        "image_embedding": torch.stack([item["image_embedding"] for item in batch], dim=0),
    }
