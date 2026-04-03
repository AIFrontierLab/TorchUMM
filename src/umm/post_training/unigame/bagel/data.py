from __future__ import annotations

import json
from collections import Counter
from io import BytesIO
from typing import Any

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def _to_answer_dict_list(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []

    if isinstance(value, list):
        out: list[dict[str, str]] = []
        for item in value:
            if isinstance(item, dict):
                s = item.get("answer")
                if s is None:
                    s = item.get("text")
                if s is not None and str(s).strip():
                    out.append({"answer": str(s)})
            else:
                s = str(item).strip()
                if s:
                    out.append({"answer": s})
        return out

    if isinstance(value, dict):
        s = value.get("answer")
        if s is None:
            s = value.get("text")
        return [{"answer": str(s)}] if s is not None and str(s).strip() else []

    s = str(value).strip()
    return [{"answer": s}] if s else []


def extract_answers_from_example(ex: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("answers", "answer", "multiple_choice_answer", "label", "labels", "target", "answers_text"):
        if key in ex:
            return _to_answer_dict_list(ex[key])
    return []


def extract_question_from_example(ex: dict[str, Any]) -> str:
    for key in ("question", "query", "prompt", "text"):
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


class VQADefaultAdapter:
    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": ex.get("image"),
            "question": extract_question_from_example(ex),
            "answers": extract_answers_from_example(ex),
        }


class TextToImageAdapter:
    """Adapter for image-caption shards such as webdataset jpg+json tar files."""

    def __init__(self, question_prompt: str = "Describe this image.") -> None:
        self.question_prompt = str(question_prompt)

    @staticmethod
    def _pick_caption(obj: Any, depth: int = 0) -> str:
        if depth > 3 or obj is None:
            return ""

        if isinstance(obj, str):
            s = obj.strip()
            return s if s else ""

        if isinstance(obj, dict):
            for key in ("caption", "text", "txt", "prompt", "description", "alt_text", "title"):
                if key in obj:
                    s = TextToImageAdapter._pick_caption(obj.get(key), depth + 1)
                    if s:
                        return s
            for value in obj.values():
                s = TextToImageAdapter._pick_caption(value, depth + 1)
                if s:
                    return s
            return ""

        if isinstance(obj, list):
            for value in obj:
                s = TextToImageAdapter._pick_caption(value, depth + 1)
                if s:
                    return s
            return ""

        return ""

    @staticmethod
    def _pick_image(ex: dict[str, Any]) -> Image.Image | None:
        for key in ("image", "jpg", "jpeg", "png", "webp"):
            raw = ex.get(key)
            if raw is None:
                continue
            if isinstance(raw, Image.Image):
                return raw.convert("RGB")
            if isinstance(raw, (bytes, bytearray)):
                return Image.open(BytesIO(raw)).convert("RGB")
        return None

    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        image = self._pick_image(ex)

        caption = ""
        json_raw = ex.get("json")
        if json_raw is not None:
            if isinstance(json_raw, (bytes, bytearray)):
                json_raw = json_raw.decode("utf-8", errors="replace")
            if isinstance(json_raw, str):
                try:
                    caption = self._pick_caption(json.loads(json_raw))
                except Exception:
                    caption = self._pick_caption(json_raw)
            else:
                caption = self._pick_caption(json_raw)

        if not caption:
            for key in ("caption", "text", "txt", "prompt", "description", "alt_text", "title"):
                value = ex.get(key)
                if value is not None:
                    caption = self._pick_caption(value)
                    if caption:
                        break

        return {
            "image": image,
            "question": self.question_prompt,
            "answers": [{"answer": caption}] if caption else [],
        }


def build_sample_adapter(cfg: dict[str, Any]) -> Any:
    dataset_name = str(cfg.get("dataset_name", "vqav2")).strip().lower()
    if dataset_name in ("text_to_image", "t2i", "text2image"):
        return TextToImageAdapter(question_prompt=str(cfg.get("t2i_question_prompt", "Describe this image.")))
    return VQADefaultAdapter()


class VQAHFDataset(Dataset):
    def __init__(
        self,
        hf_split: Any,
        image_size: int | None = None,
        return_pil: bool = False,
        sample_adapter: Any | None = None,
        skip_overlong_samples: bool = False,
        max_question_chars: int = 0,
        max_answer_chars: int = 0,
        max_resample_attempts: int = 32,
    ) -> None:
        self.ds = hf_split
        self.image_size = image_size
        self.return_pil = return_pil
        self.sample_adapter = sample_adapter or VQADefaultAdapter()
        self.skip_overlong_samples = bool(skip_overlong_samples)
        self.max_question_chars = int(max_question_chars)
        self.max_answer_chars = int(max_answer_chars)
        self.max_resample_attempts = max(1, int(max_resample_attempts))

    def __len__(self) -> int:
        return len(self.ds)

    def _is_overlong(self, question: str, answers: list[dict[str, Any]]) -> bool:
        if self.max_question_chars > 0 and len(question) > self.max_question_chars:
            return True
        if self.max_answer_chars > 0:
            answer_text = _majority_answer(answers)
            if len(answer_text) > self.max_answer_chars:
                return True
        return False

    def _load_item(self, idx: int) -> dict[str, Any]:
        ex = self.ds[idx]
        norm = self.sample_adapter(ex)
        img = norm.get("image")
        if img is None:
            raise ValueError("Sample has no image after dataset adaptation.")

        question = str(norm.get("question", ""))
        answers = norm.get("answers", [])
        if self.skip_overlong_samples and self._is_overlong(question, answers):
            raise ValueError("Sample exceeds configured text-length limits.")

        if self.return_pil:
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            image_out: Image.Image | torch.Tensor = img
        else:
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            t = TF.to_tensor(img)
            if t.dim() == 3 and t.size(0) == 1:
                t = t.expand(3, -1, -1)
            elif t.dim() == 3 and t.size(0) == 4:
                t = t[:3, ...]
            image_out = t

        return {
            "image": image_out,
            "question": question,
            "answers": answers,
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.skip_overlong_samples:
            return self._load_item(idx)

        ds_len = len(self.ds)
        for offset in range(min(self.max_resample_attempts, ds_len)):
            try:
                return self._load_item((idx + offset) % ds_len)
            except ValueError:
                continue
        raise RuntimeError(
            "Failed to find a valid Bagel training sample within max_resample_attempts. "
            "Consider relaxing max_question_chars/max_answer_chars."
        )


def _majority_answer(answers_list: list[dict[str, Any]]) -> str:
    toks: list[str] = []
    for item in answers_list:
        ans = (item.get("answer") or "").strip().lower()
        if ans:
            toks.append(ans)
    if not toks:
        return "unknown"
    return Counter(toks).most_common(1)[0][0]


def vqa_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if torch.is_tensor(batch[0]["image"]):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
    else:
        imgs = torch.stack([TF.to_tensor(b["image"].convert("RGB")) for b in batch], dim=0)

    qs = [b["question"] for b in batch]
    answers = [b.get("answers", []) for b in batch]
    ans_txt = [_majority_answer(a) for a in answers]

    return {
        "image": imgs,
        "question": qs,
        "answer_text": ans_txt,
        "answers": answers,
    }
