import ast
import base64
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms.functional as TF
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torch.utils.data import Dataset

from umm.eval.internvl_chat.eval.mmmu.data_utils import CAT_SHORT2LONG


def _to_answer_dict_list(value):
    """Normalize arbitrary answer field to list[{'answer': str}]."""
    if value is None:
        return []

    if isinstance(value, list):
        out = []
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


def extract_answers_from_example(ex):
    """Extract answers from common VQA-style fields and normalize format."""
    for key in (
        "answers",
        "answer",
        "multiple_choice_answer",
        "label",
        "labels",
        "target",
        "answers_text",
    ):
        if key in ex:
            return _to_answer_dict_list(ex[key])
    return []


def extract_question_from_example(ex):
    for key in ("question", "query", "prompt", "text"):
        if key in ex and ex[key] is not None:
            return str(ex[key])
    return ""


class VQADefaultAdapter:
    """Adapter for common VQA-style datasets (e.g., merve/vqav2-small)."""

    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        return {
            "image": ex.get("image"),
            "question": extract_question_from_example(ex),
            "answers": extract_answers_from_example(ex),
        }


class MMMUAdapter:
    """Adapter to normalize MMMU rows into the UniGame training schema."""

    def __init__(
        self,
        mc_prompt_suffix: str = "Answer with the option's letter from the given choices directly.",
        open_prompt_suffix: str = "Answer the question using a single word or phrase.",
        mc_target_mode: str = "letter",
    ):
        self.mc_prompt_suffix = str(mc_prompt_suffix).strip()
        self.open_prompt_suffix = str(open_prompt_suffix).strip()
        self.mc_target_mode = str(mc_target_mode).strip().lower()

    def _parse_options(self, raw_options: Any) -> list[str]:
        if isinstance(raw_options, list):
            return [str(x) for x in raw_options]
        if isinstance(raw_options, str):
            try:
                parsed = ast.literal_eval(raw_options)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return []
        return []

    def _pick_image(self, images: list[Any]) -> Any:
        for image in images:
            if image is not None:
                return image
        return None

    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        question = str(ex.get("question", "")).strip()
        question_type = str(ex.get("question_type", "open"))
        options = self._parse_options(ex.get("options", []))
        images = [
            ex.get("image_1"),
            ex.get("image_2"),
            ex.get("image_3"),
            ex.get("image_4"),
            ex.get("image_5"),
            ex.get("image_6"),
            ex.get("image_7"),
        ]

        if question_type == "multiple-choice" and options:
            letters = list("ABCDEFGHIJKLM")
            choice_lines = [f"{letters[i]}. {opt.strip()}" for i, opt in enumerate(options)]
            if self.mc_prompt_suffix:
                question = f"{question}\n" + "\n".join(choice_lines) + f"\n{self.mc_prompt_suffix}"
            else:
                question = f"{question}\n" + "\n".join(choice_lines)
            answer_raw = str(ex.get("answer", "")).strip()
            answer_text = answer_raw
            if self.mc_target_mode == "text" and answer_raw in letters:
                idx = letters.index(answer_raw)
                if idx < len(options):
                    answer_text = str(options[idx]).strip()
        else:
            if self.open_prompt_suffix:
                question = f"{question}\n{self.open_prompt_suffix}".strip()
            answer_text = str(ex.get("answer", "")).strip()

        image = self._pick_image(images)
        answers = [{"answer": answer_text}] if answer_text else []

        return {
            "image": image,
            "question": question,
            "answers": answers,
        }


class MMBenchAdapter:
    """Adapter to normalize MMBench-style rows into UniGame training schema."""

    def __init__(
        self,
        prompt_suffix_en: str = "Answer with the option's letter from the given choices directly.",
        prompt_suffix_cn: str = "请直接回答选项字母。",
        target_mode: str = "letter",
        include_hint: bool = True,
    ):
        self.prompt_suffix_en = str(prompt_suffix_en).strip()
        self.prompt_suffix_cn = str(prompt_suffix_cn).strip()
        self.target_mode = str(target_mode).strip().lower()
        self.include_hint = bool(include_hint)

    @staticmethod
    def _decode_image(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, str):
            s = value.strip()
            if not s:
                return None
            p = Path(s).expanduser()
            if p.exists():
                return Image.open(p).convert("RGB")
            try:
                raw = base64.b64decode(s)
                return Image.open(BytesIO(raw)).convert("RGB")
            except Exception:
                return None
        return None

    @staticmethod
    def _read_options(ex: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
        letters = ["A", "B", "C", "D", "E"]
        options: dict[str, str] = {}
        for k in letters:
            v = ex.get(k)
            if v is None:
                continue
            sv = str(v).strip()
            if sv:
                options[k] = sv
        return options, letters

    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        image = self._decode_image(ex.get("image"))

        question = str(ex.get("question", "")).strip()
        hint = str(ex.get("hint", "")).strip() if ex.get("hint") is not None else ""
        options, letters = self._read_options(ex)

        if self.include_hint and hint:
            question = f"{hint}\n{question}" if question else hint

        if options:
            question += "\n" if question else ""
            question += "\n".join([f"{k}. {v}" for k, v in options.items()])

        lang = str(ex.get("language", "en")).strip().lower()
        suffix = self.prompt_suffix_cn if lang.startswith("cn") else self.prompt_suffix_en
        if suffix:
            question = f"{question}\n{suffix}".strip()

        answer_raw = ""
        for key in ("answer", "gt_answer", "label"):
            if ex.get(key) is not None:
                answer_raw = str(ex.get(key)).strip()
                if answer_raw:
                    break

        # Convert numeric labels (e.g. 0..4) to option letters.
        if answer_raw.isdigit() and options:
            idx = int(answer_raw)
            if 0 <= idx < len(letters) and letters[idx] in options:
                answer_raw = letters[idx]

        answer_text = answer_raw
        if self.target_mode == "text" and answer_raw in options:
            answer_text = options[answer_raw]

        answers = [{"answer": answer_text}] if answer_text else []
        return {
            "image": image,
            "question": question,
            "answers": answers,
        }


class TextToImageAdapter:
    """Adapter for image-caption webdataset (e.g. jackyhate/text-to-image-2M).

    Maps each image+caption pair to VQA training format so UniGame can
    use captioning data for adversarial training.
    """

    def __init__(self, question_prompt: str = "Describe this image."):
        self.question_prompt = question_prompt

    @staticmethod
    def _pick_caption(obj: Any, depth: int = 0) -> str:
        if depth > 3 or obj is None:
            return ""

        if isinstance(obj, str):
            s = obj.strip()
            return s if s else ""

        if isinstance(obj, dict):
            # Common caption-like keys seen in webdataset/json metadata.
            for key in (
                "caption",
                "text",
                "txt",
                "prompt",
                "description",
                "alt_text",
                "title",
            ):
                if key in obj:
                    s = TextToImageAdapter._pick_caption(obj.get(key), depth + 1)
                    if s:
                        return s

            # Fallback: recursively search nested structures.
            for v in obj.values():
                s = TextToImageAdapter._pick_caption(v, depth + 1)
                if s:
                    return s
            return ""

        if isinstance(obj, list):
            for v in obj:
                s = TextToImageAdapter._pick_caption(v, depth + 1)
                if s:
                    return s
            return ""

        return ""

    def __call__(self, ex: dict[str, Any]) -> dict[str, Any]:
        import json as _json
        from io import BytesIO

        # Find image — handle both decoded PIL and raw bytes
        image = None
        for key in ("image", "jpg", "png", "webp"):
            raw = ex.get(key)
            if raw is None:
                continue
            if isinstance(raw, Image.Image):
                image = raw
                break
            if isinstance(raw, (bytes, bytearray)):
                image = Image.open(BytesIO(raw))
                break

        # Extract caption from json metadata or direct field
        caption = ""
        json_raw = ex.get("json")
        if json_raw is not None:
            if isinstance(json_raw, (bytes, bytearray)):
                json_raw = json_raw.decode("utf-8", errors="replace")
            if isinstance(json_raw, str):
                try:
                    meta = _json.loads(json_raw)
                    caption = self._pick_caption(meta)
                except Exception:
                    caption = self._pick_caption(json_raw)
            elif isinstance(json_raw, dict):
                caption = self._pick_caption(json_raw)
            elif isinstance(json_raw, list):
                caption = self._pick_caption(json_raw)

        if not caption:
            for key in ("caption", "text", "txt", "prompt", "description", "alt_text", "title"):
                v = ex.get(key)
                if v is not None:
                    caption = self._pick_caption(v)
                    if caption:
                        break

        return {
            "image": image,
            "question": self.question_prompt,
            "answers": [{"answer": caption}] if caption else [],
        }


def build_sample_adapter(cfg: dict[str, Any]):
    dataset_name = str(cfg.get("dataset_name", "vqav2")).strip().lower()
    if dataset_name in ("text_to_image", "t2i", "text2image"):
        return TextToImageAdapter(
            question_prompt=str(cfg.get("t2i_question_prompt", "Describe this image.")),
        )
    if dataset_name == "mmmu":
        return MMMUAdapter(
            mc_prompt_suffix=str(
                cfg.get("mmmu_mc_prompt_suffix", "Answer with the option's letter from the given choices directly.")
            ),
            open_prompt_suffix=str(
                cfg.get("mmmu_open_prompt_suffix", "Answer the question using a single word or phrase.")
            ),
            mc_target_mode=str(cfg.get("mmmu_mc_target_mode", "letter")),
        )
    if dataset_name == "mmbench":
        return MMBenchAdapter(
            prompt_suffix_en=str(
                cfg.get("mmbench_prompt_suffix_en", "Answer with the option's letter from the given choices directly.")
            ),
            prompt_suffix_cn=str(cfg.get("mmbench_prompt_suffix_cn", "请直接回答选项字母。")),
            target_mode=str(cfg.get("mmbench_target_mode", "letter")),
            include_hint=bool(cfg.get("mmbench_include_hint", True)),
        )
    return VQADefaultAdapter()


def load_mmmu_split(root: str, split: str, cache_dir: str | None = None):
    """Load and concatenate all MMMU subject subsets for one split."""
    ds_list = []
    for subject in CAT_SHORT2LONG.values():
        ds_list.append(load_dataset(root, subject, split=split, cache_dir=cache_dir))
    return concatenate_datasets(ds_list)

def _soft_score(n: int) -> float:
    return min(1.0, n / 3.0)  

class VQAHFDataset(Dataset):
    def __init__(self, hf_split, ans2id=None, image_size=None, return_pil=True, sample_adapter=None):
        self.ds = hf_split
        self.ans2id = ans2id
        self.image_size = image_size
        self.return_pil = return_pil
        self.sample_adapter = sample_adapter or VQADefaultAdapter()
    
    def __len__(self): 
        return len(self.ds)
    
    def __getitem__(self, idx):
        ex = self.ds[idx]
        norm = self.sample_adapter(ex)
        img = norm.get("image")

        if img is None:
            raise ValueError("Sample has no image after dataset adaptation.")

        if not self.return_pil:
            img = img.convert("RGB")
            if self.image_size is not None:
                img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
            img = TF.to_tensor(img)  

            if img.dim() == 3 and img.size(0) == 1:
                img = img.expand(3, -1, -1)
            elif img.dim() == 3 and img.size(0) == 4:
                img = img[:3, ...]
        return {
            "image": img,
            "question": str(norm.get("question", "")),
            "answers": norm.get("answers", []),
        }

def _majority_answer(answers_list):
    toks = []
    for a in answers_list:
        s = (a.get("answer") or "").strip().lower()
        if s:
            toks.append(s)
    if not toks:
        return "unknown"
    return Counter(toks).most_common(1)[0][0]

def vqa_collate(batch):
    if torch.is_tensor(batch[0]["image"]):
        imgs = torch.stack([b["image"] for b in batch], dim=0)   # [B,3,H,W]
    else:
        imgs = torch.stack([TF.to_tensor(b["image"].convert("RGB")) for b in batch], dim=0)

    qs = [b["question"] for b in batch]
    ans_txt = [_majority_answer(b["answers"]) for b in batch]    # list[str]

    return {"image": imgs, "question": qs, "answer_text": ans_txt}

