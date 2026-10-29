# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import os
import random
import re
import tarfile
from pathlib import Path
from PIL import Image
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
WORD_RE = re.compile("[A-Za-z][A-Za-z0-9'-]*|\\d+")
WRONG_DETAIL_REPLACEMENT_GROUPS = (
    (
        "attribute",
        (
            ("black", "white"),
            ("white", "black"),
            ("red", "blue"),
            ("blue", "red"),
            ("green", "yellow"),
            ("yellow", "green"),
            ("large", "small"),
            ("small", "large"),
            ("short", "long"),
            ("long", "short"),
            ("clear", "cloudy"),
        ),
    ),
    ("count", (("several", "one"), ("one", "several"), ("two", "three"), ("three", "two"))),
    (
        "relation",
        (
            ("in front of", "behind"),
            ("behind", "in front of"),
            ("above", "below"),
            ("below", "above"),
            ("left", "right"),
            ("right", "left"),
        ),
    ),
)
WRONG_DETAIL_REPLACEMENTS = {
    source: target
    for (_, replacements) in WRONG_DETAIL_REPLACEMENT_GROUPS
    for (source, target) in replacements
}


def _drop_text_span(text, span):
    span = str(span or "").strip()
    if not span:
        return text
    pattern = re.compile(
        "\\b" + "\\s+".join((re.escape(part) for part in span.split())) + "\\b", re.IGNORECASE
    )
    return pattern.sub("", str(text))


def _replace_text_span(text, span, replacement):
    span = str(span or "").strip()
    if not span:
        return text
    pattern = re.compile(
        "\\b" + "\\s+".join((re.escape(part) for part in span.split())) + "\\b", re.IGNORECASE
    )
    return pattern.sub(str(replacement), str(text), count=1)


def _clean_caption(text):
    text = str(text or "")
    text = re.sub("\\s+([,.;:!?])", "\\1", text)
    text = re.sub("([,(])\\s*([,.;:!?])", "\\2", text)
    text = re.sub("\\s+", " ", text).strip(" ,;:")
    text = re.sub("\\s+([.])", "\\1", text)
    if not text or len(WORD_RE.findall(text)) < 3:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _normalize_filter(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


class RouteManifestIterableDataset(DistributedIterableDataset):
    """Grounded phrase/crop generation or full-image caption ranking."""

    def __init__(
        self,
        dataset_name,
        tokenizer,
        data_dir_list,
        manifest_path_list,
        num_used_data,
        transform=None,
        vit_transform=None,
        route_mode="t2i_implicit",
        image_cache_dir=None,
        question_prompt="Describe this image faithfully.",
        use_pseudo_grounding=False,
        require_pseudo_grounding=False,
        local_phrase_key="local_phrase",
        local_crop_ref_key="local_crop_ref",
        local_t2i_prompt_template="{phrase}",
        i2t_objective="ranking_ce",
        i2t_num_negatives=3,
        i2t_negative_seed=20260702,
        global_i2t_negative_sources=None,
        source_filter=None,
        shuffle_lines=True,
        shuffle_seed=42,
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        if route_mode not in {"t2i_implicit", "i2t_global_ranking"}:
            raise ValueError("Only grounded T2I and full-caption I2T are supported")
        self.tokenizer = tokenizer
        self.transform = transform
        self.vit_transform = vit_transform
        self.route_mode = route_mode
        self.question_prompt = str(question_prompt)
        self.use_pseudo_grounding = bool(use_pseudo_grounding)
        self.require_pseudo_grounding = bool(require_pseudo_grounding)
        self.local_phrase_key = local_phrase_key
        self.local_crop_ref_key = local_crop_ref_key
        self.local_t2i_prompt_template = local_t2i_prompt_template
        self.local_phrase_source_filter = set()
        self.local_phrase_max_words = 0
        self.i2t_objective = i2t_objective
        self.i2t_num_negatives = int(i2t_num_negatives)
        self.i2t_negative_seed = int(i2t_negative_seed)
        self.global_i2t_negative_sources = list(
            global_i2t_negative_sources or ["detail_drop", "wrong_detail", "random_caption"]
        )
        if self.route_mode == "i2t_global_ranking":
            if self.use_pseudo_grounding or i2t_objective != "ranking_ce":
                raise ValueError("I2T uses full-image caption ranking plus positive CE")
            if self.i2t_num_negatives != 3 or self.global_i2t_negative_sources != [
                "detail_drop",
                "wrong_detail",
                "random_caption",
            ]:
                raise ValueError("I2T requires the three released negative sources, in order")
        elif not (self.use_pseudo_grounding and self.require_pseudo_grounding):
            raise ValueError("T2I requires grounded phrase/crop supervision")
        self.source_filter = _normalize_filter(source_filter)
        self.shuffle_seed = int(shuffle_seed)
        self.data_status = data_status
        self._tar_handles = {}
        self.data_paths = self.get_data_paths(
            manifest_path_list,
            num_used_data,
            shuffle_lines=shuffle_lines,
            shuffle_seed=shuffle_seed,
        )
        if not self.data_paths:
            raise ValueError("No training records matched the manifest")
        self.global_i2t_caption_pool = self._build_global_i2t_caption_pool()
        self.set_epoch()

    def __iter__(self):
        (items, worker_id) = self.get_data_paths_per_worker()
        start = self.data_status[worker_id] + 1 if self.data_status is not None else 0
        while True:
            for index, (_, _, record) in enumerate(items[start:], start=start):
                try:
                    image = self._load_image(record["image_ref"])
                    text = str(record.get("text", "")).strip()
                    if not text:
                        continue
                    if self.route_mode == "t2i_implicit":
                        sample = self._build_t2i_sample(record, image, text, worker_id, index)
                    else:
                        sample = self._build_global_i2t_ranking_samples(
                            record, image, text, worker_id, index
                        )
                    if isinstance(sample, list):
                        yield from (item for item in sample if item is not None)
                    elif sample is not None:
                        yield sample
                except Exception as exc:
                    print(f"Error processing {record.get('sample_id')}: {exc}", flush=True)
            start = 0

    def _build_t2i_sample(self, record, image, text, worker_id, row_idx):
        if self.transform is None:
            raise ValueError("T2I requires an image transform")
        local = self._local_target(record)
        if local is None:
            return None
        prompt = self._format_local_t2i_prompt(local["phrase"], text)
        image_tensor = self.transform(self._load_image(local["crop_ref"]))
        text_ids = self.tokenizer.encode(prompt)
        (height, width) = image_tensor.shape[1:]
        return {
            "image_tensor_list": [image_tensor],
            "text_ids_list": [text_ids],
            "num_tokens": width * height // self.transform.stride**2 + len(text_ids),
            "sequence_plan": [
                {
                    "type": "text",
                    "enable_cfg": 1,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
                {
                    "type": "vae_image",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            ],
            "data_indexes": self._data_indexes(record, worker_id, row_idx, "t2i_local", local),
        }

    def _negative_global_i2t_answers(self, record, positive_answer, group_id):
        positive_key = str(positive_answer).casefold()
        digest = hashlib.sha1(
            f"{self.i2t_negative_seed}:{group_id}:global".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        (candidates, seen) = ([], {positive_key})
        sources = []
        for source in self.global_i2t_negative_sources:
            if source == "detail_drop":
                values = self._detail_drop_caption_candidates(record, positive_answer, rng)
            elif source == "wrong_detail":
                values = self._wrong_detail_caption_candidates(record, positive_answer, rng)
            else:
                values = [
                    answer
                    for answer in self.global_i2t_caption_pool
                    if answer.casefold() not in seen
                ]
                rng.shuffle(values)
                values = values[: self.i2t_num_negatives]
            rng.shuffle(values)
            sources.append(values)
        while len(candidates) < self.i2t_num_negatives:
            progressed = False
            for values in sources:
                while values:
                    value = str(values.pop(0) or "").strip()
                    if value and value.casefold() not in seen:
                        seen.add(value.casefold())
                        candidates.append(value)
                        progressed = True
                        break
            if not progressed:
                break
        if len(candidates) < self.i2t_num_negatives:
            fallback = [
                answer for answer in self.global_i2t_caption_pool if answer.casefold() not in seen
            ]
            rng.shuffle(fallback)
            for answer in fallback:
                if answer.casefold() not in seen:
                    seen.add(answer.casefold())
                    candidates.append(answer)
                if len(candidates) >= self.i2t_num_negatives:
                    break
        return candidates[: self.i2t_num_negatives]

    def get_data_paths(
        self, manifest_path_list, num_used_data, shuffle_lines=True, shuffle_seed=42
    ):
        items = []
        for manifest_idx, manifest_path in enumerate(manifest_path_list):
            path = Path(manifest_path).expanduser()
            with path.open("r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
            if self.source_filter:
                records = [
                    record for record in records if record.get("source") in self.source_filter
                ]
            if shuffle_lines:
                rng = random.Random(int(shuffle_seed) + manifest_idx)
                rng.shuffle(records)
            limit = None
            if num_used_data and manifest_idx < len(num_used_data):
                limit = int(num_used_data[manifest_idx])
            if limit is not None and limit > 0:
                records = records[:limit]
            for row_idx, record in enumerate(records):
                items.append((manifest_idx, row_idx, record))
        return items

    def _build_global_i2t_ranking_samples(self, record, image, text, worker_id, row_idx):
        question = self.question_prompt
        answer = text
        sample_route_mode = "i2t_global_ranking"
        group_id = self._contrastive_group_id(record, row_idx)
        negatives = self._negative_global_i2t_answers(record, answer, group_id)
        if len(negatives) < self.i2t_num_negatives:
            return None
        candidates = [answer] + negatives[: self.i2t_num_negatives]
        return [
            self._build_i2t_candidate_sample(
                record=record,
                source_image=image,
                question=question,
                answer=candidate_answer,
                worker_id=worker_id,
                row_idx=row_idx,
                sample_route_mode=sample_route_mode,
                local=None,
                extra_indexes={
                    "i2t_objective": self.i2t_objective,
                    "contrastive_group_id": group_id,
                    "contrastive_label": 1 if candidate_idx == 0 else 0,
                    "contrastive_candidate_index": candidate_idx,
                    "contrastive_candidate_count": len(candidates),
                    "global_i2t_negative_sources": ",".join(self.global_i2t_negative_sources),
                },
            )
            for (candidate_idx, candidate_answer) in enumerate(candidates)
        ]

    def _build_i2t_candidate_sample(
        self,
        record,
        source_image,
        question,
        answer,
        worker_id,
        row_idx,
        sample_route_mode,
        local=None,
        extra_indexes=None,
    ):
        vit_image_tensor = self.vit_transform(source_image)
        question_ids = self.tokenizer.encode(question)
        answer_ids = self.tokenizer.encode(answer)
        (height, width) = vit_image_tensor.shape[1:]
        num_tokens = (
            width * height // self.vit_transform.stride**2 + len(question_ids) + len(answer_ids)
        )
        return {
            "image_tensor_list": [vit_image_tensor],
            "text_ids_list": [question_ids, answer_ids],
            "num_tokens": num_tokens,
            "sequence_plan": [
                {
                    "type": "vit_image",
                    "enable_cfg": 0,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
                {
                    "type": "text",
                    "enable_cfg": 0,
                    "loss": 0,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
                {
                    "type": "text",
                    "enable_cfg": 0,
                    "loss": 1,
                    "special_token_loss": 0,
                    "special_token_label": None,
                },
            ],
            "data_indexes": self._data_indexes(
                record, worker_id, row_idx, sample_route_mode, local, extra=extra_indexes
            ),
        }

    def _local_target(self, record):
        if not self.use_pseudo_grounding:
            return None
        phrase = str(record.get(self.local_phrase_key) or "").strip()
        crop_ref = record.get(self.local_crop_ref_key)
        if not phrase or not isinstance(crop_ref, dict):
            return None
        grounding = record.get("pseudo_grounding")
        phrase_source = grounding.get("phrase_source") if isinstance(grounding, dict) else None
        if self.local_phrase_source_filter and phrase_source not in self.local_phrase_source_filter:
            return None
        if (
            self.local_phrase_max_words > 0
            and len(WORD_RE.findall(phrase)) > self.local_phrase_max_words
        ):
            return None
        return {"phrase": phrase, "crop_ref": crop_ref}

    def _build_global_i2t_caption_pool(self):
        answer_pool = []
        seen = set()
        for _, _, record in self.data_paths:
            answer = str(record.get("text", "")).strip()
            if not answer:
                continue
            key = answer.casefold()
            if key in seen:
                continue
            seen.add(key)
            answer_pool.append(answer)
        return answer_pool

    def _contrastive_group_id(self, record, row_idx):
        sample_id = str(record.get("sample_id") or row_idx)
        return f"{self.dataset_name}:{sample_id}:{row_idx}"

    def _wrong_detail_caption_candidates(self, record, positive_answer, rng):
        text = str(positive_answer or "").strip()
        replacements = WRONG_DETAIL_REPLACEMENTS
        terms = self._target_terms(record, ("attribute_terms", "count_terms", "relation_terms"))
        rng.shuffle(terms)
        candidates = []

        def add(value):
            value = _clean_caption(value)
            if value and value.casefold() != text.casefold():
                candidates.append(value)

        for term in terms[:6]:
            key = str(term or "").strip().casefold()
            replacement = replacements.get(key)
            if replacement:
                add(_replace_text_span(text, term, replacement))
        for src, dst in replacements.items():
            if re.search("\\b" + re.escape(src) + "\\b", text, flags=re.IGNORECASE):
                add(_replace_text_span(text, src, dst))
        return candidates

    def _target_terms(self, record, keys):
        targets = record.get("targets")
        if not isinstance(targets, dict):
            return []
        terms = []
        for key in keys:
            values = targets.get(key) or []
            if isinstance(values, list):
                terms.extend((str(value).strip() for value in values if str(value).strip()))
        return terms

    def _detail_drop_caption_candidates(self, record, positive_answer, rng):
        text = str(positive_answer or "").strip()
        candidates = []

        def add(value):
            value = _clean_caption(value)
            if value and value.casefold() != text.casefold():
                candidates.append(value)

        local = self._local_target(record)
        if local is not None:
            add(_drop_text_span(text, local.get("phrase", "")))
        targets = record.get("targets")
        if isinstance(targets, dict):
            detail_terms = []
            for key in ("attribute_terms", "count_terms", "relation_terms"):
                values = targets.get(key) or []
                if isinstance(values, list):
                    detail_terms.extend((str(value) for value in values if str(value).strip()))
            rng.shuffle(detail_terms)
            for term in detail_terms[:4]:
                add(_drop_text_span(text, term))
        object_heads = self._object_head_candidates(record)
        rng.shuffle(object_heads)
        for object_head in object_heads[:4]:
            add(_drop_text_span(text, object_head))
        sentences = [
            sentence.strip() for sentence in re.split("(?<=[.!?])\\s+", text) if sentence.strip()
        ]
        if len(sentences) > 1:
            add(sentences[0])
        words = WORD_RE.findall(text)
        if len(words) >= 12:
            keep = max(8, int(len(words) * 0.6))
            add(" ".join(words[:keep]))
        seen = set()
        unique = []
        for candidate in candidates:
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    def _format_local_t2i_prompt(self, phrase, full_caption):
        return self.local_t2i_prompt_template.format(phrase=phrase, text=full_caption)

    def _object_head_candidates(self, record):
        targets = record.get("targets")
        if not isinstance(targets, dict):
            return []
        object_heads = targets.get("object_heads") or []
        if not isinstance(object_heads, list):
            return []
        candidates = []
        seen = set()
        for object_head in object_heads:
            head = str(object_head or "").strip()
            key = head.casefold()
            if not head or key in seen:
                continue
            seen.add(key)
            candidates.append(head)
        return candidates

    def _data_indexes(self, record, worker_id, row_idx, sample_mode, local=None, extra=None):
        indexes = {
            "data_indexes": row_idx,
            "worker_id": worker_id,
            "dataset_name": self.dataset_name,
            "sample_id": record.get("sample_id", ""),
            "route": record.get("route", ""),
            "route_mode": sample_mode,
        }
        if local is not None:
            indexes["local_phrase"] = local.get("phrase", "")
            indexes["local_bbox"] = record.get("local_bbox")
            grounding = record.get("pseudo_grounding")
            if isinstance(grounding, dict):
                indexes["pseudo_grounding_score"] = grounding.get("score")
                indexes["pseudo_grounding_phrase_source"] = grounding.get("phrase_source")
        if extra:
            indexes.update(extra)
        return indexes

    def _load_image(self, image_ref):
        image_type = image_ref.get("type")
        if image_type in {"local_file", "file"}:
            return pil_img2rgb(Image.open(Path(image_ref["path"]).expanduser()))
        if image_type == "image_crop":
            base = self._load_image(image_ref["image_ref"])
            bbox = image_ref.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"image_crop requires bbox=[x0,y0,x1,y1], got: {bbox}")
            return pil_img2rgb(base.crop(tuple((int(v) for v in bbox))))
        if image_type == "webdataset_tar_member":
            return self._load_tar_member(image_ref)
        raise ValueError(f"Unsupported image_ref type: {image_type}")

    def _load_tar_member(self, image_ref):
        shard_path = str(image_ref["shard_path"])
        member = str(image_ref["member"])
        tar = self._tar_handles.get(shard_path)
        if tar is None:
            tar = tarfile.open(shard_path, "r")
            self._tar_handles[shard_path] = tar
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"{member} not found in {shard_path}")
        return pil_img2rgb(Image.open(io.BytesIO(extracted.read())))
