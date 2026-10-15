"""Canonical benchmark input and scoring protocol constants."""

from __future__ import annotations


VLM_EVAL_VIT_TRANSFORM = {
    "max_image_size": 980,
    "min_image_size": 378,
    "image_stride": 14,
    "max_pixels": 2_007_040,
}

TORCHUMM_REFERENCE_VIT_TRANSFORM = {
    "max_image_size": 980,
    "min_image_size": 224,
    "image_stride": 14,
    "max_pixels": 14 * 14 * 9 * 1024,
}

MMBENCH_ANSWER_INSTRUCTIONS = {
    "en": "Answer with the option's letter from the given choices directly.",
    "cn": "请直接回答选项字母。",
}
