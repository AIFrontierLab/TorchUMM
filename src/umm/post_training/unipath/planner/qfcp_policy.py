from __future__ import annotations

import math
import re

from umm.post_training.unipath.planner.models import PATHS


QFCP_POLICY = {
    "binary_choice": "anchor:l3:deep:m0.0",
    "binary_explicit": "post:binary_exp:drm0.28",
    "binary_open": "post:binary_open:drm0.42",
    "choice": "argmax:shallow:t1.0",
    "default": "post:shallow:drm0.16",
    "inline_binary_choice": "argmax:none:t1.0",
    "inline_choice": "post:direct:drm0.8",
    "inline_math": "anchor:l3:shallow:m1.5",
    "math": "argmax:mid:t1.8",
    "structured_choice": "anchor:l3:none:m1.5",
    "structured_math": "argmax:l3:t1.0",
    "unstructured_default": "argmax:l0:t1.0",
}

QFCP_BIASES = {
    "none": [0.0, 0.0, 0.0, 0.0, 0.0],
    "direct": [0.7, 0.0, 0.0, -0.2, -0.3],
    "shallow": [0.4, 0.2, 0.0, -0.1, -0.2],
    "l0": [0.0, 0.5, 0.0, -0.1, -0.2],
    "mid": [0.0, 0.1, 0.4, 0.2, 0.0],
    "deep": [-0.2, 0.0, 0.2, 0.5, 0.4],
    "l3": [-0.2, 0.0, 0.1, 0.3, 0.7],
    "binary_exp": [0.2, 0.1, 0.0, -0.1, -0.2],
    "binary_open": [0.1, 0.25, 0.0, 0.1, -0.1],
}

QFCP_REFINED_CLEAN_POLICY = {
    **QFCP_POLICY,
    "binary_choice": "canchor:direct:direct:t1.8:m0.0",
    "structured_choice": "canchor:l0:l0:t1.8:m0.5",
    "structured_math_arithmetic": "fixed:current",
    "structured_math_chart": "canchor:l0:geom_l3:t1.0:m0.75",
    "structured_math_count": "canchor:l3:geom_l3:t2.4:m1.5",
    "structured_math_default": "fixed:current",
    "structured_math_equation": "canchor:current:l0:t1.0:m0.25",
    "structured_math_geometry": "canchor:l3:none:t2.4:m0.0",
    "structured_math_integer": "canchor:l0:current:t1.8:m0.5",
}

QFCP_REFINED_PRAGMATIC_POLICY = {
    **QFCP_REFINED_CLEAN_POLICY,
    "structured_choice_unit": "fixed:direct",
    "structured_math_count_number_of": "fixed:l1",
    "structured_math_geometry_area_volume": "fixed:l3",
}

QFCP_REFINED_PRAGMATIC_INLINE_DIRECT_POLICY = {
    **QFCP_REFINED_PRAGMATIC_POLICY,
    "inline_math": "post:direct:drm0.8",
}

QFCP_REFINED_PRAGMATIC_COMPACT_POLICY = {
    **QFCP_REFINED_PRAGMATIC_INLINE_DIRECT_POLICY,
    "inline_binary_choice": "post:direct:drm0.8",
    "inline_math": "post:direct:drm0.8",
    "default": "fixed:current",
}

QFCP_REFINED_PRAGMATIC_SAFE_POLICY = {
    **QFCP_REFINED_PRAGMATIC_COMPACT_POLICY,
    "choice": "post:direct:drm0.8",
    "binary_choice": "canchor:direct:direct:t1.8:m0.0",
}

QFCP_REFINED_CANDIDATE_BIASES = {
    "none": {},
    "current": {"current": 0.4},
    "direct": {"direct": 0.6, "current": 0.0, "l3": -0.1},
    "l0": {"l0": 0.55, "l1": 0.1, "l3": -0.2},
    "geom_l3": {"l3": 0.65, "l1": 0.15, "current": 0.0},
}


def build_mmmu_query(question: str, question_type: str, options: list[str]) -> str:
    del question_type
    if options:
        letters = "ABCDEFGHIJKLM"
        choice_text = "\n".join(f"{letters[idx]}. {str(option).strip()}" for idx, option in enumerate(options))
        suffix = "Answer with the option's letter from the given choices directly."
        return f"{question.strip()}\n{choice_text}\n{suffix}".strip()
    suffix = "Answer the question using a single word or phrase."
    return f"{question.strip()}\n{suffix}".strip()


def qfcp_is_yesno_query(query: str) -> bool:
    compact = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if "yes or no" in compact or re.search(r"\banswer (yes|no)\b", compact):
        return True
    if re.search(r"\b(yes|no)\b", compact):
        return True
    return bool(
        re.match(
            r"^(is|are|was|were|do|does|did|can|could|would|will|should|has|have|had)\b",
            compact,
        )
    )


def qfcp_is_math_query(query: str) -> bool:
    text = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if re.search(
        r"\b(integer answer|floating-point|numeric answer|number with|percentage|calculate|"
        r"equation|solve|geometry|algebra|math|value of|how many|sum of|subtract|"
        r"add |multiply|divide|find x|length of|angle|area|volume|shown in the figure)\b",
        text,
    ):
        return True
    return bool(re.search(r"\b(if|given)\b.*\b(=|\d+\.\d+|\d+)\b", text))


def qfcp_query_regime(query: str) -> str:
    text = (query or "").lower()
    has_choices = "options:" in text or "choices:" in text or "option's letter" in text or "correct option letter" in text
    if qfcp_is_yesno_query(query):
        if has_choices:
            return "binary_choice"
        if "yes or no" in text or re.search(r"\banswer yes or no\b", text):
            return "binary_explicit"
        return "binary_open"
    if qfcp_is_math_query(query):
        return "math"
    if has_choices:
        return "choice"
    return "default"


def qfcp_is_structured_answer_instruction(query: str) -> bool:
    text = re.sub(r"\s+", " ", (query or "").lower()).strip()
    return (
        text.startswith("hint: please answer")
        or "provide the final value" in text
        or ("provide the correct option letter" in text and "question:" in text)
    )


def qfcp_is_inline_options_query(query: str) -> bool:
    text = re.sub(r"\s+", " ", (query or "").lower()).strip()
    return "options:" in text and bool(re.search(r"\b[a-h]\s*:", text))


def qfcp_is_unstructured_open_query(query: str) -> bool:
    text = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if not text:
        return False
    if qfcp_is_structured_answer_instruction(query) or qfcp_is_inline_options_query(query):
        return False
    if "options:" in text or "choices:" in text or "correct option letter" in text or "option's letter" in text:
        return False
    if qfcp_is_yesno_query(query):
        if "yes or no" in text or re.search(r"\banswer yes or no\b", text):
            return False
        if re.match(r"^(is|are|was|were|do|does|did|can|could|would|will|should|has|have|had)\b", text):
            return False
    return True


def qfcp_query_bucket(query: str) -> str:
    regime = qfcp_query_regime(query)
    structured = qfcp_is_structured_answer_instruction(query)
    inline = qfcp_is_inline_options_query(query)
    unstructured = qfcp_is_unstructured_open_query(query)
    if structured and regime == "math":
        return "structured_math"
    if structured and regime == "choice":
        return "structured_choice"
    if inline and regime == "choice":
        return "inline_choice"
    if inline and regime == "math":
        return "inline_math"
    if inline and regime == "binary_choice":
        return "inline_binary_choice"
    if unstructured and regime == "default":
        return "unstructured_default"
    return regime


def qfcp_refined_query_bucket(query: str) -> str:
    bucket = qfcp_query_bucket(query)
    text = re.sub(r"\s+", " ", (query or "").lower()).strip()
    if bucket == "structured_choice" and re.search(
        r"\bunit:|\(unit:| kg| cm| mm| m\b| g\b| liters?| dollars?|\$|°",
        text,
    ):
        return "structured_choice_unit"
    if bucket != "structured_math":
        return bucket
    if re.search(r"\b(table|graph|chart|plot|bar graph|line graph|scatter|histogram)\b", text):
        return "structured_math_chart"
    if re.search(
        r"\b(angle|m\\angle|triangle|square|rectangle|circle|radius|diameter|perimeter|"
        r"area|volume|length|distance|geometry|coordinate|parallel|line segment|slope|degree)\b",
        text,
    ):
        if re.search(r"\b(area|perimeter|volume)\b", text):
            return "structured_math_geometry_area_volume"
        return "structured_math_geometry"
    if re.search(
        r"\b(equation|solve|find x|value of x|system of|function|expression|formula|"
        r"integral|derivative)\b|[a-z]\s*=",
        text,
    ):
        return "structured_math_equation"
    if re.search(r"\bnumber of\b", text):
        return "structured_math_count_number_of"
    if re.search(r"\b(how many|number of|count|total number)\b", text):
        return "structured_math_count"
    if re.search(r"\binteger answer\b", text):
        return "structured_math_integer"
    if re.search(
        r"\b(percent|percentage|ratio|rate|average|mean|sum|difference|subtract|add|"
        r"plus|minus|multiply|divide|product|total)\b",
        text,
    ):
        return "structured_math_arithmetic"
    return "structured_math_default"


def qfcp_refined_pragmatic_safe_query_bucket(query: str) -> str:
    if qfcp_is_inline_options_query(query):
        text = (query or "").lower()
        return "binary_choice" if "yes" in text and "no" in text else "choice"
    bucket = qfcp_refined_query_bucket(query)
    if bucket == "structured_math_count_number_of":
        return "structured_math_count"
    if bucket == "default":
        return "structured_math_default"
    return bucket


def prob_to_logit(x: float) -> float:
    x = min(max(float(x), 1e-6), 1.0 - 1e-6)
    return math.log(x / (1.0 - x))


def qfcp_sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(float(value), 30.0), -30.0)))


def qfcp_argmax(planner_scores: dict[str, float], bias_name: str, temperature: float) -> str:
    bias = QFCP_BIASES[bias_name]
    return max(PATHS, key=lambda path: prob_to_logit(planner_scores[path]) / temperature + float(bias[PATHS.index(path)]))


def qfcp_anchor(planner_scores: dict[str, float], anchor: str, bias_name: str, margin: float) -> str:
    bias = QFCP_BIASES[bias_name]
    values = {path: prob_to_logit(planner_scores[path]) / 1.8 + float(bias[PATHS.index(path)]) for path in PATHS}
    best = max(PATHS, key=lambda path: values[path])
    return best if values[best] >= values[anchor] + float(margin) else anchor


def qfcp_post(planner_scores: dict[str, float], bias_name: str, direct_rescue_margin: float) -> str:
    bias = QFCP_BIASES[bias_name]
    scores = {
        path: qfcp_sigmoid(prob_to_logit(planner_scores[path]) / 1.8 + float(bias[PATHS.index(path)]))
        for path in PATHS
    }
    if scores["direct"] >= max(scores.values()) - float(direct_rescue_margin):
        return "direct"
    penalties = dict(zip(PATHS, [0.0, 0.0, 0.05, 0.10, 0.18]))
    adjusted = {path: float(scores[path]) - penalties[path] for path in PATHS}
    margins = [0.0, 0.0, 0.25, 0.35]
    selected = "direct"
    if adjusted["l0"] >= adjusted["direct"] + margins[0]:
        selected = "l0"
    if adjusted["l1"] >= adjusted["l0"] + margins[1] and adjusted["l1"] >= adjusted[selected]:
        selected = "l1"
    if adjusted["l2"] >= adjusted["l1"] + margins[2] and adjusted["l2"] >= adjusted[selected]:
        selected = "l2"
    if adjusted["l3"] >= adjusted["l2"] + margins[3] and adjusted["l3"] >= adjusted[selected]:
        selected = "l3"
    return selected


def qfcp_choose_selector(planner_scores: dict[str, float], selector: str) -> str:
    parts = selector.split(":")
    if parts[0] == "argmax":
        return qfcp_argmax(planner_scores, parts[1], float(parts[2][1:]))
    if parts[0] == "anchor":
        return qfcp_anchor(planner_scores, parts[1], parts[2], float(parts[3][1:]))
    if parts[0] == "post":
        return qfcp_post(planner_scores, parts[1], float(parts[2][3:]))
    raise ValueError(f"Unsupported QFCP selector: {selector}")


def choose_qfcp_path(planner_scores: dict[str, float], query: str) -> tuple[str, str, str]:
    bucket = qfcp_query_bucket(query)
    selector = QFCP_POLICY.get(bucket, QFCP_POLICY["default"])
    return qfcp_choose_selector(planner_scores, selector), bucket, selector


def qfcp_refined_candidate_score(
    planner_scores: dict[str, float],
    current_path: str,
    candidate: str,
    bias: dict[str, float],
    temperature: float,
) -> float:
    path = current_path if candidate == "current" else candidate
    return prob_to_logit(planner_scores[path]) / temperature + float(bias.get(candidate, 0.0))


def qfcp_refined_anchor(
    planner_scores: dict[str, float],
    current_path: str,
    anchor: str,
    bias_name: str,
    temperature: float,
    margin: float,
) -> str:
    candidates = ["current", "direct", "l0", "l1", "l3"]
    bias = QFCP_REFINED_CANDIDATE_BIASES[bias_name]
    values = {
        candidate: qfcp_refined_candidate_score(planner_scores, current_path, candidate, bias, temperature)
        for candidate in candidates
    }
    best = max(candidates, key=lambda candidate: values[candidate])
    chosen = best if values[best] >= values[anchor] + margin else anchor
    return current_path if chosen == "current" else chosen


def choose_qfcp_refined_pragmatic_safe_path(planner_scores: dict[str, float], query: str) -> tuple[str, str, str]:
    current_path, _, _ = choose_qfcp_path(planner_scores, query)
    bucket = qfcp_refined_pragmatic_safe_query_bucket(query)
    selector = QFCP_REFINED_PRAGMATIC_SAFE_POLICY.get(
        bucket,
        QFCP_REFINED_PRAGMATIC_COMPACT_POLICY.get(
            bucket,
            QFCP_REFINED_PRAGMATIC_INLINE_DIRECT_POLICY.get(
                bucket,
                QFCP_REFINED_PRAGMATIC_POLICY.get(bucket, QFCP_REFINED_CLEAN_POLICY.get(bucket, QFCP_POLICY["default"])),
            ),
        ),
    )
    if selector == "fixed:current":
        return current_path, bucket, selector
    if selector.startswith("fixed:"):
        return selector.split(":", 1)[1], bucket, selector
    parts = selector.split(":")
    if parts[0] == "canchor":
        return (
            qfcp_refined_anchor(
                planner_scores,
                current_path,
                parts[1],
                parts[2],
                float(parts[3][1:]),
                float(parts[4][1:]),
            ),
            bucket,
            selector,
        )
    return qfcp_choose_selector(planner_scores, selector), bucket, selector
