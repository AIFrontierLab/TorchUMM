from __future__ import annotations

from typing import Any

DEFAULT_LEVEL_PROMPTS: dict[str, str] = {
    "l0": (
        "You are a multimodal reasoning model. For this task, you must follow this EXACT reasoning path:\n\n"
        "Understanding -> Answer\n\n"
        "Where:\n"
        "- Understanding = Text Understanding: give exactly one sentence that directly describes the visible input and the task, without inference\n"
        "- Answer = Final Answer: answer directly from perception only\n\n"
        "Output format:\n\n"
        "Understanding:\n"
        "(Exactly one sentence directly describing the visible input and the task)\n\n"
        "Answer:\n"
        "(Direct answer based only on what is explicitly visible or stated)\n\n"
        "Now solve the following query using ONLY this path (Understanding -> Answer):"
    ),
    "l1": (
        "You are a multimodal reasoning model. For this task, you must follow this EXACT reasoning path:\n\n"
        "Understanding -> Reasoning -> Answer\n\n"
        "Where:\n"
        "- Understanding = Text Understanding: give exactly one sentence that directly describes the visible input and the task, without inference\n"
        "- Reasoning = Text Reasoning: give a short textual reasoning chain using only the provided information\n"
        "- Answer = Final Answer: answer directly and concisely\n\n"
        "Output format:\n\n"
        "Understanding:\n"
        "(Exactly one sentence directly describing the visible input and the task)\n\n"
        "Reasoning:\n"
        "(A short textual reasoning chain)\n\n"
        "Answer:\n"
        "(Final answer)\n\n"
        "Now solve the following query using ONLY this path (Understanding -> Reasoning -> Answer):"
    ),
    "l2": (
        "You are a multimodal reasoning model. For this task, you must follow this EXACT reasoning path:\n\n"
        "Understanding -> Reasoning -> Visual -> Reasoning -> Answer\n\n"
        "Where:\n"
        "- Understanding = Text Understanding: give exactly one sentence that directly describes the visible input and the task, without inference\n"
        "- Reasoning = Text Reasoning: give a short textual reasoning chain using only the provided information\n"
        "- Visual = Visual Reasoning: describe one intermediate visual construct or visual step\n"
        "- Answer = Final Answer: answer directly and concisely\n\n"
        "Output format:\n\n"
        "Understanding:\n"
        "(Exactly one sentence directly describing the visible input and the task)\n\n"
        "Reasoning:\n"
        "(A short textual reasoning chain)\n\n"
        "Visual:\n"
        "(One intermediate visual construct or visual step)\n\n"
        "Reasoning:\n"
        "(A short textual reasoning chain that uses the visual step)\n\n"
        "Answer:\n"
        "(Final answer)\n\n"
        "Now solve the following query using ONLY this path (Understanding -> Reasoning -> Visual -> Reasoning -> Answer):"
    ),
    "l3": (
        "You are a multimodal reasoning model. For this task, you must follow this EXACT reasoning path:\n\n"
        "Understanding -> Reasoning -> Hypothesis -> Reasoning -> Answer\n\n"
        "Where:\n"
        "- Understanding = Text Understanding: give exactly one sentence that directly describes the visible input and the task, without inference\n"
        "- Reasoning = Text Reasoning: give a short textual reasoning chain using only the provided information\n"
        "- Hypothesis = Visual Hypothesis: describe candidate visual states or hypotheses before answering\n"
        "- Answer = Final Answer: answer directly and concisely\n\n"
        "Output format:\n\n"
        "Understanding:\n"
        "(Exactly one sentence directly describing the visible input and the task)\n\n"
        "Reasoning:\n"
        "(A short textual reasoning chain)\n\n"
        "Hypothesis:\n"
        "(Candidate visual states or hypotheses)\n\n"
        "Reasoning:\n"
        "(A short textual reasoning chain that uses the visual hypotheses)\n\n"
        "Answer:\n"
        "(Final answer)\n\n"
        "Now solve the following query using ONLY this path (Understanding -> Reasoning -> Hypothesis -> Reasoning -> Answer):"
    ),
}


def canonicalize_level(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in DEFAULT_LEVEL_PROMPTS:
        return text
    if text.startswith("l") and len(text) >= 2 and text[1].isdigit():
        return f"l{text[1]}"
    return None


def infer_reasoning_level(row: dict[str, Any]) -> str | None:
    for key in ("reasoning_level", "path_level", "level"):
        level = canonicalize_level(row.get(key))
        if level is not None:
            return level

    sample_id = str(row.get("id") or "").lower()
    for level in DEFAULT_LEVEL_PROMPTS:
        if sample_id.startswith(f"{level}_") or f"_{level}_" in sample_id:
            return level
    return None


def _wrap_with_think(prompt: str) -> str:
    return (
        prompt
        + "\n\nWhen you respond, put all reasoning sections inside <think>...</think> and place the final Answer section after </think>."
    )


def resolve_level_prompt(row: dict[str, Any], think_mode: bool = False) -> tuple[str | None, str | None]:
    level = infer_reasoning_level(row)
    if level is None:
        return None, None
    prompt = DEFAULT_LEVEL_PROMPTS[level]
    if think_mode:
        prompt = _wrap_with_think(prompt)
    return level, prompt
