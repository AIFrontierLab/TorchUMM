from __future__ import annotations

from pathlib import Path
from typing import Any


def install_lora_adapter_for_bagel(model: Any, adapter_dir: str | Path) -> Any:
    """Inject a PEFT LoRA adapter into BAGEL's Qwen language model for UniPath eval."""
    adapter_path = Path(adapter_dir).expanduser()
    if not adapter_path.exists():
        raise FileNotFoundError(f"UniPath LoRA adapter not found: {adapter_path}")

    from peft import PeftModel

    for param in model.parameters():
        param.requires_grad = False
    if not hasattr(model.language_model, "prepare_inputs_for_generation"):

        def _prepare_inputs_for_generation(input_ids=None, **kwargs):
            payload = dict(kwargs)
            if input_ids is not None:
                payload["input_ids"] = input_ids
            return payload

        model.language_model.prepare_inputs_for_generation = _prepare_inputs_for_generation

    peft_model = PeftModel.from_pretrained(model.language_model, str(adapter_path), is_trainable=False)
    wrapped_qwen = peft_model.base_model.model
    inner_transformer = getattr(wrapped_qwen, "model", None)
    if inner_transformer is not None:
        if not hasattr(wrapped_qwen, "embed_tokens") and hasattr(inner_transformer, "embed_tokens"):
            wrapped_qwen.embed_tokens = inner_transformer.embed_tokens
        if not hasattr(wrapped_qwen, "norm") and hasattr(inner_transformer, "norm"):
            wrapped_qwen.norm = inner_transformer.norm
    model.language_model = wrapped_qwen
    model.eval()
    print(f"[unipath] loaded BAGEL LoRA adapter: {adapter_path}", flush=True)
    return model
