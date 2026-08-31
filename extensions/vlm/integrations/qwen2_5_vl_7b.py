from extensions.vlm import VLMModelSpec, factories_from_specs


SPEC = VLMModelSpec(
    name="qwen2_5_vl_7b",
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
