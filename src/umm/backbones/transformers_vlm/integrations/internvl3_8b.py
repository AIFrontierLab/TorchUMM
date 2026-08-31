from umm.backbones.transformers_vlm import VLMModelSpec, factories_from_specs


SPEC = VLMModelSpec(
    name="internvl3_8b",
    model_id="OpenGVLab/InternVL3-8B",
    trust_remote_code=True,
    default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
