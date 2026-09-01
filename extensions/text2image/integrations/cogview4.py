from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="cogview4",
    model_id="THUDM/CogView4-6B",
    pipeline_class="CogView4Pipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
