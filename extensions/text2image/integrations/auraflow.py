from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="auraflow",
    model_id="fal/AuraFlow-v0.3",
    pipeline_class="AuraFlowPipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 3.5,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
