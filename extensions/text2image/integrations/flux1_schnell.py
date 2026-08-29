from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="flux1_schnell",
    model_id="black-forest-labs/FLUX.1-schnell",
    pipeline_class="FluxPipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
