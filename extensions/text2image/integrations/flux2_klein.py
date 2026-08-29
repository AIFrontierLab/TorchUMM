from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="flux2_klein",
    model_id="black-forest-labs/FLUX.2-klein-4B",
    pipeline_class="Flux2KleinPipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 4,
        "guidance_scale": 1.0,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
