from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="z_image",
    model_id="Tongyi-MAI/Z-Image-Turbo",
    pipeline_class="ZImagePipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 8,
        "guidance_scale": 0.0,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
