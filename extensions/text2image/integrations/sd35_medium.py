from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="sd35_medium",
    model_id="stabilityai/stable-diffusion-3.5-medium",
    pipeline_class="StableDiffusion3Pipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 40,
        "guidance_scale": 4.5,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
