from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="pixart_sigma",
    model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    pipeline_class="PixArtSigmaPipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 20,
        "guidance_scale": 4.5,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
