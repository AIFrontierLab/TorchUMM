from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="sana_1_5",
    model_id="Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers",
    pipeline_class="SanaPipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 18,
        "guidance_scale": 5.0,
        "pag_guidance_scale": 2.0,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
