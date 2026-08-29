from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="lumina_image_2",
    model_id="Alpha-VLLM/Lumina-Image-2.0",
    pipeline_class="Lumina2Pipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "cfg_trunc_ratio": 0.25,
        "cfg_normalization": True,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
