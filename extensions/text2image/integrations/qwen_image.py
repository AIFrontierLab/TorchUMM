from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="qwen_image",
    model_id="Qwen/Qwen-Image-2512",
    pipeline_class="QwenImagePipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 50,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "negative_prompt": " ",
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
