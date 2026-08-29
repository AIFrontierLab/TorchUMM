from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPEC = TextToImageModelSpec(
    name="hidream_i1",
    model_id="HiDream-ai/HiDream-I1-Dev",
    pipeline_class="HiDreamImagePipeline",
    default_generation_cfg={
        "height": 1024,
        "width": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 0.0,
    },
)

BACKBONES = factories_from_specs({SPEC.name: SPEC})
