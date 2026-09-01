from extensions.text2image import TextToImageModelSpec, factories_from_specs


SPECS = {
    "cogview4": TextToImageModelSpec(
        name="cogview4",
        model_id="THUDM/CogView4-6B",
        pipeline_class="CogView4Pipeline",
        default_generation_cfg={"height": 1024, "width": 1024, "num_inference_steps": 50, "guidance_scale": 3.5},
    ),
    "kolors": TextToImageModelSpec(
        name="kolors",
        model_id="Kwai-Kolors/Kolors-diffusers",
        pipeline_class="DiffusionPipeline",
        default_generation_cfg={"height": 1024, "width": 1024, "num_inference_steps": 50, "guidance_scale": 5.0},
        default_load_cfg={"trust_remote_code": True},
    ),
}

BACKBONES = factories_from_specs(SPECS)
