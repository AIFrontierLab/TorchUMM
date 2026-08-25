from umm.backbones.diffusers_t2i import TextToImageModelSpec, factories_from_specs


SPECS = {
    "sana_1_5": TextToImageModelSpec(
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
    ),
    "lumina_image_2": TextToImageModelSpec(
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
    ),
}

BACKBONES = factories_from_specs(SPECS)
