from __future__ import annotations

from umm.backbones.transformers_vlm import VLMModelSpec, factories_from_specs


SPECS = {
    "qwen2_5_vl_7b": VLMModelSpec(
        name="qwen2_5_vl_7b",
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "internvl3_8b": VLMModelSpec(
        name="internvl3_8b",
        model_id="OpenGVLab/InternVL3-8B",
        trust_remote_code=True,
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "llava_onevision_7b": VLMModelSpec(
        name="llava_onevision_7b",
        model_id="llava-hf/llava-onevision-qwen2-7b-ov-hf",
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "idefics3_8b": VLMModelSpec(
        name="idefics3_8b",
        model_id="HuggingFaceM4/Idefics3-8B-Llama3",
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "smolvlm_2b": VLMModelSpec(
        name="smolvlm_2b",
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "phi3_5_vision": VLMModelSpec(
        name="phi3_5_vision",
        model_id="microsoft/Phi-3.5-vision-instruct",
        trust_remote_code=True,
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "gemma3_4b": VLMModelSpec(
        name="gemma3_4b",
        model_id="google/gemma-3-4b-it",
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "molmo_7b": VLMModelSpec(
        name="molmo_7b",
        model_id="allenai/Molmo-7B-D-0924",
        trust_remote_code=True,
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "minicpm_v_2_6": VLMModelSpec(
        name="minicpm_v_2_6",
        model_id="openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        default_generation_cfg={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
    ),
    "paligemma2_3b": VLMModelSpec(
        name="paligemma2_3b",
        model_id="google/paligemma2-3b-mix-224",
        prompt_style="paligemma",
        default_generation_cfg={"max_new_tokens": 128, "do_sample": False, "return_full_text": False},
    ),
}


BACKBONES = factories_from_specs(SPECS)
