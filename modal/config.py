# Modal infrastructure constants for umm_codebase.
# All Modal-related files import from here.

# ---------------------------------------------------------------------------
# Volume names (persistent distributed filesystems, created once)
# ---------------------------------------------------------------------------
MODEL_VOLUME_NAME = "umm-model-cache"       # model weights from HuggingFace
POST_TRAIN_MODEL_VOLUME_NAME = "umm-post-train-model-cache"  # post_train model weights (separate volume)
DATASET_VOLUME_NAME = "umm-datasets-cache"  # datasets from HuggingFace
CHECKPOINT_VOLUME_NAME = "umm-checkpoints"  # training outputs & checkpoints
OUTPUT_VOLUME_NAME = "umm-outputs"           # inference & evaluation outputs
OUTPUT_APR4_VOLUME_NAME = "umm-output-apr4"  # new output volume (apr4, avoids inode limit)
CODEBASE_VOLUME_NAME = "umm-codebase"       # project source code (synced manually)

# ---------------------------------------------------------------------------
# Remote paths inside Modal containers (where Volumes are mounted)
# ---------------------------------------------------------------------------
MODEL_CACHE_PATH = "/model_cache"       # model weights Volume mount point
POST_TRAIN_MODEL_CACHE_PATH = "/post_train_model_cache"  # post_train Volume mount point
DATASET_CACHE_PATH = "/datasets"        # datasets Volume mount point
CHECKPOINT_PATH = "/checkpoints"        # checkpoints Volume mount point
OUTPUT_PATH = "/outputs"                # inference & eval outputs mount point
OUTPUT_APR4_PATH = "/output-apr4"       # new output volume mount point
WORKSPACE_PATH = "/workspace"           # codebase mount point

# ---------------------------------------------------------------------------
# HuggingFace model IDs → local path inside MODEL_CACHE_PATH
#
# Structure: { "repo_name": [ (hf_repo_id, local_subdir), ... ] }
# Each repo can have multiple models to download.
# ---------------------------------------------------------------------------
HF_MODELS = {
    "bagel": [
        ("ByteDance-Seed/BAGEL-7B-MoT", "bagel/BAGEL-7B-MoT"),
    ],
    "blip3o": [
        ("BLIP3o/BLIP3o-NEXT-SFT-3B",          "blip3o/BLIP3o-NEXT-SFT-3B"),
    ],
    "deepgen": [
        ("deepgenteam/DeepGen-1.0-diffusers", "deepgen/DeepGen-1.0-diffusers"),
    ],
    "emu3": [
        ("BAAI/Emu3-Chat",              "emu3/Emu3-Chat"),
        ("BAAI/Emu3-Gen",               "emu3/Emu3-Gen"),
        ("BAAI/Emu3-VisionTokenizer",   "emu3/Emu3-VisionTokenizer"),
    ],
    "emu3_5": [
        ("BAAI/Emu3.5",                  "emu3_5/Emu3.5"),
        ("BAAI/Emu3.5-Image",            "emu3_5/Emu3.5-Image"),
        ("BAAI/Emu3.5-VisionTokenizer",  "emu3_5/Emu3.5-VisionTokenizer"),
    ],
    "janus_pro": [
        ("deepseek-ai/Janus-Pro-7B", "janus_pro/Janus-Pro-7B"),
        ("deepseek-ai/Janus-1.3B", "janus_pro/Janus-1.3B"),
    ],
    "janus_flow": [
        ("deepseek-ai/JanusFlow-1.3B", "janus_flow/JanusFlow-1.3B"),
        ("stabilityai/sdxl-vae", "janus_flow/sdxl-vae"),
    ],
    "mmada": [
        ("Gen-Verse/MMaDA-8B-Base", "mmada/MMaDA-8B-Base"),
        ("Gen-Verse/MMaDA-8B-MixCoT", "mmada/MMaDA-8B-MixCoT"),
        ("showlab/magvitv2", "mmada/magvitv2"),
    ],
    "omnigen2": [
        ("OmniGen2/OmniGen2", "omnigen2/OmniGen2"),
    ],
    "ovis_u1": [
        ("AIDC-AI/Ovis-U1-3B", "ovis_u1/Ovis-U1-3B"),
    ],
    "show_o": [
        ("showlab/show-o-w-clip-vit-512x512", "show_o/show-o-w-clip-vit-512x512"),
        ("showlab/magvitv2", "show_o/magvitv2"),
    ],
    "show_o2": [
        ("showlab/show-o2-7B", "show_o2/show-o2-7B"),
        ("showlab/show-o2-1.5B", "show_o2/show-o2-1.5B"),
    ],
    "tokenflow": [
        ("ByteVisionLab/TokenFlow-t2i", "tokenflow/TokenFlow-t2i"),
        ("ByteVisionLab/TokenFlow", "tokenflow/TokenFlow"),  # contains tokenflow_clipb_32k_enhanced.pt
    ],
    "evaluator": [
        ("Qwen/Qwen2.5-VL-72B-Instruct", "evaluator/Qwen2.5-VL-72B-Instruct"),
        ("Qwen/Qwen3-32B",               "evaluator/Qwen3-32B"),
        ("Qwen/Qwen3-32B-AWQ",           "evaluator/Qwen3-32B-AWQ"),
    ],
    # Unified-Bench scoring models (CLIP / DINOv2). LongCLIP is a single .pt file
    # (see HF_SINGLE_FILES). DINOv3 is split into its own entry because the repo
    # is gated by Meta and requires per-user access approval on HuggingFace; run
    # `--model unified_bench_dinov3` separately once your HF account is approved.
    "unified_bench_scorers": [
        ("openai/clip-vit-large-patch14",           "unified_bench/clip-vit-large-patch14"),
        ("facebook/dinov2-large",                   "unified_bench/dinov2-large"),
    ],
    "unified_bench_dinov3": [
        ("facebook/dinov3-vitl16-pretrain-lvd1689m", "unified_bench/dinov3-vitl16"),
    ],
    "post_train": [
        ("aifronter/post_train", "post_train"),
    ],  # NOTE: post_train uses a separate volume (POST_TRAIN_MODEL_VOLUME_NAME)
}

# ---------------------------------------------------------------------------
# Single-file downloads from HuggingFace (not full repos)
#
# Structure: { "repo_name": [ (hf_repo_id, filename, local_path), ... ] }
# local_path is relative to MODEL_CACHE_PATH.
# ---------------------------------------------------------------------------
HF_SINGLE_FILES = {
    "show_o2": [
        ("Wan-AI/Wan2.1-T2V-14B", "Wan2.1_VAE.pth", "show_o2/Wan2.1_VAE.pth"),
    ],
    "unified_bench_scorers": [
        ("BeichenZhang/LongCLIP-L", "longclip-L.pt", "unified_bench/longclip-L.pt"),
    ],
}

# ---------------------------------------------------------------------------
# Direct URL downloads (non-HuggingFace files) → local path inside MODEL_CACHE_PATH
#
# Structure: { "repo_name": [ (url, local_path), ... ] }
# local_path is relative to MODEL_CACHE_PATH.
# ---------------------------------------------------------------------------
URL_DOWNLOADS = {
    "geneval": [
        (
            "https://download.openmmlab.com/mmdetection/v2.0/mask2former/"
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/"
            "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth",
            "evaluator/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
        ),
    ],
}

# ---------------------------------------------------------------------------
# HuggingFace datasets → local path inside DATASET_CACHE_PATH
# ---------------------------------------------------------------------------
HF_DATASETS = {
    "ueval": [
        ("zlab-princeton/UEval", "ueval/UEval"),
    ],
    "uni_mmmu": [
        ("Vchitect/Uni-MMMU-Eval", "uni_mmmu/Uni-MMMU-Eval"),
    ],
    "gedit": [
        ("stepfun-ai/GEdit-Bench", "gedit/GEdit-Bench"),
    ],
    "mathvista": [
        ("AI4Math/MathVista", "mathvista"),
    ],
    # imgedit: special handling — downloads only Benchmark.tar, extracts & renames
    #          see download.py download_dataset() for the extraction logic
    "imgedit": [
        ("sysuyy/ImgEdit", "imgedit"),
    ],
    # unified_bench: special handling — downloads Image.zip from GitHub release
    #                (not HuggingFace), extracts to /datasets/unified_bench/Image/
    #                See _download_unified_bench() in download.py.
    "unified_bench": [],  # marker only; actual URL handled in download.py
    # geneval: data is bundled with the repo code, no HF dataset needed
    # wise:    data is bundled with the repo code, no HF dataset needed
}

# ---------------------------------------------------------------------------
# Dataset URL downloads (non-HuggingFace zip/tar archives).
# Structure: { "name": (url, local_subdir_under_DATASET_CACHE_PATH) }
# ---------------------------------------------------------------------------
DATASET_URL_ARCHIVES = {
    "unified_bench": (
        "https://github.com/PKU-YuanGroup/UAE/releases/download/v1.0/Image.zip",
        "unified_bench",
    ),
}

# ---------------------------------------------------------------------------
# VLU benchmark datasets — downloaded and prepared via download_vlu_dataset()
#
# "cache" format: pre-cache via datasets.load_dataset() (auto-download on eval)
# "extract" format: download from HF and extract to raw files expected by CLI
# ---------------------------------------------------------------------------
VLU_DATASETS = {
    "mme": {
        "hf_repo": "lmms-lab/MME",
        "local_dir": "mme",
        "format": "extract",
    },
    "mmmu": {
        "hf_repo": "MMMU/MMMU",
        "local_dir": "mmmu",
        "format": "cache",
    },
    "mmbench": {
        "hf_repo": "lmms-lab/MMBench",
        "local_dir": "mmbench",
        "format": "extract",
    },
    "mmvet": {
        "hf_repo": "lmms-lab/MMVet",
        "local_dir": "mmvet",
        "format": "extract",
    },
    "mathvista": {
        "hf_repo": "AI4Math/MathVista",
        "local_dir": "mathvista",
        "format": "cache",
    },
}
