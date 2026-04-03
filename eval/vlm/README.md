# Understanding Benchmarks -- Data Preparation

This guide covers data preparation for the understanding evaluation benchmarks: **MME**, **MMMU**, **MMBench**, **MM-Vet**, and **MathVista**.

All benchmark data is stored under `data/` at the repository root.

The data preparation follows the InternVL pipeline. For the full reference, see:
https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html

---

## MME

Download the MME benchmark data:

```bash
cd /path/to/umm_codebase
mkdir -p data/mme
cd data/mme
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip
```

Expected directory structure:

```
data/mme/
└── MME_Benchmark_release_version/
    ├── artwork/
    ├── celebrity/
    └── ...
```

Referenced in config as:

```yaml
image_root: data/mme/MME_Benchmark_release_version
```

---

## MMMU

MMMU is auto-downloaded from HuggingFace (`MMMU/MMMU`) during evaluation. No manual download is needed.

- Cached at: `data/MMMU/`

Referenced in config as:

```yaml
root: MMMU/MMMU
cache_dir: data/MMMU
```

---

## MMBench

Download the MMBench TSV files:

```bash
cd /path/to/umm_codebase
mkdir -p data/mmbench
cd data/mmbench
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv
```

Expected directory structure:

```
data/mmbench/
├── mmbench_dev_20230712.tsv
├── mmbench_dev_cn_20231003.tsv
├── mmbench_dev_en_20231003.tsv
├── mmbench_test_cn_20231003.tsv
└── mmbench_test_en_20231003.tsv
```

---

## MM-Vet

Download the MM-Vet benchmark data and question file:

```bash
cd /path/to/umm_codebase
mkdir -p data/mm-vet
cd data/mm-vet
wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
unzip mm-vet.zip
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
```

Expected directory structure:

```
data/mm-vet/
├── mm-vet/
│   └── images/
│       ├── v1_0.png
│       └── ...
└── llava-mm-vet.jsonl
```

Referenced in config as:

```yaml
image_root: data/mm-vet/mm-vet/images
question: data/mm-vet/llava-mm-vet.jsonl
```

---

## MathVista

Download the annotation file:

```bash
cd /path/to/umm_codebase
mkdir -p data/MathVista
cd data/MathVista
wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
```

Images are auto-downloaded from HuggingFace (`AI4Math/MathVista`) during evaluation.

Referenced in config as:

```yaml
gt_file: data/MathVista/annot_testmini.json
cache_dir: data/MathVista
```
