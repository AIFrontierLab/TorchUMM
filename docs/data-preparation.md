# Data Preparation

Each benchmark requires specific data. Generation benchmarks (DPG Bench, GenEval, WISE) include their data in the repository. Understanding benchmarks follow the [InternVL data preparation](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) pipeline. All data is stored under `data/` at the repository root.

!!! tip "Cloud users"
    If you are running evaluations via [Modal](cloud.md), datasets are cached in persistent volumes. Use `modal run modal/download.py --dataset <name>` to download datasets to the cloud. You do not need to prepare data locally.

---

=== "Understanding Benchmarks"

    Understanding benchmarks data is prepared following the [InternVL evaluation data preparation](https://internvl.readthedocs.io/en/latest/get_started/eval_data_preparation.html) guide.

    ### MME

    ```bash
    mkdir -p data/mme
    cd data/mme
    wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/MME_Benchmark_release_version.zip
    unzip MME_Benchmark_release_version.zip
    cd -
    ```

    ### MMMU

    MMMU is auto-downloaded from HuggingFace (`MMMU/MMMU`) at evaluation time and cached in `data/MMMU/`. No manual download is needed.

    ### MMBench

    ```bash
    mkdir -p data/mmbench
    cd data/mmbench
    wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv
    wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv
    wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_en_20231003.tsv
    wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_cn_20231003.tsv
    wget https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_test_en_20231003.tsv
    cd -
    ```

    ### MM-Vet

    ```bash
    mkdir -p data/mm-vet
    cd data/mm-vet
    wget https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip
    unzip mm-vet.zip
    wget https://huggingface.co/OpenGVLab/InternVL/raw/main/llava-mm-vet.jsonl
    cd -
    ```

    ### MathVista

    ```bash
    mkdir -p data/MathVista
    cd data/MathVista
    wget https://huggingface.co/datasets/AI4Math/MathVista/raw/main/annot_testmini.json
    cd -
    ```

=== "Generation Benchmarks"

    These benchmarks include their data in the repository. No additional download is needed.

    ### DPG Bench

    Prompts are stored in `eval/generation/dpg_bench/prompts/` (100 prompt files). No download required.

    ### GenEval

    Metadata and prompts are included in `model/geneval/`. No download required.

    ### WISE

    Benchmark data is included in `model/WISE/`. No download required.

=== "Other Benchmarks"

    ### UEval

    UEval data is auto-downloaded from HuggingFace ([primerL/UEval-all](https://huggingface.co/datasets/primerL/UEval-all)) at evaluation time. No manual download is needed for local execution.

    For Modal cloud execution:

    ```bash
    modal run modal/download.py --dataset ueval
    ```

    ### Uni-MMMU

    Uni-MMMU data is hosted on HuggingFace ([Vchitect/Uni-MMMU-Eval](https://huggingface.co/datasets/Vchitect/Uni-MMMU-Eval)). Reference: [Vchitect/Uni-MMMU](https://github.com/Vchitect/Uni-MMMU).

    For Modal cloud execution:

    ```bash
    modal run modal/download.py --dataset uni_mmmu
    ```

    See [eval/generation/uni_mmmu/README.md](../eval/generation/uni_mmmu/README.md) for details.

    ### GEdit-Bench

    GEdit-Bench data is hosted on HuggingFace ([stepfun-ai/GEdit-Bench](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench)). It is auto-downloaded at evaluation time if not pre-downloaded.

    For Modal cloud execution:

    ```bash
    modal run modal/download.py --dataset gedit
    ```

    Scoring uses Qwen2.5-VL-72B-Instruct (same evaluator as WISE).
