# LatentUMM Two-Stage Training

LatentUMM is wired into the TorchUMM training CLI.

Run commands from the TorchUMM repository root.

## Stage 1

Stage 1 trains the shared latent alignment model from paired text/image embeddings:

```bash
export LATENTUMM_PROMPTS_PATH=/path/to/dataset/t2i/prompts.json
export LATENTUMM_IMAGE_ROOT=/path/to/dataset/t2i
export LATENTUMM_EMBEDDING_ROOT=/path/to/dataset_embedding/t2i
export LATENTUMM_STAGE1_OUTPUT_DIR=/path/to/outputs/latentumm/stage1_alignment

PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/latentumm_stage1.yaml
```

The Stage 1 checkpoint is written to:

```text
<output_dir>/stage1_shared_latent_model.pt
```

## Stage 2

Stage 2 runs BAGEL fine-tuning with LatentUMM auxiliary losses using the Stage 1 checkpoint:

```bash
export LATENTUMM_BAGEL_MODEL_PATH=/path/to/BAGEL-7B-MoT
export LATENTUMM_STAGE1_CKPT=/path/to/outputs/latentumm/stage1_alignment/stage1_shared_latent_model.pt
export LATENTUMM_EMBEDDING_ROOT=/path/to/dataset_embedding/t2i
export LATENTUMM_STAGE2_OUTPUT_DIR=/path/to/outputs/latentumm/stage2_aux

PYTHONPATH=src python -m umm.cli.main train --config configs/posttrain/latentumm_stage2.yaml
```

Training checkpoints are written under:

```text
<checkpoint_dir>
```

## Citation

```bibtex
@article{luo2026latentumm,
  title={LatentUMM: Dual Latent Alignment for Unified Multimodal Models},
  author={Luo, Yinyi and Wang, Wenwen and Bai, Hayes and Savvides, Marios and Wang, Jindong},
  journal={arXiv preprint arXiv:2605.17766},
  year={2026}
}
```
```bibtex
@article{luo2026torchumm,
  title={TorchUMM: A Unified Multimodal Model Codebase for Evaluation, Analysis, and Post-training},
  author={Luo, Yinyi and Wang, Wenwen and Bai, Hayes and Zhu, Hongyu and Chen, Hao and He, Pan and Savvides, Marios and Li, Sharon and Wang, Jindong},
  journal={arXiv preprint arXiv:2604.10784},
  year={2026}
}
```

