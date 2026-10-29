# Checkpoints

All three released models use BAGEL-7B-MoT. Repository: `HuggingSelf/test0519`; fixed revision: `34a5e9d9b847f092f1c72f815ecc986c787f9ed6`.

| Model | Directory |
|---|---|
| Unimore | `checkpoints/n12_l40t2i_qdw_tv_i2t002` |
| Generation branch | `checkpoints/v2_semantic_ground_implicit_t2i` |
| Understanding branch | `checkpoints/qv3_i2t_top4k_detailwrong_lr18e5_s600_step0599` |

Use `python -m umm.post_training.UniMore download --model unimore`, `--model t2i`, or `--model i2t` with an empty output directory. Existing published artifact paths are used directly.

The original model is `ByteDance-Seed/BAGEL-7B-MoT@5019f57d168e5816e8f3f701b17cc816bb7cf24b`. `python -m umm.post_training.UniMore download --model base` retrieves its weights, VAE, and tokenizer/configuration files.
