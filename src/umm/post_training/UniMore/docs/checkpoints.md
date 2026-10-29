# Checkpoints

All three released models use BAGEL-7B-MoT. Repository: `HuggingSelf/test0519`; fixed revision: `34a5e9d9b847f092f1c72f815ecc986c787f9ed6`.

| Model | Directory | SHA256 of `ema.safetensors` |
|---|---|---|
| Unimore | `checkpoints/n12_l40t2i_qdw_tv_i2t002` | `16eed14b3d372219dbf5efe38c9cb4500d072676a892a2cdbaacf4f3c38c226f` |
| Generation branch | `checkpoints/v2_semantic_ground_implicit_t2i` | `375e6c7dd83b30ede6a3016b6ca05ed0203f1990f715e913dc828b694970504a` |
| Understanding branch | `checkpoints/qv3_i2t_top4k_detailwrong_lr18e5_s600_step0599` | `8082e10e8e23a1f0749ae6c828278db68e4f07121681c5cd8543041c62970adb` |

Use `python -m umm.post_training.UniMore download --model unimore`, `--model t2i`, or `--model i2t` with an empty output directory. Existing published artifact paths are used directly.

The original model is `ByteDance-Seed/BAGEL-7B-MoT@5019f57d168e5816e8f3f701b17cc816bb7cf24b`. Its EMA SHA256 is `0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3`. `python -m umm.post_training.UniMore download --model base` retrieves its weights, VAE, and tokenizer/configuration files.
