# Training Data

The generation manifest has 2,460 grounded phrase/crop records. The understanding manifest has 4,000 full-image caption records. Keep the record order, sample identifiers, complete captions, target annotations, and bounding boxes unchanged.

## Download

Sources are hosted in `HuggingSelf/test0519`:

- Ordered training manifests: `handoffs/20261022_four_experiments/runtime.tar`, revision `a78aaa6910afa70f54f00d4e72d0bb03d0ca4499`. Extract only `original/data/t2i.jsonl` and `original/data/i2t.jsonl`.
- Grounded crops: `data_artifacts/unicross_semantic_grounding/semantic_grounding_20260629/semantic_grounded_v2.tar.zst`, revision `34a5e9d9b847f092f1c72f815ecc986c787f9ed6`. Preserve the crop files from this archive.
- Full images: `jackyhate/text-to-image-2M@d6e16ffc2d9051fa91c08b4ebca82554287c28f3`, dataset file `data_1024_10K/data_000000.tar`.

Do not extract every image from the source tar. The dataset loader reads the required members directly.

## Relocate Paths

Image references use either a local image path, a tar member, or a crop of another image reference. Create a JSON mapping from the source prefixes in the downloaded manifests to their locations on your machine:

```json
{
  "/original/images/data_000000.tar": "/data/unimore/data_000000.tar",
  "/original/grounding/crops": "/data/unimore/crops"
}
```

Then run:

```bash
python -m umm.post_training.UniMore prepare-data \
  --t2i-manifest /data/downloaded/t2i.jsonl \
  --i2t-manifest /data/downloaded/i2t.jsonl \
  --path-map /data/path-map.json --output "$UNIMORE_DATA_ROOT"
```

Only image-reference paths are rewritten. Captions and annotations are not altered. The command validates counts, unique identifiers, and required image paths, and records the number of samples in each manifest. Understanding uses the full images and does not require its optional crop references.

The manifest identifiers in `configs/data/` are stable inputs to deterministic negative sampling. Keep them unchanged when reproducing the released model.
