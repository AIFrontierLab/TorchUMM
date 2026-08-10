from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from umm.post_training.unipath.planner.models import (
    PATHS,
    build_planner_model,
    choose_path_for_model,
    planner_probs,
    prepare_feature_tensor,
)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a UniPath path planner on cached BAGEL features.")
    parser.add_argument("--features-pt", required=True, help="Feature cache with `features` and metadata labels.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bagel-feature-mode", choices=["full", "joint_text_mean_image", "joint_text_mean_last_image"], default="joint_text_mean_last_image")
    parser.add_argument("--planner-type", choices=["mlp", "bpr", "gate_bpr", "two_stage"], default="mlp")
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--best-by", choices=["selected_target", "val_loss", "mmmu_accuracy"], default="val_loss")
    parser.add_argument("--drop-all-wrong", type=str2bool, default=True)
    parser.add_argument("--single-positive-weight", type=float, default=3.0)
    parser.add_argument("--double-positive-weight", type=float, default=2.0)
    parser.add_argument("--multi-positive-weight", type=float, default=1.0)
    parser.add_argument("--nondirect-positive-weight", type=float, default=1.3)
    parser.add_argument("--train-sampler", choices=["epoch", "uniform"], default="epoch")
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--feature-dropout", type=float, default=0.05)
    parser.add_argument("--feature-noise-std", type=float, default=0.02)
    parser.add_argument("--dist-reg-weight", type=float, default=0.03)
    parser.add_argument("--dist-reg-prior", default="0.58,0.20,0.13,0.05,0.04")
    parser.add_argument("--dist-reg-mode", choices=["prob_kl", "choice_kl"], default="prob_kl")
    parser.add_argument("--dist-reg-temperature", type=float, default=1.0)
    parser.add_argument("--level-weight", action="append", default=[], help="LEVEL=WEIGHT, can be repeated.")
    parser.add_argument("--source-tag-weight", action="append", default=[], help="SOURCE_TAG=WEIGHT, can be repeated.")
    parser.add_argument("--mmmu-features-pt", default="", help="Optional MMMU feature cache for checkpoint selection.")
    parser.add_argument("--mmmu-results-root", default="", help="Optional root with mmmu_<path>/cases_with_lengths.jsonl.")
    parser.add_argument("--mmmu-eval-every", type=int, default=1)
    parser.add_argument("--mmmu-max-samples", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="Smoke-test limiter before train/val split.")
    return parser.parse_args()


class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], features_by_id: dict[str, torch.Tensor]) -> None:
        self.rows = rows
        self.features_by_id = features_by_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        return {
            "x": self.features_by_id[row["data_id"]].float(),
            "y": torch.tensor(row["labels"], dtype=torch.float32),
            "sample_weight": float(row["sample_weight"]),
            "data_id": row["data_id"],
            "level": row["level"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        "data_id": [item["data_id"] for item in batch],
        "level": [item["level"] for item in batch],
    }


def parse_weight_overrides(values: list[str], *, name: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{name} must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = float(value)
    return out


def parse_float_list(value: str, expected_len: int) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated floats, got {len(out)}")
    return out


def sample_weight_for(num_positive: int, args: argparse.Namespace) -> float:
    if num_positive <= 0:
        return 0.0
    if num_positive == 1:
        return float(args.single_positive_weight)
    if num_positive == 2:
        return float(args.double_positive_weight)
    return float(args.multi_positive_weight)


def metadata_weight(row: dict[str, Any], level_weights: dict[str, float], source_tag_weights: dict[str, float]) -> float:
    weight = float(level_weights.get(str(row.get("level", "")), 1.0))
    source_tag = row.get("source_tag")
    if source_tag is not None:
        weight *= float(source_tag_weights.get(str(source_tag), 1.0))
    return weight


def split_by_level(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["level"])].append(row)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for level_rows in buckets.values():
        rng.shuffle(level_rows)
        n_val = max(1, round(len(level_rows) * val_ratio)) if len(level_rows) > 1 else 0
        val.extend(level_rows[:n_val])
        train.extend(level_rows[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def augment_features(x: torch.Tensor, dropout_p: float, noise_std: float) -> torch.Tensor:
    out = x
    if dropout_p > 0.0:
        keep = torch.bernoulli(torch.full_like(out, 1.0 - dropout_p))
        out = out * keep / max(1.0 - dropout_p, 1e-8)
    if noise_std > 0.0:
        out = out + torch.randn_like(out) * float(noise_std)
    return out


def weighted_bce_loss(logits: torch.Tensor, labels: torch.Tensor, sample_weight: torch.Tensor, nondirect_positive_weight: float) -> torch.Tensor:
    element_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    class_weights = torch.ones_like(labels)
    class_weights[:, 1:] = torch.where(
        labels[:, 1:] > 0.5,
        torch.full_like(labels[:, 1:], float(nondirect_positive_weight)),
        torch.ones_like(labels[:, 1:]),
    )
    per_sample = (element_loss * class_weights).mean(dim=1)
    return (per_sample * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)


def distribution_regularization(
    model: torch.nn.Module,
    probs: torch.Tensor,
    *,
    prior: torch.Tensor,
    weight: float,
    mode: str,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return probs.new_tensor(0.0)
    prior = prior.to(device=probs.device, dtype=probs.dtype)
    prior = prior / prior.sum().clamp_min(1e-8)
    if mode == "choice_kl":
        counts = probs.new_zeros(len(PATHS))
        for i in range(probs.shape[0]):
            counts[choose_path_for_model(model, probs[i].detach(), margin)] += 1.0
        observed = (counts + float(temperature)) / (counts.sum() + float(temperature) * len(PATHS))
    else:
        observed = probs.mean(dim=0)
        if temperature != 1.0:
            observed = torch.softmax(torch.log(observed.clamp_min(1e-8)) / max(float(temperature), 1e-8), dim=0)
    observed = observed / observed.sum().clamp_min(1e-8)
    return float(weight) * torch.sum(observed * (torch.log(observed.clamp_min(1e-8)) - torch.log(prior.clamp_min(1e-8))))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    features_by_id: dict[str, torch.Tensor],
    batch_size: int,
    margin: float,
    nondirect_positive_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(FeatureDataset(rows, features_by_id), batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    selected_target_sum = 0.0
    path_counts = Counter()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(device)
        labels = batch["y"].to(device)
        sample_weight = batch["sample_weight"].to(device)
        logits = model(x)
        probs = planner_probs(model, x)
        loss = weighted_bce_loss(logits, labels, sample_weight, nondirect_positive_weight)
        total_loss += float(loss.item()) * float(sample_weight.sum().item())
        total_weight += float(sample_weight.sum().item())
        for i in range(labels.shape[0]):
            chosen = choose_path_for_model(model, probs[i], margin)
            selected_target = float(labels[i, chosen].item())
            selected_target_sum += selected_target
            path_counts[PATHS[chosen]] += 1
            predictions.append(
                {
                    "data_id": batch["data_id"][i],
                    "level": batch["level"][i],
                    "scores": {PATHS[j]: float(probs[i, j].item()) for j in range(len(PATHS))},
                    "selected_path": PATHS[chosen],
                    "selected_target": selected_target,
                }
            )
    return {
        "loss": total_loss / max(total_weight, 1.0),
        "selected_target_mean": selected_target_sum / max(len(rows), 1),
        "path_counts": dict(path_counts),
        "predictions": predictions,
    }


def load_mmmu_answer_maps(results_root: Path) -> dict[str, dict[str, bool]]:
    maps: dict[str, dict[str, bool]] = {}
    for path_name in PATHS:
        cases_path = results_root / f"mmmu_{path_name}" / "cases_with_lengths.jsonl"
        if not cases_path.exists():
            cases_path = results_root / f"mmmu_{path_name}" / "cases.jsonl"
        if not cases_path.exists():
            raise FileNotFoundError(cases_path)
        path_map: dict[str, bool] = {}
        with cases_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    path_map[str(row["data_id"])] = bool(row.get("answer_ok"))
        maps[path_name] = path_map
    return maps


@torch.no_grad()
def evaluate_mmmu_offline(
    model: torch.nn.Module,
    *,
    mmmu_features: torch.Tensor,
    mmmu_metadata: list[dict[str, Any]],
    answer_maps: dict[str, dict[str, bool]],
    batch_size: int,
    margin: float,
    device: torch.device,
    max_samples: int,
) -> dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    path_counts = Counter()
    n = len(mmmu_metadata) if max_samples <= 0 else min(len(mmmu_metadata), int(max_samples))
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        probs = planner_probs(model, mmmu_features[start:end].to(device))
        for offset in range(end - start):
            data_id = str(mmmu_metadata[start + offset]["data_id"])
            chosen = choose_path_for_model(model, probs[offset], margin)
            path_name = PATHS[chosen]
            path_counts[path_name] += 1
            total += 1
            correct += int(answer_maps[path_name].get(data_id, False))
    return {"accuracy": correct / max(total, 1), "correct": correct, "total": total, "path_counts": dict(path_counts)}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(args.features_pt, map_location="cpu")
    features = prepare_feature_tensor(payload, args.bagel_feature_mode)
    metadata = list(payload["metadata"])
    if len(metadata) != int(features.shape[0]):
        raise ValueError(f"Feature/metadata length mismatch: {features.shape[0]} vs {len(metadata)}")

    features_by_id = {str(row["data_id"]): features[idx] for idx, row in enumerate(metadata)}
    rows = [
        {
            "data_id": str(row["data_id"]),
            "level": str(row.get("level") or ""),
            "source_tag": row.get("source_tag"),
            "labels": [float(v) for v in row["labels"]],
            "num_positive": int(sum(1 for value in row["labels"] if float(value) > 0.5)),
        }
        for row in metadata
    ]
    if args.drop_all_wrong:
        rows = [row for row in rows if row["num_positive"] > 0]
    if args.max_train_samples > 0:
        rows = rows[: int(args.max_train_samples)]

    level_weights = parse_weight_overrides(args.level_weight, name="--level-weight")
    source_tag_weights = parse_weight_overrides(args.source_tag_weight, name="--source-tag-weight")
    for row in rows:
        row["sample_weight"] = sample_weight_for(int(row["num_positive"]), args) * metadata_weight(
            row,
            level_weights,
            source_tag_weights,
        )

    train_rows, val_rows = split_by_level(rows, args.val_ratio, args.seed)
    device = torch.device(args.device)
    model = build_planner_model(args.planner_type, input_dim=int(features.shape[1]), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_dataset = FeatureDataset(train_rows, features_by_id)
    if args.train_sampler == "uniform":
        samples_per_epoch = int(args.batch_size * max(args.steps_per_epoch, 1))
        sampler = RandomSampler(train_dataset, replacement=True, num_samples=samples_per_epoch)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    dist_prior = torch.tensor(parse_float_list(args.dist_reg_prior, len(PATHS)), dtype=torch.float32)
    mmmu_features = None
    mmmu_metadata = None
    mmmu_answer_maps = None
    if args.best_by == "mmmu_accuracy" or args.mmmu_features_pt or args.mmmu_results_root:
        if not args.mmmu_features_pt or not args.mmmu_results_root:
            raise ValueError("--best-by mmmu_accuracy requires --mmmu-features-pt and --mmmu-results-root")
        mmmu_payload = torch.load(args.mmmu_features_pt, map_location="cpu")
        mmmu_features = prepare_feature_tensor(mmmu_payload, args.bagel_feature_mode)
        mmmu_metadata = list(mmmu_payload["metadata"])
        mmmu_answer_maps = load_mmmu_answer_maps(Path(args.mmmu_results_root))

    best_state = None
    best_metric = float("inf") if args.best_by == "val_loss" else float("-inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        seen = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            labels = batch["y"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            train_x = augment_features(x, args.feature_dropout, args.feature_noise_std)
            optimizer.zero_grad(set_to_none=True)
            logits = model(train_x)
            probs = planner_probs(model, train_x)
            loss = weighted_bce_loss(logits, labels, sample_weight, args.nondirect_positive_weight)
            loss = loss + distribution_regularization(
                model,
                probs,
                prior=dist_prior,
                weight=args.dist_reg_weight,
                mode=args.dist_reg_mode,
                margin=args.margin,
                temperature=args.dist_reg_temperature,
            )
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * float(sample_weight.sum().item())
            seen += float(sample_weight.sum().item())

        val_eval = evaluate(model, val_rows, features_by_id, args.batch_size, args.margin, args.nondirect_positive_weight, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(seen, 1.0),
            "val_loss": val_eval["loss"],
            "val_selected_target_mean": val_eval["selected_target_mean"],
            "val_path_counts": val_eval["path_counts"],
        }
        mmmu_eval = None
        if mmmu_features is not None and mmmu_metadata is not None and mmmu_answer_maps is not None:
            if epoch == 1 or epoch % max(int(args.mmmu_eval_every), 1) == 0 or args.best_by == "mmmu_accuracy":
                mmmu_eval = evaluate_mmmu_offline(
                    model,
                    mmmu_features=mmmu_features,
                    mmmu_metadata=mmmu_metadata,
                    answer_maps=mmmu_answer_maps,
                    batch_size=args.batch_size,
                    margin=args.margin,
                    device=device,
                    max_samples=args.mmmu_max_samples,
                )
                row.update({"mmmu_accuracy": mmmu_eval["accuracy"], "mmmu_correct": mmmu_eval["correct"], "mmmu_path_counts": mmmu_eval["path_counts"]})
        history.append(row)

        if args.best_by == "selected_target":
            metric = float(val_eval["selected_target_mean"])
        elif args.best_by == "val_loss":
            metric = float(val_eval["loss"])
        else:
            if mmmu_eval is None:
                assert mmmu_features is not None and mmmu_metadata is not None and mmmu_answer_maps is not None
                mmmu_eval = evaluate_mmmu_offline(
                    model,
                    mmmu_features=mmmu_features,
                    mmmu_metadata=mmmu_metadata,
                    answer_maps=mmmu_answer_maps,
                    batch_size=args.batch_size,
                    margin=args.margin,
                    device=device,
                    max_samples=args.mmmu_max_samples,
                )
            metric = float(mmmu_eval["accuracy"])
        improved = metric < best_metric if args.best_by == "val_loss" else metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(
            f"[planner] epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val={row['val_selected_target_mean']:.4f} metric={metric:.4f} best={best_metric:.4f} "
            f"counts={row['val_path_counts']} mmmu={row.get('mmmu_accuracy')}",
            flush=True,
        )
        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_eval = evaluate(model, val_rows, features_by_id, args.batch_size, args.margin, args.nondirect_positive_weight, device)
    all_eval = evaluate(model, rows, features_by_id, args.batch_size, args.margin, args.nondirect_positive_weight, device)
    final_mmmu_eval = None
    if mmmu_features is not None and mmmu_metadata is not None and mmmu_answer_maps is not None:
        final_mmmu_eval = evaluate_mmmu_offline(
            model,
            mmmu_features=mmmu_features,
            mmmu_metadata=mmmu_metadata,
            answer_maps=mmmu_answer_maps,
            batch_size=args.batch_size,
            margin=args.margin,
            device=device,
            max_samples=args.mmmu_max_samples,
        )

    metrics = {
        "paths": PATHS,
        "feature_source": args.features_pt,
        "feature_layout": str(payload.get("feature_layout") or ""),
        "bagel_feature_mode": args.bagel_feature_mode,
        "input_dim": int(features.shape[1]),
        "hidden_dim": args.hidden_dim,
        "num_examples": len(rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "level_counts": dict(Counter(row["level"] for row in rows)),
        "source_tag_counts": dict(Counter(str(row.get("source_tag")) for row in rows)),
        "level_weights": level_weights,
        "source_tag_weights": source_tag_weights,
        "dist_reg_weight": args.dist_reg_weight,
        "dist_reg_prior": parse_float_list(args.dist_reg_prior, len(PATHS)),
        "dist_reg_mode": args.dist_reg_mode,
        "margin": args.margin,
        "best_by": args.best_by,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "drop_all_wrong": args.drop_all_wrong,
        "planner_type": args.planner_type,
        "val": {key: value for key, value in val_eval.items() if key != "predictions"},
        "all": {key: value for key, value in all_eval.items() if key != "predictions"},
        "mmmu": final_mmmu_eval,
        "history": history,
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden_dim": args.hidden_dim,
            "paths": PATHS,
            "cost_order": PATHS,
            "margin": args.margin,
            "feature_source": args.features_pt,
            "feature_layout": str(payload.get("feature_layout") or ""),
            "bagel_feature_mode": args.bagel_feature_mode,
            "pretrain_type": f"unipath_outcome_{args.planner_type}",
            "planner_type": args.planner_type,
            "level_weights": level_weights,
            "source_tag_weights": source_tag_weights,
            "dist_reg_weight": args.dist_reg_weight,
            "dist_reg_prior": parse_float_list(args.dist_reg_prior, len(PATHS)),
            "dist_reg_mode": args.dist_reg_mode,
        },
        output_dir / "planner.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "val_predictions.jsonl", val_eval["predictions"])
    write_jsonl(output_dir / "all_predictions.jsonl", all_eval["predictions"])
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
