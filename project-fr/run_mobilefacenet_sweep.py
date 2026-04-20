#!/usr/bin/env python3
"""Sequential hyperparameter sweep runner for MobileFaceNet on Colab-friendly GPUs."""

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
import time
from pathlib import Path


SEARCH_SPACE = {
    "lr": [3e-4, 5e-4, 7e-4, 1e-3],
    "weight_decay": [1e-5, 3e-5, 1e-4, 3e-4],
    "batch_size": [128, 192, 256],
    "arcface_m": [0.35, 0.40, 0.45, 0.50],
    "arcface_s": [32.0, 48.0, 64.0],
    "warmup_epochs": [1, 2],
}


BALANCED_12_TRIALS = [
    {"lr": 3e-4, "weight_decay": 1e-5, "batch_size": 128, "arcface_m": 0.35, "arcface_s": 32.0, "warmup_epochs": 1},
    {"lr": 5e-4, "weight_decay": 3e-5, "batch_size": 192, "arcface_m": 0.40, "arcface_s": 48.0, "warmup_epochs": 2},
    {"lr": 7e-4, "weight_decay": 1e-4, "batch_size": 256, "arcface_m": 0.45, "arcface_s": 64.0, "warmup_epochs": 1},
    {"lr": 1e-3, "weight_decay": 3e-4, "batch_size": 128, "arcface_m": 0.50, "arcface_s": 32.0, "warmup_epochs": 2},
    {"lr": 3e-4, "weight_decay": 3e-5, "batch_size": 192, "arcface_m": 0.45, "arcface_s": 64.0, "warmup_epochs": 2},
    {"lr": 5e-4, "weight_decay": 1e-4, "batch_size": 256, "arcface_m": 0.50, "arcface_s": 32.0, "warmup_epochs": 1},
    {"lr": 7e-4, "weight_decay": 3e-4, "batch_size": 128, "arcface_m": 0.35, "arcface_s": 48.0, "warmup_epochs": 2},
    {"lr": 1e-3, "weight_decay": 1e-5, "batch_size": 192, "arcface_m": 0.40, "arcface_s": 64.0, "warmup_epochs": 1},
    {"lr": 3e-4, "weight_decay": 1e-4, "batch_size": 256, "arcface_m": 0.50, "arcface_s": 48.0, "warmup_epochs": 1},
    {"lr": 5e-4, "weight_decay": 3e-4, "batch_size": 128, "arcface_m": 0.35, "arcface_s": 64.0, "warmup_epochs": 2},
    {"lr": 7e-4, "weight_decay": 1e-5, "batch_size": 192, "arcface_m": 0.40, "arcface_s": 32.0, "warmup_epochs": 1},
    {"lr": 1e-3, "weight_decay": 3e-5, "batch_size": 256, "arcface_m": 0.45, "arcface_s": 48.0, "warmup_epochs": 2},
]


PRESETS = {
    "quick": {"max_trials": 6, "epochs_per_trial": 4},
    "standard": {"max_trials": 10, "epochs_per_trial": 5},
    "full": {"max_trials": 16, "epochs_per_trial": 6},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a MobileFaceNet hyperparameter sweep")
    parser.add_argument("--data_root", type=str, default="datasets/dataset_a", help="Dataset root passed to train_example.py")
    parser.add_argument("--train_metadata", type=str, default=None, help="Optional training metadata parquet")
    parser.add_argument("--val_metadata", type=str, default=None, help="Optional validation metadata parquet")
    parser.add_argument("--val_pairs", type=str, default=None, help="Optional validation pairs parquet")
    parser.add_argument("--label_column", type=str, default=None, help="Optional label column override")
    parser.add_argument("--save_root", type=str, default="sweeps/mobilefacenet", help="Directory for trial outputs")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="standard", help="Sweep budget preset")
    parser.add_argument("--max_trials", type=int, default=None, help="Override number of sampled trials")
    parser.add_argument("--epochs_per_trial", type=int, default=None, help="Override epochs per trial")
    parser.add_argument("--eval_batch_size", type=int, default=256, help="Evaluation batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device forwarded to train_example.py")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trial sampling")
    parser.add_argument(
        "--strategy",
        choices=["random", "balanced"],
        default="random",
        help="Trial selection strategy: random sampling or a deterministic balanced coverage plan",
    )
    parser.add_argument("--select_metric", type=str, default="AUC", help="Validation metric used to rank trials")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision for every trial")
    parser.add_argument("--no_augment", action="store_true", help="Disable augmentations for every trial")
    return parser.parse_args()


def random_trials(max_trials, seed):
    keys = list(SEARCH_SPACE.keys())
    all_values = list(itertools.product(*(SEARCH_SPACE[key] for key in keys)))
    rng = random.Random(seed)

    if max_trials >= len(all_values):
        rng.shuffle(all_values)
        sampled = all_values
    else:
        sampled = rng.sample(all_values, k=max_trials)

    return [dict(zip(keys, values)) for values in sampled]


def balanced_trials(max_trials, seed):
    if max_trials <= len(BALANCED_12_TRIALS):
        return BALANCED_12_TRIALS[:max_trials]

    used = {tuple(sorted(trial.items())) for trial in BALANCED_12_TRIALS}
    extras = []
    for trial in random_trials(max_trials=10_000, seed=seed):
        trial_key = tuple(sorted(trial.items()))
        if trial_key in used:
            continue
        extras.append(trial)
        used.add(trial_key)
        if len(BALANCED_12_TRIALS) + len(extras) >= max_trials:
            break

    return BALANCED_12_TRIALS + extras[: max_trials - len(BALANCED_12_TRIALS)]


def sample_trials(max_trials, seed, strategy):
    if strategy == "balanced":
        return balanced_trials(max_trials=max_trials, seed=seed)
    return random_trials(max_trials=max_trials, seed=seed)


def metric_candidates(metric_name):
    compact = metric_name.replace("1e-0", "1e-")
    return [metric_name, compact]


def read_metric(summary, metric_name):
    for candidate in metric_candidates(metric_name):
        if candidate in summary and summary[candidate] is not None:
            return float(summary[candidate])

    best_val = summary.get("best_val_performance") or {}
    for candidate in metric_candidates(metric_name):
        if candidate in best_val and best_val[candidate] is not None:
            return float(best_val[candidate])

    return None


def build_command(args, config, save_dir):
    cmd = [
        sys.executable,
        "train_example.py",
        "--data_root",
        args.data_root,
        "--save_dir",
        str(save_dir),
        "--backbone",
        "mobilefacenet",
        "--embedding_dim",
        "128",
        "--loss",
        "arcface",
        "--epochs",
        str(args.epochs_per_trial),
        "--select_metric",
        args.select_metric,
        "--lr",
        str(config["lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--batch_size",
        str(config["batch_size"]),
        "--eval_batch_size",
        str(args.eval_batch_size),
        "--arcface_m",
        str(config["arcface_m"]),
        "--arcface_s",
        str(config["arcface_s"]),
        "--warmup_epochs",
        str(config["warmup_epochs"]),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]

    optional_args = {
        "--train_metadata": args.train_metadata,
        "--val_metadata": args.val_metadata,
        "--val_pairs": args.val_pairs,
        "--label_column": args.label_column,
    }
    for flag, value in optional_args.items():
        if value:
            cmd.extend([flag, value])

    if args.amp:
        cmd.append("--amp")
    if args.no_augment:
        cmd.append("--no_augment")

    return cmd


def write_results_csv(path, rows):
    fieldnames = [
        "trial_id",
        "status",
        "select_metric",
        "best_epoch",
        "AUC",
        "TAR@FAR=1e-6",
        "TAR@FAR=1e-5",
        "TAR@FAR=1e-4",
        "TAR@FAR=1e-3",
        "total_time_seconds",
        "train_images_per_second",
        "peak_gpu_memory_reserved_mb",
        "save_dir",
        "returncode",
        "lr",
        "weight_decay",
        "batch_size",
        "arcface_m",
        "arcface_s",
        "warmup_epochs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main():
    args = parse_args()
    preset = PRESETS[args.preset]
    args.max_trials = args.max_trials or preset["max_trials"]
    args.epochs_per_trial = args.epochs_per_trial or preset["epochs_per_trial"]

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    trials = sample_trials(args.max_trials, args.seed, args.strategy)
    manifest = {
        "preset": args.preset,
        "max_trials": args.max_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "strategy": args.strategy,
        "select_metric": args.select_metric,
        "search_space": SEARCH_SPACE,
        "trials": trials,
    }
    (save_root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Running {len(trials)} MobileFaceNet trials with {args.epochs_per_trial} epochs each. "
        f"Results will be saved under {save_root}."
    )

    results = []
    best_result = None

    for index, config in enumerate(trials, start=1):
        trial_id = f"trial_{index:02d}"
        trial_dir = save_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(args, config, trial_dir)
        print(f"\n[{index}/{len(trials)}] {trial_id}: {json.dumps(config, sort_keys=True)}")
        print(" ".join(cmd))

        start = time.time()
        completed = subprocess.run(cmd, check=False)
        wall_time = time.time() - start

        metrics_path = trial_dir / "metrics.json"
        summary = {}
        if metrics_path.exists():
            summary = json.loads(metrics_path.read_text(encoding="utf-8"))

        result = {
            "trial_id": trial_id,
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "save_dir": str(trial_dir),
            "best_epoch": summary.get("best_epoch"),
            "select_metric": read_metric(summary, args.select_metric),
            "AUC": read_metric(summary, "AUC"),
            "TAR@FAR=1e-6": read_metric(summary, "TAR@FAR=1e-6"),
            "TAR@FAR=1e-5": read_metric(summary, "TAR@FAR=1e-5"),
            "TAR@FAR=1e-4": read_metric(summary, "TAR@FAR=1e-4"),
            "TAR@FAR=1e-3": read_metric(summary, "TAR@FAR=1e-3"),
            "total_time_seconds": summary.get("total_time_seconds", wall_time),
            "train_images_per_second": summary.get("train_images_per_second"),
            "peak_gpu_memory_reserved_mb": summary.get("peak_gpu_memory_reserved_mb"),
        }
        result.update(config)
        results.append(result)

        ranked = [row for row in results if row["select_metric"] is not None]
        ranked.sort(key=lambda row: row["select_metric"], reverse=True)
        best_result = ranked[0] if ranked else None

        write_results_csv(save_root / "sweep_results.csv", sorted(results, key=lambda row: row["trial_id"]))
        if best_result is not None:
            (save_root / "best_config.json").write_text(json.dumps(best_result, indent=2), encoding="utf-8")
            print(
                f"Current best: {best_result['trial_id']} "
                f"({args.select_metric}={best_result['select_metric']:.4f})"
            )

    ranked_results = sorted(
        results,
        key=lambda row: (
            row["select_metric"] is None,
            -(row["select_metric"] if row["select_metric"] is not None else 0.0),
            row["trial_id"],
        ),
    )
    write_results_csv(save_root / "sweep_results_ranked.csv", ranked_results)

    summary = {
        "preset": args.preset,
        "max_trials": args.max_trials,
        "epochs_per_trial": args.epochs_per_trial,
        "select_metric": args.select_metric,
        "num_completed_trials": sum(row["status"] == "ok" for row in results),
        "num_failed_trials": sum(row["status"] != "ok" for row in results),
        "best_result": best_result,
    }
    (save_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSweep complete.")
    if best_result is not None:
        print(json.dumps(best_result, indent=2))
    else:
        print("No successful trial produced a ranked metric.")


if __name__ == "__main__":
    main()
