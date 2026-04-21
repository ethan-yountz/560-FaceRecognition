#!/usr/bin/env python3
"""
COMP560 Face Recognition Evaluation Script

Evaluates student-submitted prediction files against ground truth.
Students submit a CSV file with cosine similarity scores for each template pair.

Submission format (CSV):
    template_id_1,template_id_2,score
    1,11065,0.732
    1,11066,0.215
    ...

Usage:
    python evaluate.py --student_id <your_id> --prediction <scores.csv>
    python evaluate.py --student_id <your_id> --prediction <scores.csv> --datasets dataset_a dataset_b
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm

# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_tar_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    far_targets: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
) -> Dict[str, float]:
    """Compute True Accept Rate (TAR) at specified False Accept Rates (FAR)."""
    fpr, tpr, _ = roc_curve(labels, scores)

    results = {}
    for far in far_targets:
        idx = np.argmin(np.abs(fpr - far))
        results[f"TAR@FAR={far:.0e}"] = float(tpr[idx] * 100)

    results["AUC"] = float(auc(fpr, tpr) * 100)
    return results


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_dataset(
    prediction_path: str,
    gt_pairs_path: str,
    dataset_name: str,
) -> Dict:
    """Evaluate predictions against ground truth pairs."""

    # Load ground truth
    gt_df = pd.read_parquet(gt_pairs_path)
    gt_df["pair_key"] = gt_df["template_id_1"].astype(str) + "_" + gt_df["template_id_2"].astype(str)

    # Load predictions
    pred_df = pd.read_csv(prediction_path)
    required_cols = {"template_id_1", "template_id_2", "score"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(f"Prediction CSV must have columns: {required_cols}. Got: {set(pred_df.columns)}")

    pred_df["pair_key"] = pred_df["template_id_1"].astype(str) + "_" + pred_df["template_id_2"].astype(str)

    # Merge: only evaluate pairs that exist in ground truth
    merged = gt_df.merge(pred_df[["pair_key", "score"]], on="pair_key", how="left")

    missing = merged["score"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing}/{len(merged)} pairs have no prediction (will use score=0)")
        merged["score"] = merged["score"].fillna(0.0)

    scores = merged["score"].values
    labels = merged["label"].values

    # Compute metrics
    performance = compute_tar_at_far(scores, labels)

    results = {
        "performance": performance,
        "submission_info": {
            "num_predicted_pairs": int(len(pred_df)),
            "num_gt_pairs": int(len(gt_df)),
            "num_matched_pairs": int(len(merged) - missing),
            "num_missing_pairs": int(missing),
            "num_positive_pairs": int(labels.sum()),
            "num_negative_pairs": int((labels == 0).sum()),
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="COMP560 Face Recognition Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--student_id", type=str, required=True, help="Your student ID")
    parser.add_argument(
        "--prediction", type=str, required=True,
        help="Path to prediction CSV file (or directory containing per-dataset CSVs named dataset_a.csv, dataset_b.csv)",
    )
    parser.add_argument("--datasets_root", type=str, default="./datasets", help="Root directory containing datasets")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument(
        "--datasets", type=str, nargs="+",
        default=["dataset_a", "dataset_b"],
        choices=["dataset_a", "dataset_b"],
        help="Datasets to evaluate on",
    )
    parser.add_argument(
        "--acknowledge_benchmark_labels",
        action="store_true",
        help="Required opt-in because this script reads labeled benchmark pairs.parquet files.",
    )

    args = parser.parse_args()
    if not args.acknowledge_benchmark_labels:
        parser.error(
            "Local evaluation reads labeled benchmark pairs.parquet files. "
            "Re-run with --acknowledge_benchmark_labels only for an intentional one-shot evaluation."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("COMP560 Face Recognition Evaluation")
    print("=" * 60)
    print(f"Student ID: {args.student_id}")
    print(f"Prediction: {args.prediction}")
    print(f"Datasets: {args.datasets}")
    print("=" * 60)

    all_results = {
        "student_id": args.student_id,
        "timestamp": timestamp,
        "prediction_path": args.prediction,
        "datasets": {},
    }

    for dataset_name in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Evaluating on {dataset_name}")
        print("=" * 60)

        gt_pairs_path = os.path.join(args.datasets_root, dataset_name, "pairs.parquet")
        if not os.path.exists(gt_pairs_path):
            print(f"  ERROR: Ground truth not found at {gt_pairs_path}")
            all_results["datasets"][dataset_name] = {"error": "ground truth not found"}
            continue

        # Determine prediction file path
        if os.path.isdir(args.prediction):
            pred_path = os.path.join(args.prediction, f"{dataset_name}.csv")
        else:
            pred_path = args.prediction

        if not os.path.exists(pred_path):
            print(f"  ERROR: Prediction file not found at {pred_path}")
            all_results["datasets"][dataset_name] = {"error": "prediction file not found"}
            continue

        try:
            results = evaluate_dataset(pred_path, gt_pairs_path, dataset_name)
            all_results["datasets"][dataset_name] = results

            print(f"\nPerformance Metrics (TAR@FAR):")
            for metric, value in results["performance"].items():
                print(f"  {metric}: {value:.2f}%")

            print(f"\nSubmission Info:")
            for key, value in results["submission_info"].items():
                print(f"  {key}: {value}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["datasets"][dataset_name] = {"error": str(e)}

    # Save results
    output_file = output_dir / f"{args.student_id}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Summary CSV
    summary_file = output_dir / f"{args.student_id}_{timestamp}_summary.csv"
    with open(summary_file, "w") as f:
        f.write("dataset,TAR@FAR=1e-6,TAR@FAR=1e-5,TAR@FAR=1e-4,TAR@FAR=1e-3,AUC\n")
        for dataset_name, results in all_results["datasets"].items():
            if "error" not in results:
                perf = results["performance"]
                f.write(
                    f"{dataset_name},"
                    f"{perf.get('TAR@FAR=1e-06', 0):.2f},"
                    f"{perf.get('TAR@FAR=1e-05', 0):.2f},"
                    f"{perf.get('TAR@FAR=1e-04', 0):.2f},"
                    f"{perf.get('TAR@FAR=1e-03', 0):.2f},"
                    f"{perf.get('AUC', 0):.2f}\n"
                )
    print(f"Summary saved to: {summary_file}")

    return all_results


if __name__ == "__main__":
    main()
