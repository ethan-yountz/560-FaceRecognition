#!/usr/bin/env python3
"""Run a baseline backbone and record performance and efficiency metrics."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate_dataset
from models.resnet_baseline import FaceDataset, aggregate_template_features, create_encoder


def maybe_efficiency_score(args, throughput, peak_gpu_memory_mb, embedding_dim):
    values = [
        args.alpha,
        args.beta,
        args.gamma,
        args.throughput_max,
        args.memory_max,
        args.embdim_max,
        peak_gpu_memory_mb,
    ]
    if any(value is None for value in values):
        return None

    return (
        args.alpha * (throughput / args.throughput_max)
        + args.beta * (1.0 - (peak_gpu_memory_mb / args.memory_max))
        + args.gamma * (1.0 - (embedding_dim / args.embdim_max))
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark a baseline face recognition backbone")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Prediction CSV output path")
    parser.add_argument("--metrics_output", type=str, required=True, help="JSON metrics output path")
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilefacenet"],
        help="Image backbone used to produce embeddings",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path required for MobileFaceNet inference")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--alpha", type=float, default=None, help="Efficiency score alpha weight")
    parser.add_argument("--beta", type=float, default=None, help="Efficiency score beta weight")
    parser.add_argument("--gamma", type=float, default=None, help="Efficiency score gamma weight")
    parser.add_argument("--throughput_max", type=float, default=None, help="Efficiency score throughput max")
    parser.add_argument("--memory_max", type=float, default=None, help="Efficiency score memory max in MB")
    parser.add_argument("--embdim_max", type=float, default=None, help="Efficiency score embedding dimension max")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = FaceDataset(str(dataset_root))
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    pairs_df = pd.read_parquet(dataset_root / "pairs.parquet")

    wall_start = time.time()
    model = create_encoder(args.backbone, args.device, checkpoint_path=args.checkpoint)

    if args.device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    embeddings_list = []
    indices_list = []
    encode_start = time.time()
    with torch.inference_mode():
        for images, indices in tqdm(dataloader, desc="Encoding"):
            emb = model.encode(images)
            embeddings_list.append(emb.cpu().numpy())
            indices_list.append(indices.numpy())
    if args.device == "cuda":
        torch.cuda.synchronize()
    encode_time = time.time() - encode_start

    peak_gpu_memory_allocated_mb = None
    peak_gpu_memory_reserved_mb = None
    peak_gpu_memory_mb = None
    if args.device == "cuda":
        peak_gpu_memory_allocated_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_gpu_memory_reserved_mb = torch.cuda.max_memory_reserved() / (1024 * 1024)
        peak_gpu_memory_mb = peak_gpu_memory_reserved_mb

    embeddings = np.vstack(embeddings_list)
    indices = np.concatenate(indices_list)
    embeddings = embeddings[np.argsort(indices)]

    aggregate_start = time.time()
    template_features = aggregate_template_features(
        embeddings,
        dataset.template_ids,
        dataset.media_ids,
    )
    aggregate_time = time.time() - aggregate_start

    score_start = time.time()
    tid_list = sorted(template_features.keys())
    tid_to_idx = {tid: i for i, tid in enumerate(tid_list)}
    feat_matrix = np.zeros((len(tid_list), embeddings.shape[1]), dtype=np.float32)
    for tid, feat in template_features.items():
        feat_matrix[tid_to_idx[tid]] = feat

    t1s = pairs_df["template_id_1"].values
    t2s = pairs_df["template_id_2"].values
    scores = np.zeros(len(pairs_df), dtype=np.float32)

    batch = 500_000
    for i in tqdm(range(0, len(pairs_df), batch), desc="Scoring"):
        end = min(i + batch, len(pairs_df))
        idx1 = np.array([tid_to_idx[t] for t in t1s[i:end]])
        idx2 = np.array([tid_to_idx[t] for t in t2s[i:end]])
        scores[i:end] = np.sum(feat_matrix[idx1] * feat_matrix[idx2], axis=1)
    score_time = time.time() - score_start

    out_df = pd.DataFrame({
        "template_id_1": t1s,
        "template_id_2": t2s,
        "score": scores,
    })
    write_start = time.time()
    out_df.to_csv(output_path, index=False)
    write_time = time.time() - write_start

    total_time = time.time() - wall_start
    throughput = len(dataset) / encode_time
    embedding_dim = int(embeddings.shape[1])

    performance = evaluate_dataset(
        str(output_path),
        str(dataset_root / "pairs.parquet"),
        dataset_root.name,
    )["performance"]

    result = {
        "dataset": dataset_root.name,
        "backbone": args.backbone,
        "checkpoint": args.checkpoint,
        "device": args.device,
        "num_images": len(dataset),
        "num_pairs": int(len(pairs_df)),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "encode_time_seconds": encode_time,
        "aggregate_time_seconds": aggregate_time,
        "score_time_seconds": score_time,
        "csv_write_time_seconds": write_time,
        "total_time_seconds": total_time,
        "throughput_images_per_second": throughput,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "peak_gpu_memory_allocated_mb": peak_gpu_memory_allocated_mb,
        "peak_gpu_memory_reserved_mb": peak_gpu_memory_reserved_mb,
        "embedding_dim": embedding_dim,
        "performance": performance,
        "efficiency_score": maybe_efficiency_score(args, throughput, peak_gpu_memory_mb, embedding_dim),
    }

    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
