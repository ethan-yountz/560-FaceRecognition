#!/usr/bin/env python3
"""
ResNet50 Baseline for Face Recognition

Generates prediction CSV files for evaluation. Uses pretrained ResNet50
to encode face images, aggregates template features, and computes pair scores.

Usage:
    python models/resnet_baseline.py --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
    python models/resnet_baseline.py --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm


class FaceDataset(Dataset):
    def __init__(self, root, image_size=(112, 112)):
        self.root = root
        df = pd.read_parquet(os.path.join(root, "test.parquet"))
        self.image_paths = df["image_path"].tolist()
        self.template_ids = df["template_id"].values
        self.media_ids = df["media_id"].values
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


class ResNetEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Identity()
        self.to(device).eval()

    @torch.inference_mode()
    def encode(self, images):
        return F.normalize(self.backbone(images.to(self.device)), p=2, dim=1)


def aggregate_template_features(embeddings, template_ids, media_ids):
    """Aggregate image embeddings into template features."""
    template_features = {}
    for tid in np.unique(template_ids):
        mask = template_ids == tid
        t_embs = embeddings[mask]
        t_mids = media_ids[mask]

        media_feats = []
        for mid in np.unique(t_mids):
            media_feats.append(t_embs[t_mids == mid].mean(axis=0))

        feat = np.sum(media_feats, axis=0)
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        template_features[tid] = feat

    return template_features


def main():
    parser = argparse.ArgumentParser(description="ResNet50 Baseline - Generate Face Recognition Predictions")
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loading workers")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Load dataset
    dataset = FaceDataset(args.dataset_root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pairs_df = pd.read_parquet(os.path.join(args.dataset_root, "pairs.parquet"))

    print(f"Images: {len(dataset)}, Pairs: {len(pairs_df)}")

    # Encode
    model = ResNetEncoder(args.device)
    embeddings_list, indices_list = [], []

    for images, indices in tqdm(dataloader, desc="Encoding"):
        emb = model.encode(images)
        embeddings_list.append(emb.cpu().numpy())
        indices_list.append(indices.numpy())

    embeddings = np.vstack(embeddings_list)
    indices = np.concatenate(indices_list)
    embeddings = embeddings[np.argsort(indices)]

    # Aggregate templates
    print("Aggregating template features...")
    template_features = aggregate_template_features(
        embeddings, dataset.template_ids, dataset.media_ids
    )

    # Compute pair scores
    print("Computing pair scores...")
    tid_list = sorted(template_features.keys())
    tid_to_idx = {tid: i for i, tid in enumerate(tid_list)}
    feat_matrix = np.zeros((len(tid_list), embeddings.shape[1]), dtype=np.float32)
    for tid, feat in template_features.items():
        feat_matrix[tid_to_idx[tid]] = feat

    t1s = pairs_df["template_id_1"].values
    t2s = pairs_df["template_id_2"].values
    scores = np.zeros(len(pairs_df), dtype=np.float32)

    batch = 500000
    for i in tqdm(range(0, len(pairs_df), batch), desc="Scoring"):
        end = min(i + batch, len(pairs_df))
        idx1 = np.array([tid_to_idx[t] for t in t1s[i:end]])
        idx2 = np.array([tid_to_idx[t] for t in t2s[i:end]])
        scores[i:end] = np.sum(feat_matrix[idx1] * feat_matrix[idx2], axis=1)

    # Save predictions
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "template_id_1": t1s,
        "template_id_2": t2s,
        "score": scores,
    })
    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output} ({len(out_df)} pairs)")


if __name__ == "__main__":
    main()
