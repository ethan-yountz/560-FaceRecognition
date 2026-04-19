#!/usr/bin/env python3
"""
COMP560 Face Recognition training and prediction utilities.

The training path supports:
- ArcFace or triplet loss
- graph-based validation splits written as parquet files
- validation-based checkpoint selection
- mixed precision on CUDA

Examples:
    python train_example.py --data_root ./datasets/dataset_a --train_metadata ./datasets/dataset_a/splits/val_15_seed42/train_metadata.parquet --val_metadata ./datasets/dataset_a/splits/val_15_seed42/val_metadata.parquet --val_pairs ./datasets/dataset_a/splits/val_15_seed42/val_pairs.parquet --loss arcface --epochs 5

    python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

from evaluate import compute_tar_at_far


def resolve_path(root: str, value: str | None, default_name: str | None = None) -> Path | None:
    if value is None:
        if default_name is None:
            return None
        return Path(root) / default_name

    path = Path(value)
    if path.exists():
        return path
    return Path(root) / value


def build_train_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class FaceTrainDataset(Dataset):
    """Training dataset loaded from parquet metadata."""

    def __init__(self, root: str, parquet_file: str, image_size=(112, 112), label_column: str | None = None, augment: bool = True):
        self.root = root
        parquet_path = resolve_path(root, parquet_file)
        df = pd.read_parquet(parquet_path)

        if label_column is None:
            label_column = "component_id" if "component_id" in df.columns else "template_id"
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {parquet_path}")

        unique_labels = sorted(df[label_column].unique())
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        self.label_column = label_column

        self.image_paths = df["image_path"].tolist()
        self.labels = [self.label_to_index[label] for label in df[label_column].tolist()]
        self.transform = build_train_transform(image_size) if augment else build_eval_transform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


class FaceEvalDataset(Dataset):
    """Evaluation dataset for template scoring."""

    def __init__(self, root: str, parquet_file: str, image_size=(112, 112)):
        self.root = root
        parquet_path = resolve_path(root, parquet_file)
        df = pd.read_parquet(parquet_path)
        self.image_paths = df["image_path"].tolist()
        self.template_ids = df["template_id"].values
        self.media_ids = df["media_id"].values
        self.transform = build_eval_transform(image_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        return self.transform(image), idx


class ArcFaceLoss(nn.Module):
    """Additive angular margin loss."""

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        logits = torch.cos(theta + self.m * one_hot) * self.s
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """Online hard triplet mining loss."""

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = 1 - torch.mm(embeddings, embeddings.t())

        labels = labels.unsqueeze(0)
        same_identity = labels == labels.t()

        loss = torch.tensor(0.0, device=embeddings.device)
        count = 0

        for i in range(embeddings.size(0)):
            pos_mask = same_identity[i].clone()
            pos_mask[i] = False
            neg_mask = ~same_identity[i]

            if pos_mask.any() and neg_mask.any():
                hardest_pos = dist_mat[i][pos_mask].max()
                hardest_neg = dist_mat[i][neg_mask].min()
                loss += F.relu(hardest_pos - hardest_neg + self.margin)
                count += 1

        return loss / max(count, 1)


class TrainableModel(nn.Module):
    """ResNet50 backbone with a projection head."""

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Linear(2048, embedding_dim)
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, images):
        return self.backbone(images)

    def encode(self, images):
        features = self.forward(images)
        return F.normalize(features, p=2, dim=1)


def aggregate_template_features(embeddings, template_ids, media_ids):
    """Average within media, sum across media, then L2-normalize."""
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


def encode_dataset(model, dataloader, device):
    emb_list = []
    idx_list = []
    with torch.inference_mode():
        for images, indices in tqdm(dataloader, desc="Encoding", leave=False):
            emb = model.encode(images.to(device, non_blocking=(device.type == "cuda")))
            emb_list.append(emb.cpu().numpy())
            idx_list.append(indices.numpy())

    embeddings = np.vstack(emb_list)
    indices = np.concatenate(idx_list)
    return embeddings[np.argsort(indices)]


def score_pairs(template_features, pairs_df, embedding_dim):
    tid_list = sorted(template_features.keys())
    tid_to_idx = {tid: i for i, tid in enumerate(tid_list)}
    feat_matrix = np.zeros((len(tid_list), embedding_dim), dtype=np.float32)
    for tid, feat in template_features.items():
        feat_matrix[tid_to_idx[tid]] = feat

    t1s = pairs_df["template_id_1"].values
    t2s = pairs_df["template_id_2"].values
    scores = np.zeros(len(pairs_df), dtype=np.float32)

    batch = 500_000
    for i in tqdm(range(0, len(pairs_df), batch), desc="Scoring", leave=False):
        end = min(i + batch, len(pairs_df))
        idx1 = np.array([tid_to_idx[t] for t in t1s[i:end]])
        idx2 = np.array([tid_to_idx[t] for t in t2s[i:end]])
        scores[i:end] = np.sum(feat_matrix[idx1] * feat_matrix[idx2], axis=1)
    return scores


def evaluate_model(model, args, device):
    if not args.val_metadata or not args.val_pairs:
        return None

    val_dataset = FaceEvalDataset(args.data_root, args.val_metadata, image_size=(args.image_size, args.image_size))
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_pairs = pd.read_parquet(resolve_path(args.data_root, args.val_pairs))

    start = time.time()
    embeddings = encode_dataset(model, val_loader, device)
    template_features = aggregate_template_features(embeddings, val_dataset.template_ids, val_dataset.media_ids)
    scores = score_pairs(template_features, val_pairs, model.embedding_dim)
    elapsed = time.time() - start

    performance = compute_tar_at_far(scores, val_pairs["label"].values)
    performance["eval_time_seconds"] = elapsed
    performance["eval_images_per_second"] = len(val_dataset) / max(elapsed, 1e-8)
    return performance


def checkpoint_payload(epoch, model, optimizer, args, extra=None):
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "embedding_dim": args.embedding_dim,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    return payload


def metric_value(name, performance):
    if performance is None:
        return None
    if name not in performance:
        raise ValueError(f"Selection metric '{name}' not found in validation metrics: {sorted(performance.keys())}")
    return float(performance[name])


def summary_validation_metrics(performance):
    if performance is None:
        return {}
    def metric(*names):
        for name in names:
            if name in performance:
                return performance.get(name)
        return None
    return {
        "AUC": performance.get("AUC"),
        "TAR@FAR=1e-6": metric("TAR@FAR=1e-6", "TAR@FAR=1e-06"),
        "TAR@FAR=1e-5": metric("TAR@FAR=1e-5", "TAR@FAR=1e-05"),
        "TAR@FAR=1e-4": metric("TAR@FAR=1e-4", "TAR@FAR=1e-04"),
        "TAR@FAR=1e-3": metric("TAR@FAR=1e-3", "TAR@FAR=1e-03"),
        "eval_time_seconds": performance.get("eval_time_seconds"),
        "eval_images_per_second": performance.get("eval_images_per_second"),
    }


def train(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    amp_enabled = args.amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = FaceTrainDataset(
        args.data_root,
        args.train_metadata,
        image_size=(args.image_size, args.image_size),
        label_column=args.label_column,
        augment=not args.no_augment,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Training set: {len(train_dataset)} images, {train_dataset.num_classes} classes using label '{train_dataset.label_column}'")

    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)

    if args.loss == "arcface":
        criterion = ArcFaceLoss(args.embedding_dim, train_dataset.num_classes, s=args.arcface_s, m=args.arcface_m).to(device)
    elif args.loss == "triplet":
        criterion = TripletLoss(margin=args.margin)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    params = list(model.parameters())
    if hasattr(criterion, "parameters"):
        params += list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_metric = -float("inf")
    best_epoch = None
    best_performance = None
    best_train_loss = None

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        if hasattr(criterion, "train"):
            criterion.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", file=sys.stdout, dynamic_ncols=True)
        for images, labels in pbar:
            images = images.to(device, non_blocking=(device.type == "cuda"))
            labels = labels.to(device, non_blocking=(device.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            if amp_enabled:
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_loss = running_loss / len(train_loader)
        epoch_seconds = time.time() - epoch_start
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}, time={epoch_seconds:.2f}s")

        performance = evaluate_model(model, args, device)
        trial_metric = None
        if performance is not None:
            trial_metric = metric_value(args.select_metric, performance)
            print(f"  Validation {args.select_metric}: {trial_metric:.4f}")
            print(
                "  Validation metrics: "
                + ", ".join(f"{key}={value:.4f}" for key, value in performance.items() if isinstance(value, (int, float)))
            )

        history_entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "epoch_seconds": epoch_seconds,
            "lr": scheduler.get_last_lr()[0],
            "val_performance": performance,
        }
        history.append(history_entry)

        save_extra = {"loss": avg_loss, "val_performance": performance}
        if performance is not None:
            if trial_metric is not None and trial_metric > best_metric:
                best_metric = trial_metric
                best_epoch = epoch + 1
                best_performance = performance
                best_train_loss = avg_loss
                torch.save(checkpoint_payload(epoch, model, optimizer, args, save_extra), save_dir / "best_model.pth")
                print(f"  Saved best model ({args.select_metric}={trial_metric:.4f})")
        else:
            if best_train_loss is None or avg_loss < best_train_loss:
                best_train_loss = avg_loss
                best_epoch = epoch + 1
                torch.save(checkpoint_payload(epoch, model, optimizer, args, save_extra), save_dir / "best_model.pth")
                print(f"  Saved best model (loss={avg_loss:.4f})")

        if (epoch + 1) % args.save_every == 0:
            torch.save(checkpoint_payload(epoch, model, optimizer, args, save_extra), save_dir / f"checkpoint_epoch{epoch + 1}.pth")

    summary = {
        "args": vars(args),
        "num_train_images": len(train_dataset),
        "num_train_classes": train_dataset.num_classes,
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_select_metric": best_metric if best_performance is not None else None,
        "best_select_metric_name": args.select_metric if best_performance is not None else None,
        "best_val_performance": best_performance,
        "history": history,
    }
    summary.update({key: value for key, value in summary_validation_metrics(best_performance).items() if value is not None})

    with open(save_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nTraining complete. Best epoch: {best_epoch}")
    print(f"Artifacts saved to: {save_dir}")
    return summary


def predict(args):
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    embedding_dim = checkpoint.get("embedding_dim", args.embedding_dim)
    model = TrainableModel(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    dataset = FaceEvalDataset(args.dataset_root, args.eval_metadata or "test.parquet", image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    pairs_path = resolve_path(args.dataset_root, args.eval_pairs, "pairs.parquet")
    pairs_df = pd.read_parquet(pairs_path)
    print(f"Images: {len(dataset)}, Pairs: {len(pairs_df)}")

    embeddings = encode_dataset(model, dataloader, device)
    print("Aggregating template features...")
    template_features = aggregate_template_features(embeddings, dataset.template_ids, dataset.media_ids)
    print("Computing pair scores...")
    scores = score_pairs(template_features, pairs_df, model.embedding_dim)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame({
        "template_id_1": pairs_df["template_id_1"].values,
        "template_id_2": pairs_df["template_id_2"].values,
        "score": scores,
    })
    out_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output} ({len(out_df)} pairs)")


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Training Example")

    parser.add_argument("--predict", action="store_true", help="Generate predictions from a checkpoint")

    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a", help="Dataset root for training")
    parser.add_argument("--train_metadata", type=str, default="test.parquet", help="Training metadata parquet")
    parser.add_argument("--val_metadata", type=str, default=None, help="Validation metadata parquet")
    parser.add_argument("--val_pairs", type=str, default=None, help="Validation pairs parquet")
    parser.add_argument("--label_column", type=str, default=None, help="Label column for training; defaults to component_id if present")

    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="arcface", choices=["arcface", "triplet"], help="Loss function")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--arcface_s", type=float, default=30.0, help="ArcFace scale")
    parser.add_argument("--arcface_m", type=float, default=0.50, help="ArcFace angular margin")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--select_metric", type=str, default="AUC", help="Validation metric used to pick the best checkpoint")
    parser.add_argument("--no_augment", action="store_true", help="Disable training augmentations")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")

    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Checkpoint path for prediction")
    parser.add_argument("--dataset_root", type=str, help="Dataset root for prediction")
    parser.add_argument("--eval_metadata", type=str, default=None, help="Evaluation metadata parquet for prediction mode")
    parser.add_argument("--eval_pairs", type=str, default=None, help="Evaluation pairs parquet for prediction mode")
    parser.add_argument("--output", type=str, default="predictions/dataset_a.csv", help="Output CSV path for predictions")

    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Evaluation batch size")
    parser.add_argument("--image_size", type=int, default=112, help="Image size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    if args.predict:
        if not args.dataset_root:
            parser.error("--dataset_root is required for prediction mode")
        predict(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
