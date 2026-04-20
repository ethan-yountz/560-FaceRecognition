#!/usr/bin/env python3
"""
COMP560 Face Recognition Training Example

Demonstrates how to train a ResNet50-based face recognition model using
ArcFace loss with the provided Parquet datasets, and how to generate
prediction CSV files for submission.

Usage (training):
    python train_example.py --data_root ./datasets/dataset_a --epochs 10
    python train_example.py --data_root ./datasets/dataset_a --loss arcface --epochs 20

Usage (prediction generation):
    python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_a --output predictions/dataset_a.csv
    python train_example.py --predict --checkpoint ./checkpoints/best_model.pth --dataset_root ./datasets/dataset_b --output predictions/dataset_b.csv

This script trains on the test split's unique identities (for demonstration).
Students should design their own training strategies.
"""

import argparse
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ============================================================================
# Dataset
# ============================================================================

class FaceTrainDataset(Dataset):
    """Training dataset loaded from Parquet metadata."""

    def __init__(self, root: str, parquet_file: str = "test.parquet", image_size=(112, 112)):
        self.root = root
        df = pd.read_parquet(os.path.join(root, parquet_file))

        # Build per-identity class labels
        unique_templates = sorted(df["template_id"].unique())
        self.tid_to_label = {tid: i for i, tid in enumerate(unique_templates)}
        self.num_classes = len(unique_templates)

        self.image_paths = df["image_path"].tolist()
        self.labels = [self.tid_to_label[tid] for tid in df["template_id"].values]

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, self.labels[idx]


# ============================================================================
# Loss Functions
# ============================================================================

class ArcFaceLoss(nn.Module):
    """Additive Angular Margin Loss (ArcFace)."""

    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))

        one_hot = torch.zeros_like(cosine).scatter_(1, labels.unsqueeze(1), 1.0)
        target_logits = torch.cos(theta + self.m * one_hot)

        logits = target_logits * self.s
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


# ============================================================================
# Model
# ============================================================================

class TrainableModel(nn.Module):
    """ResNet50 backbone with projection head for training."""

    def __init__(self, embedding_dim=512):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone.fc = nn.Sequential(
              nn.Linear(2048, 1024),
              nn.BatchNorm1d(1024),
              nn.ReLU(inplace=True),
              nn.Linear(1024, embedding_dim),
          )
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self):
        return self._embedding_dim

    def forward(self, images):
        return self.backbone(images)

    def encode(self, images):
        features = self.forward(images)
        return F.normalize(features, p=2, dim=1)


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = FaceTrainDataset(args.data_root, image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    print(f"Training set: {len(dataset)} images, {dataset.num_classes} identities")

    # Model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    # Freeze backbone
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    # important: prevent BN running stats update
    model.backbone.eval()
    # Loss

    if args.loss == "arcface":
        criterion = ArcFaceLoss(args.embedding_dim, dataset.num_classes).to(device)
    elif args.loss == "triplet":
        criterion = TripletLoss(margin=args.margin)
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    # Optimizer
    params = list(model.backbone.fc.parameters())
    if hasattr(criterion, 'parameters'):
        params += list(criterion.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # LR Scheduler: cosine annealing with warmup
    total_steps = len(dataloader) * args.epochs
    warmup_steps = len(dataloader) * args.warmup_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')

    # Resume training (generated with MS Copilot)
    start_epoch = 0
    if args.resume is not None:
       checkpoint = torch.load(args.resume, map_location=device)
       state_dict = checkpoint['model_state_dict']

       filtered_dict = {
           k: v for k, v in state_dict.items()
           if not k.startswith("backbone.fc")
       }

       model.load_state_dict(filtered_dict, strict=False)

       start_epoch = checkpoint['epoch'] + 1
       best_loss = checkpoint.get('loss', float('inf'))

       print(f"Resumed from checkpoint {args.resume} at epoch {start_epoch}. The best loss is now {best_loss}.")


    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(criterion, 'train'):
            criterion.train()

        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.6f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
            }, save_dir / "best_model.pth")
            print(f"  Saved best model (loss={avg_loss:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'embedding_dim': args.embedding_dim,
            }, save_dir / f"checkpoint_epoch{epoch+1}.pth")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


# ============================================================================
# Prediction Generation
# ============================================================================

class FaceTestDataset(Dataset):
    """Test dataset for generating predictions."""

    def __init__(self, root: str, image_size=(112, 112)):
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


def aggregate_template_features(embeddings, template_ids, media_ids):
    """Aggregate image embeddings into template features.

    Protocol: average within each media, then sum across media, then L2-normalize.
    """
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


def predict(args):
    """Generate prediction CSV from a trained checkpoint.

    Loads the trained model, encodes all test images, aggregates template
    features, and computes cosine similarity scores for all pairs.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrainableModel(embedding_dim=args.embedding_dim).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load test data
    dataset = FaceTestDataset(args.dataset_root, image_size=(args.image_size, args.image_size))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    pairs_df = pd.read_parquet(os.path.join(args.dataset_root, "pairs.parquet"))
    print(f"Images: {len(dataset)}, Pairs: {len(pairs_df)}")

    # Encode all images
    emb_list, idx_list = [], []
    with torch.inference_mode():
        for images, indices in tqdm(dataloader, desc="Encoding"):
            emb = model.encode(images.to(device))
            emb_list.append(emb.cpu().numpy())
            idx_list.append(indices.numpy())

    embeddings = np.vstack(emb_list)
    indices = np.concatenate(idx_list)
    embeddings = embeddings[np.argsort(indices)]

    # Aggregate template features
    print("Aggregating template features...")
    template_features = aggregate_template_features(
        embeddings, dataset.template_ids, dataset.media_ids
    )

    # Compute pair scores
    print("Computing pair scores...")
    tid_list = sorted(template_features.keys())
    tid_to_idx = {tid: i for i, tid in enumerate(tid_list)}
    feat_matrix = np.zeros((len(tid_list), args.embedding_dim), dtype=np.float32)
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


def main():
    parser = argparse.ArgumentParser(description="Face Recognition Training Example")

    # Mode
    parser.add_argument("--predict", action="store_true", help="Generate predictions from a checkpoint")

    # Training args
    parser.add_argument("--data_root", type=str, default="./datasets/dataset_a", help="Dataset root for training")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--loss", type=str, default="arcface", choices=["arcface", "triplet"], help="Loss function")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--warmup_epochs", type=int, default=2, help="Warmup epochs")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet loss margin")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    # Prediction args
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pth", help="Checkpoint path for prediction")
    parser.add_argument("--dataset_root", type=str, help="Dataset root for prediction")
    parser.add_argument("--output", type=str, default="predictions/dataset_a.csv", help="Output CSV path for predictions")

    # Shared args
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
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
