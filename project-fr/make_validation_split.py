#!/usr/bin/env python3
"""Create train/validation splits from connected components of positive pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        sa = self.size.get(ra, 1)
        sb = self.size.get(rb, 1)
        if sa < sb:
            ra, rb = rb, ra
            sa, sb = sb, sa
        self.parent[rb] = ra
        self.size[ra] = sa + sb


def build_component_mapping(metadata_df, pairs_df):
    uf = UnionFind()
    template_ids = metadata_df["template_id"].drop_duplicates().tolist()
    for tid in template_ids:
        uf.find(int(tid))

    positives = pairs_df.loc[pairs_df["label"] == 1, ["template_id_1", "template_id_2"]]
    for row in positives.itertuples(index=False):
        uf.union(int(row.template_id_1), int(row.template_id_2))

    roots = {int(tid): uf.find(int(tid)) for tid in template_ids}
    root_to_component = {root: idx for idx, root in enumerate(sorted(set(roots.values())))}
    template_to_component = {tid: root_to_component[root] for tid, root in roots.items()}
    return template_to_component


def select_validation_components(metadata_df, template_to_component, val_fraction, seed):
    template_component_df = metadata_df[["template_id"]].drop_duplicates().copy()
    template_component_df["component_id"] = template_component_df["template_id"].map(template_to_component)

    component_sizes = (
        template_component_df.groupby("component_id")["template_id"]
        .nunique()
        .reset_index(name="num_templates")
    )

    rng = np.random.default_rng(seed)
    order = component_sizes["component_id"].to_numpy().copy()
    rng.shuffle(order)

    target_templates = max(1, int(round(template_component_df["template_id"].nunique() * val_fraction)))
    chosen = []
    running = 0

    size_lookup = dict(zip(component_sizes["component_id"], component_sizes["num_templates"]))
    for component_id in order:
        chosen.append(int(component_id))
        running += int(size_lookup[int(component_id)])
        if running >= target_templates:
            break

    return set(chosen)


def main():
    parser = argparse.ArgumentParser(description="Create graph-based validation splits for face recognition")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset directory")
    parser.add_argument("--metadata", type=str, default="test.parquet", help="Metadata parquet filename")
    parser.add_argument("--pairs", type=str, default="pairs.parquet", help="Pairs parquet filename")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for split files")
    parser.add_argument("--val_fraction", type=float, default=0.15, help="Fraction of templates to hold out")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    metadata_path = dataset_root / args.metadata
    pairs_path = dataset_root / args.pairs
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "splits" / f"val_{int(args.val_fraction * 100):02d}_seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_parquet(metadata_path)
    pairs_df = pd.read_parquet(pairs_path)

    template_to_component = build_component_mapping(metadata_df, pairs_df)
    val_components = select_validation_components(metadata_df, template_to_component, args.val_fraction, args.seed)

    annotated_metadata = metadata_df.copy()
    annotated_metadata["component_id"] = annotated_metadata["template_id"].map(template_to_component)

    val_mask = annotated_metadata["component_id"].isin(val_components)
    train_metadata = annotated_metadata.loc[~val_mask].copy()
    val_metadata = annotated_metadata.loc[val_mask].copy()

    train_templates = set(train_metadata["template_id"].drop_duplicates().tolist())
    val_templates = set(val_metadata["template_id"].drop_duplicates().tolist())

    train_pairs = pairs_df[
        pairs_df["template_id_1"].isin(train_templates) & pairs_df["template_id_2"].isin(train_templates)
    ].copy()
    val_pairs = pairs_df[
        pairs_df["template_id_1"].isin(val_templates) & pairs_df["template_id_2"].isin(val_templates)
    ].copy()

    train_metadata_path = output_dir / "train_metadata.parquet"
    val_metadata_path = output_dir / "val_metadata.parquet"
    train_pairs_path = output_dir / "train_pairs.parquet"
    val_pairs_path = output_dir / "val_pairs.parquet"
    manifest_path = output_dir / "split_manifest.json"

    train_metadata.to_parquet(train_metadata_path, index=False)
    val_metadata.to_parquet(val_metadata_path, index=False)
    train_pairs.to_parquet(train_pairs_path, index=False)
    val_pairs.to_parquet(val_pairs_path, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "metadata": str(metadata_path),
        "pairs": str(pairs_path),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "num_templates_total": int(annotated_metadata["template_id"].nunique()),
        "num_components_total": int(annotated_metadata["component_id"].nunique()),
        "num_train_templates": int(train_metadata["template_id"].nunique()),
        "num_val_templates": int(val_metadata["template_id"].nunique()),
        "num_train_components": int(train_metadata["component_id"].nunique()),
        "num_val_components": int(val_metadata["component_id"].nunique()),
        "num_train_images": int(len(train_metadata)),
        "num_val_images": int(len(val_metadata)),
        "num_train_pairs": int(len(train_pairs)),
        "num_val_pairs": int(len(val_pairs)),
        "num_train_positive_pairs": int(train_pairs["label"].sum()),
        "num_val_positive_pairs": int(val_pairs["label"].sum()),
        "train_metadata": str(train_metadata_path),
        "val_metadata": str(val_metadata_path),
        "train_pairs": str(train_pairs_path),
        "val_pairs": str(val_pairs_path),
    }

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
