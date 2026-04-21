#!/usr/bin/env python3
"""Starter scaffold for Qwen3-VL + Tinker fine-tuning on face verification.

This reframes the repository's template-verification task as multimodal
supervised fine-tuning:

- input: a small set of images from template A and template B
- target: the single-token label "same" or "different"

The file is intentionally split into two layers:

1. Data preparation that works with only the repo's local parquet files.
2. A Tinker/Cookbook launch path that becomes active when those dependencies
   are installed in the runtime.

Examples
--------
Preview a few generated conversations:

    python models/qwen_vl_tinker_finetune.py preview \
        --data_root ./datasets/dataset_a

Export a JSONL training file:

    python models/qwen_vl_tinker_finetune.py prepare \
        --data_root ./datasets/dataset_a \
        --output_jsonl ./datasets/dataset_a/qwen_face_train.jsonl \
        --max_examples 5000

Prepare data and launch a first Tinker SFT run:

    python models/qwen_vl_tinker_finetune.py train \
        --data_root ./datasets/dataset_a \
        --output_jsonl ./datasets/dataset_a/qwen_face_train.jsonl \
        --log_path ./logs/qwen_face_tinker
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


SYSTEM_PROMPT = (
    "You are a face verification model. Compare template A and template B. "
    "Respond with exactly one lowercase word: same or different."
)


@dataclass(frozen=True)
class FaceVerificationConversation:
    messages: list[dict[str, Any]]
    metadata: dict[str, Any]


def resolve_path(root: str | Path, value: str | None, default_name: str | None = None) -> Path | None:
    root = Path(root)
    if value is None:
        if default_name is None:
            return None
        return root / default_name

    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    if path.parts[: len(root.parts)] == root.parts:
        return path
    return root / value


def default_split_paths(data_root: str | Path) -> dict[str, Path]:
    split_dir = Path(data_root) / "splits" / "val_15_seed42"
    return {
        "train_metadata": split_dir / "train_metadata.parquet",
        "val_metadata": split_dir / "val_metadata.parquet",
        "train_pairs": split_dir / "train_pairs.parquet",
        "val_pairs": split_dir / "val_pairs.parquet",
    }


def load_metadata_frame(data_root: str | Path, metadata_path: str | None) -> pd.DataFrame:
    resolved = resolve_path(data_root, metadata_path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Metadata parquet not found: {resolved}")
    return pd.read_parquet(resolved)


def load_pairs_frame(data_root: str | Path, pairs_path: str | None) -> pd.DataFrame | None:
    resolved = resolve_path(data_root, pairs_path)
    if resolved is None or not resolved.exists():
        return None
    return pd.read_parquet(resolved)


def build_template_to_images(metadata_df: pd.DataFrame, data_root: str | Path) -> dict[int, list[Path]]:
    data_root = Path(data_root)
    grouped: dict[int, list[Path]] = defaultdict(list)
    for row in metadata_df.itertuples(index=False):
        grouped[int(row.template_id)].append(data_root / row.image_path)
    return {template_id: sorted(paths) for template_id, paths in grouped.items()}


def sample_pairs_from_components(
    metadata_df: pd.DataFrame,
    max_examples: int,
    seed: int,
) -> pd.DataFrame:
    if "component_id" not in metadata_df.columns:
        raise ValueError(
            "No labeled pairs parquet was provided and train_metadata.parquet does not contain "
            "'component_id', so positive/negative training pairs cannot be constructed."
        )

    rng = random.Random(seed)
    template_component_df = metadata_df[["template_id", "component_id"]].drop_duplicates().copy()

    templates_by_component: dict[int, list[int]] = defaultdict(list)
    for row in template_component_df.itertuples(index=False):
        templates_by_component[int(row.component_id)].append(int(row.template_id))

    component_ids = sorted(templates_by_component)
    positive_pairs: list[tuple[int, int, int]] = []
    for component_id, template_ids in templates_by_component.items():
        template_ids = sorted(template_ids)
        if len(template_ids) < 2:
            continue
        local_pairs = []
        for i, left in enumerate(template_ids):
            for right in template_ids[i + 1 :]:
                local_pairs.append((left, right, 1))
        rng.shuffle(local_pairs)
        positive_pairs.extend(local_pairs)

    if not positive_pairs:
        raise ValueError("Could not construct any positive template pairs from component_id labels.")

    if max_examples <= 0:
        max_examples = len(positive_pairs) * 2

    target_positive = min(len(positive_pairs), max_examples // 2)
    target_negative = max_examples - target_positive
    if target_positive == 0:
        target_positive = 1
        target_negative = 0
    positive_pairs = positive_pairs[:target_positive]

    negative_pairs: list[tuple[int, int, int]] = []
    while len(negative_pairs) < target_negative:
        left_component, right_component = rng.sample(component_ids, 2)
        left_template = rng.choice(templates_by_component[left_component])
        right_template = rng.choice(templates_by_component[right_component])
        if left_template == right_template:
            continue
        left_template, right_template = sorted((left_template, right_template))
        negative_pairs.append((left_template, right_template, 0))

    all_pairs = positive_pairs + negative_pairs
    rng.shuffle(all_pairs)
    return pd.DataFrame(all_pairs, columns=["template_id_1", "template_id_2", "label"])


def sample_balanced_pairs(pairs_df: pd.DataFrame, max_examples: int, seed: int) -> pd.DataFrame:
    if max_examples <= 0 or len(pairs_df) <= max_examples:
        return pairs_df.copy()

    rng = random.Random(seed)
    if "label" not in pairs_df.columns:
        sampled_idx = rng.sample(range(len(pairs_df)), k=max_examples)
        return pairs_df.iloc[sampled_idx].reset_index(drop=True)

    pos_df = pairs_df[pairs_df["label"] == 1]
    neg_df = pairs_df[pairs_df["label"] == 0]
    if pos_df.empty or neg_df.empty:
        sampled_idx = rng.sample(range(len(pairs_df)), k=max_examples)
        return pairs_df.iloc[sampled_idx].reset_index(drop=True)

    pos_target = min(len(pos_df), max_examples // 2)
    neg_target = min(len(neg_df), max_examples - pos_target)

    pos_sample = pos_df.sample(n=pos_target, random_state=seed)
    neg_sample = neg_df.sample(n=neg_target, random_state=seed + 1)
    sampled = pd.concat([pos_sample, neg_sample], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)
    return sampled


def sample_template_images(
    template_to_images: dict[int, list[Path]],
    template_id: int,
    max_images: int,
    rng: random.Random,
) -> list[Path]:
    images = list(template_to_images.get(int(template_id), []))
    if not images:
        raise KeyError(f"No images found for template_id={template_id}")
    if len(images) <= max_images:
        return images
    return sorted(rng.sample(images, k=max_images))


def image_reference(image_path: Path, inline_images: bool) -> str:
    if not inline_images:
        return str(image_path)

    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_messages_for_pair(
    pair_row: pd.Series,
    template_to_images: dict[int, list[Path]],
    images_per_template: int,
    inline_images: bool,
    rng: random.Random,
) -> FaceVerificationConversation:
    template_id_1 = int(pair_row["template_id_1"])
    template_id_2 = int(pair_row["template_id_2"])
    label = int(pair_row["label"])

    template_a_images = sample_template_images(template_to_images, template_id_1, images_per_template, rng)
    template_b_images = sample_template_images(template_to_images, template_id_2, images_per_template, rng)

    user_content: list[dict[str, str]] = [
        {"type": "text", "text": "Template A images:"},
    ]
    for image_path in template_a_images:
        user_content.append({"type": "image", "image": image_reference(image_path, inline_images)})

    user_content.append({"type": "text", "text": "Template B images:"})
    for image_path in template_b_images:
        user_content.append({"type": "image", "image": image_reference(image_path, inline_images)})

    user_content.append(
        {
            "type": "text",
            "text": (
                "Do template A and template B belong to the same identity? "
                "Answer with exactly one lowercase word: same or different."
            ),
        }
    )

    answer = "same" if label == 1 else "different"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]
    metadata = {
        "template_id_1": template_id_1,
        "template_id_2": template_id_2,
        "label": label,
        "template_a_num_images": len(template_a_images),
        "template_b_num_images": len(template_b_images),
    }
    return FaceVerificationConversation(messages=messages, metadata=metadata)


def build_conversations(
    metadata_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    data_root: str | Path,
    images_per_template: int,
    inline_images: bool,
    seed: int,
) -> list[FaceVerificationConversation]:
    rng = random.Random(seed)
    template_to_images = build_template_to_images(metadata_df, data_root)

    conversations: list[FaceVerificationConversation] = []
    for _, pair_row in pairs_df.iterrows():
        conversations.append(
            build_messages_for_pair(
                pair_row=pair_row,
                template_to_images=template_to_images,
                images_per_template=images_per_template,
                inline_images=inline_images,
                rng=rng,
            )
        )
    return conversations


def write_jsonl(path: Path, conversations: list[FaceVerificationConversation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in conversations:
            handle.write(
                json.dumps(
                    {
                        "messages": example.messages,
                        "metadata": example.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def prepare_training_conversations(args: argparse.Namespace) -> tuple[list[FaceVerificationConversation], Path]:
    split_paths = default_split_paths(args.data_root)
    train_metadata_path = args.train_metadata or split_paths["train_metadata"]
    train_pairs_path = args.train_pairs
    if train_pairs_path is None and split_paths["train_pairs"].exists():
        train_pairs_path = split_paths["train_pairs"]

    metadata_df = load_metadata_frame(args.data_root, train_metadata_path)
    pairs_df = load_pairs_frame(args.data_root, train_pairs_path)
    if pairs_df is None:
        pairs_df = sample_pairs_from_components(
            metadata_df=metadata_df,
            max_examples=args.max_examples,
            seed=args.seed,
        )
    else:
        pairs_df = sample_balanced_pairs(
            pairs_df=pairs_df,
            max_examples=args.max_examples,
            seed=args.seed,
        )

    conversations = build_conversations(
        metadata_df=metadata_df,
        pairs_df=pairs_df,
        data_root=args.data_root,
        images_per_template=args.images_per_template,
        inline_images=args.inline_images,
        seed=args.seed,
    )
    output_jsonl = Path(args.output_jsonl)
    return conversations, output_jsonl


def preview_command(args: argparse.Namespace) -> None:
    conversations, _ = prepare_training_conversations(args)
    for idx, example in enumerate(conversations[: args.num_preview], start=1):
        print(f"===== EXAMPLE {idx} =====")
        print(json.dumps({"messages": example.messages, "metadata": example.metadata}, indent=2))


def prepare_command(args: argparse.Namespace) -> None:
    conversations, output_jsonl = prepare_training_conversations(args)
    write_jsonl(output_jsonl, conversations)
    summary = {
        "output_jsonl": str(output_jsonl),
        "num_examples": len(conversations),
        "images_per_template": args.images_per_template,
        "inline_images": args.inline_images,
        "model_name": args.model_name,
    }
    print(json.dumps(summary, indent=2))


def train_command(args: argparse.Namespace) -> None:
    conversations, output_jsonl = prepare_training_conversations(args)
    write_jsonl(output_jsonl, conversations)

    try:
        import asyncio

        import chz
        from tinker_cookbook import cli_utils, model_info
        from tinker_cookbook.recipes import sl_basic  # noqa: F401  # Keep import path honest.
        from tinker_cookbook.renderers import TrainOnWhat
        from tinker_cookbook.supervised import train as tinker_train
        from tinker_cookbook.supervised.data import FromConversationFileBuilder
        from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
    except ImportError as exc:
        raise SystemExit(
            "Tinker dependencies are not installed. Install with:\n"
            "  uv pip install tinker tinker-cookbook marimo chz\n"
            f"Prepared dataset is still available at: {output_jsonl}\n"
            f"Import failure: {exc}"
        ) from exc

    renderer_name = args.renderer_name or model_info.get_recommended_renderer_name(args.model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=args.model_name,
        renderer_name=renderer_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=str(output_jsonl),
        test_size=0,
    )

    blueprint = chz.Blueprint(tinker_train.Config).apply(
        {
            "log_path": args.log_path,
            "model_name": args.model_name,
            "renderer_name": renderer_name,
            "dataset_builder": dataset_builder,
            "learning_rate": args.learning_rate,
            "lr_schedule": "linear",
            "num_epochs": args.num_epochs,
            "eval_every": args.eval_every,
            "lora_rank": args.lora_rank,
        }
    )
    config = blueprint.make()
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="overwrite")
    asyncio.run(tinker_train.main(config))


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_root", type=str, required=True, help="Dataset root, e.g. ./datasets/dataset_a")
    parser.add_argument(
        "--train_metadata",
        type=str,
        default=None,
        help="Training metadata parquet. Defaults to splits/val_15_seed42/train_metadata.parquet.",
    )
    parser.add_argument(
        "--train_pairs",
        type=str,
        default=None,
        help=(
            "Training pairs parquet. Defaults to splits/val_15_seed42/train_pairs.parquet when present. "
            "If omitted and missing, pairs are sampled from component_id labels."
        ),
    )
    parser.add_argument("--output_jsonl", type=str, default="datasets/qwen_face_train.jsonl", help="Output JSONL path")
    parser.add_argument("--max_examples", type=int, default=4096, help="Maximum number of supervised pair examples")
    parser.add_argument("--images_per_template", type=int, default=2, help="Maximum images to include per template")
    parser.add_argument("--inline_images", action="store_true", help="Inline images as data URIs in the JSONL output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="Base model name for Tinker fine-tuning",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or launch Qwen3-VL Tinker fine-tuning for face verification")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="Preview a few generated face-verification conversations")
    add_common_args(preview_parser)
    preview_parser.add_argument("--num_preview", type=int, default=2, help="Number of examples to print")
    preview_parser.set_defaults(func=preview_command)

    prepare_parser = subparsers.add_parser("prepare", help="Export a JSONL SFT dataset for Qwen3-VL")
    add_common_args(prepare_parser)
    prepare_parser.set_defaults(func=prepare_command)

    train_parser = subparsers.add_parser("train", help="Prepare JSONL and launch Tinker Cookbook SFT")
    add_common_args(train_parser)
    train_parser.add_argument("--renderer_name", type=str, default=None, help="Override renderer name")
    train_parser.add_argument("--log_path", type=str, default="./logs/qwen_face_tinker", help="Tinker log directory")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Training batch size for the cookbook dataset")
    train_parser.add_argument("--learning_rate", type=float, default=2e-4, help="Initial learning rate")
    train_parser.add_argument("--num_epochs", type=int, default=1, help="Number of supervised epochs")
    train_parser.add_argument("--eval_every", type=int, default=50, help="Periodic evaluation cadence in steps")
    train_parser.add_argument("--lora_rank", type=int, default=32, help="LoRA rank")
    train_parser.add_argument("--max_length", type=int, default=32768, help="Maximum rendered sequence length")
    train_parser.set_defaults(func=train_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
