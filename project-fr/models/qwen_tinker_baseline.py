#!/usr/bin/env python3
"""Generate face-verification prediction CSVs with a Tinker-hosted Qwen3-VL checkpoint.

This script preserves the repository's prediction interface:

    template_id_1,template_id_2,score

Unlike the embedding baselines, scoring is performed by querying a multimodal
Qwen3-VL checkpoint through Tinker's OpenAI-compatible inference endpoint.

Practical note:
    This is much slower than the ResNet50 or MobileFaceNet paths. It is mainly
    intended for smaller validation runs, smoke tests, and early capability
    checks rather than full multi-million-pair benchmark sweeps.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import mimetypes
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm


BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
SYSTEM_PROMPT = (
    "You are a face verification model. Determine whether the provided face images "
    "belong to the same identity. Respond with exactly one lowercase word: same or different."
)
LABEL_RE = re.compile(r"\b(same|different)\b", re.IGNORECASE)


def resolve_path(root: str | Path, value: str | None, default_name: str | None = None) -> Path:
    root = Path(root)
    if value is None:
        if default_name is None:
            raise ValueError("A path value or default_name is required.")
        return root / default_name

    path = Path(value)
    if path.exists():
        return path
    return root / value


def image_to_data_uri(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_template_to_images(metadata_df: pd.DataFrame, dataset_root: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for row in metadata_df.itertuples(index=False):
        grouped[int(row.template_id)].append(dataset_root / row.image_path)
    return {template_id: sorted(paths) for template_id, paths in grouped.items()}


def select_template_images(template_to_images: dict[int, list[Path]], template_id: int, max_images: int) -> list[Path]:
    images = template_to_images.get(int(template_id), [])
    if not images:
        raise KeyError(f"No images found for template_id={template_id}")
    return images[:max_images]


def extract_label(text: str) -> str | None:
    match = LABEL_RE.search(text.strip())
    if not match:
        return None
    return match.group(1).lower()


def label_to_score(label: str | None, unknown_score: float) -> float:
    if label == "same":
        return 1.0
    if label == "different":
        return 0.0
    return unknown_score


def build_template_messages(template_a_images: list[Path], template_b_images: list[Path]) -> list[dict]:
    user_content: list[dict] = [
        {"type": "text", "text": "Template A face images:"},
    ]
    for image_path in template_a_images:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})

    user_content.append({"type": "text", "text": "Template B face images:"})
    for image_path in template_b_images:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})

    user_content.append(
        {
            "type": "text",
            "text": (
                "Do template A and template B belong to the same identity? "
                "Respond with exactly one lowercase word: same or different."
            ),
        }
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def build_image_pair_messages(left_image: Path, right_image: Path) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A:"},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(left_image)}},
                {"type": "text", "text": "Image B:"},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(right_image)}},
                {
                    "type": "text",
                    "text": "Do these two face images belong to the same identity? Respond with exactly one lowercase word: same or different.",
                },
            ],
        },
    ]


def request_label(
    client: Any,
    model_path: str,
    messages: list[dict],
    max_retries: int,
    sleep_seconds: float,
) -> tuple[str | None, str]:
    last_text = ""
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model=model_path,
            messages=messages,
            max_tokens=4,
            temperature=0.0,
            top_p=1.0,
        )
        text = response.choices[0].message.content or ""
        if isinstance(text, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in text
            )
        last_text = str(text).strip()
        label = extract_label(last_text)
        if label is not None:
            return label, last_text
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return None, last_text


def iter_image_pairs(template_a_images: Iterable[Path], template_b_images: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for left_image in template_a_images:
        for right_image in template_b_images:
            yield left_image, right_image


def score_template_pair(
    client: Any,
    model_path: str,
    template_a_images: list[Path],
    template_b_images: list[Path],
    compare_mode: str,
    unknown_score: float,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[float, dict]:
    if compare_mode == "template":
        label, raw_text = request_label(
            client=client,
            model_path=model_path,
            messages=build_template_messages(template_a_images, template_b_images),
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        return label_to_score(label, unknown_score), {
            "mode": "template",
            "raw_text": raw_text,
            "label": label,
        }

    image_pair_scores = []
    raw_outputs = []
    for left_image, right_image in iter_image_pairs(template_a_images, template_b_images):
        label, raw_text = request_label(
            client=client,
            model_path=model_path,
            messages=build_image_pair_messages(left_image, right_image),
            max_retries=max_retries,
            sleep_seconds=sleep_seconds,
        )
        image_pair_scores.append(label_to_score(label, unknown_score))
        raw_outputs.append(
            {
                "left_image": str(left_image),
                "right_image": str(right_image),
                "label": label,
                "raw_text": raw_text,
            }
        )

    if not image_pair_scores:
        return unknown_score, {"mode": "image_pairs", "raw_outputs": []}

    return sum(image_pair_scores) / len(image_pair_scores), {
        "mode": "image_pairs",
        "raw_outputs": raw_outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate face-verification prediction CSVs with a Tinker Qwen3-VL checkpoint")
    parser.add_argument("--dataset_root", type=str, required=True, help="Dataset directory, e.g. ./datasets/dataset_a")
    parser.add_argument("--output", type=str, required=True, help="Output prediction CSV path")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Tinker sampler checkpoint path, e.g. tinker://.../sampler_weights/000080",
    )
    parser.add_argument("--api_key", type=str, default=None, help="Tinker API key. Defaults to TINKER_API_KEY env var.")
    parser.add_argument("--base_url", type=str, default=BASE_URL, help="OpenAI-compatible Tinker base URL")
    parser.add_argument("--metadata", type=str, default=None, help="Evaluation metadata parquet. Defaults to test.parquet")
    parser.add_argument("--pairs", type=str, default=None, help="Pairs parquet. Defaults to pairs.parquet")
    parser.add_argument("--images_per_template", type=int, default=2, help="Images to include per template")
    parser.add_argument(
        "--compare_mode",
        type=str,
        default="template",
        choices=["template", "image_pairs"],
        help="Template-level query or average over all selected image pairs",
    )
    parser.add_argument("--unknown_score", type=float, default=0.5, help="Fallback score when the label cannot be parsed")
    parser.add_argument("--max_retries", type=int, default=2, help="Retries when the model output is unparsable")
    parser.add_argument("--sleep_seconds", type=float, default=0.0, help="Optional delay between retries")
    parser.add_argument("--limit_pairs", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--debug_jsonl", type=str, default=None, help="Optional JSONL file for raw model outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The 'openai' package is required for Qwen/Tinker inference. "
            "Install it with: pip install openai"
        ) from exc

    api_key = args.api_key or os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set TINKER_API_KEY in the environment.")

    dataset_root = Path(args.dataset_root)
    metadata_path = resolve_path(dataset_root, args.metadata, "test.parquet")
    pairs_path = resolve_path(dataset_root, args.pairs, "pairs.parquet")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.read_parquet(metadata_path)
    pairs_df = pd.read_parquet(pairs_path)
    if args.limit_pairs is not None:
        pairs_df = pairs_df.head(args.limit_pairs).copy()

    template_to_images = build_template_to_images(metadata_df, dataset_root)
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    print(f"Dataset root: {dataset_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Pairs: {pairs_path}")
    print(f"Model path: {args.model_path}")
    print(f"Compare mode: {args.compare_mode}")
    print(f"Pairs to score: {len(pairs_df)}")

    rows = []
    debug_records = []
    for row in tqdm(pairs_df.itertuples(index=False), total=len(pairs_df), desc="Scoring pairs"):
        template_id_1 = int(row.template_id_1)
        template_id_2 = int(row.template_id_2)

        template_a_images = select_template_images(template_to_images, template_id_1, args.images_per_template)
        template_b_images = select_template_images(template_to_images, template_id_2, args.images_per_template)
        score, debug_info = score_template_pair(
            client=client,
            model_path=args.model_path,
            template_a_images=template_a_images,
            template_b_images=template_b_images,
            compare_mode=args.compare_mode,
            unknown_score=args.unknown_score,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        rows.append(
            {
                "template_id_1": template_id_1,
                "template_id_2": template_id_2,
                "score": float(score),
            }
        )
        if args.debug_jsonl:
            debug_records.append(
                {
                    "template_id_1": template_id_1,
                    "template_id_2": template_id_2,
                    "score": float(score),
                    "template_a_images": [str(path) for path in template_a_images],
                    "template_b_images": [str(path) for path in template_b_images],
                    "debug": debug_info,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path} ({len(out_df)} pairs)")

    if args.debug_jsonl:
        debug_path = Path(args.debug_jsonl)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8") as handle:
            for record in debug_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Debug records saved to: {debug_path}")


if __name__ == "__main__":
    main()
