#!/usr/bin/env python3
"""Generate face-verification prediction CSVs with a Tinker-hosted Qwen3-VL checkpoint.

This preserves the repository's prediction interface:

    template_id_1,template_id_2,score

Unlike the embedding baselines, scoring is performed by querying a multimodal
Qwen3-VL checkpoint through Tinker's native sampling client plus the same
renderer family used during fine-tuning.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm


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
    if path.exists() or path.is_absolute():
        return path
    if path.parts[: len(root.parts)] == root.parts:
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


def build_template_messages(template_a_images: list[Path], template_b_images: list[Path]) -> list[dict[str, Any]]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": "Template A face images:"},
    ]
    for image_path in template_a_images:
        user_content.append({"type": "image", "image": image_to_data_uri(image_path)})

    user_content.append({"type": "text", "text": "Template B face images:"})
    for image_path in template_b_images:
        user_content.append({"type": "image", "image": image_to_data_uri(image_path)})

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


def build_image_pair_messages(left_image: Path, right_image: Path) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image A:"},
                {"type": "image", "image": image_to_data_uri(left_image)},
                {"type": "text", "text": "Image B:"},
                {"type": "image", "image": image_to_data_uri(right_image)},
                {
                    "type": "text",
                    "text": (
                        "Do these two face images belong to the same identity? "
                        "Respond with exactly one lowercase word: same or different."
                    ),
                },
            ],
        },
    ]


def extract_message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    parts.append(part)
            return "".join(parts)
    if isinstance(message, list):
        return "".join(extract_message_text(part) for part in message)
    return str(message)


def build_sampling_stack(model_path: str, api_key: str) -> tuple[Any, Any, Any, str]:
    try:
        import tinker
        from tinker.types import SamplingParams
        from tinker_cookbook import model_info, renderers, tokenizer_utils
        from tinker_cookbook.image_processing_utils import get_image_processor
    except ImportError as exc:
        raise SystemExit(
            "Tinker inference dependencies are not installed. Install with:\n"
            "  pip install tinker tinker-cookbook\n"
            f"Import failure: {exc}"
        ) from exc

    os.environ.setdefault("TINKER_API_KEY", api_key)

    service_client = tinker.ServiceClient()
    rest_client = service_client.create_rest_client()
    weights_info = rest_client.get_weights_info_by_tinker_path(model_path).result()
    base_model = weights_info.base_model
    if not base_model:
        raise ValueError(f"Could not determine base model for checkpoint: {model_path}")

    tokenizer = tokenizer_utils.get_tokenizer(base_model)
    image_processor = get_image_processor(base_model)
    renderer_name = model_info.get_recommended_renderer_name(base_model)
    renderer = renderers.get_renderer(
        renderer_name,
        tokenizer,
        image_processor=image_processor,
        model_name=base_model,
    )
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )
    return sampling_client, renderer, sampling_params, base_model


def request_label(
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    messages: list[dict[str, Any]],
    max_retries: int,
    sleep_seconds: float,
) -> tuple[str | None, str]:
    last_text = ""
    for _ in range(max_retries):
        prompt = renderer.build_generation_prompt(messages)
        output = sampling_client.sample(
            prompt,
            sampling_params=sampling_params,
            num_samples=1,
        ).result()
        sampled_message, parse_success = renderer.parse_response(output.sequences[0].tokens)
        raw_text = extract_message_text(sampled_message).strip()
        last_text = raw_text
        if parse_success:
            label = extract_label(raw_text)
            if label is not None:
                return label, raw_text
        else:
            label = extract_label(raw_text)
            if label is not None:
                return label, raw_text
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return None, last_text


def iter_image_pairs(template_a_images: Iterable[Path], template_b_images: Iterable[Path]) -> Iterable[tuple[Path, Path]]:
    for left_image in template_a_images:
        for right_image in template_b_images:
            yield left_image, right_image


def score_template_pair(
    sampling_client: Any,
    renderer: Any,
    sampling_params: Any,
    template_a_images: list[Path],
    template_b_images: list[Path],
    compare_mode: str,
    unknown_score: float,
    max_retries: int,
    sleep_seconds: float,
) -> tuple[float, dict[str, Any]]:
    if compare_mode == "template":
        label, raw_text = request_label(
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
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
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
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
        help="Tinker sampler checkpoint path, e.g. tinker://.../sampler_weights/final",
    )
    parser.add_argument("--api_key", type=str, default=None, help="Tinker API key. Defaults to TINKER_API_KEY env var.")
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
    sampling_client, renderer, sampling_params, base_model = build_sampling_stack(args.model_path, api_key)

    print(f"Dataset root: {dataset_root}")
    print(f"Metadata: {metadata_path}")
    print(f"Pairs: {pairs_path}")
    print(f"Model path: {args.model_path}")
    print(f"Base model: {base_model}")
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
            sampling_client=sampling_client,
            renderer=renderer,
            sampling_params=sampling_params,
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
