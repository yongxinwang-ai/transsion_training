#!/usr/bin/env python3
"""Export a math-focused PVRD-SG dataset for LlamaFactory.

This keeps the same answer supervision as PVRD no-crop, but augments prompt-in-image
examples with training-only semantic grounding metadata:
- sg_enabled
- sg_crop_images
- sg_full_images
- sg_prompt_box
- sg_prompt_image_size
- sg_text_embedding_path
"""

from __future__ import annotations

import argparse
import json
import random
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_LLAMAFACTORY_DATA_DIR = Path("/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data")
MINIMAL_PROMPT = "Help me solve the problem"
DEFAULT_ALLOW_SOURCES = (
    "MMR1",
    "BMMR",
    "Euclid30K",
    "MMK12",
    "FineVision-geo170k(qa)",
    "FineVision-geometry3k(mathv360k)",
    "WeMath2-Pro",
    "WeMath2-Standard",
    "WeMath2-SFT",
    "mmopenr1-8k",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PVRD-SG dataset for LlamaFactory")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_LLAMAFACTORY_DATA_DIR)
    parser.add_argument("--dataset-name", type=str, default="pvrd_sg_math_thinking_sft")
    parser.add_argument("--max-prompt-in-image", type=int, default=8000)
    parser.add_argument("--max-original", type=int, default=2000)
    parser.add_argument("--max-gold-chars", type=int, default=256)
    parser.add_argument("--max-think-chars", type=int, default=1536)
    parser.add_argument("--short-think-chars", type=int, default=192)
    parser.add_argument("--prompt-target-mode", type=str, choices=["think_answer", "answer_only"], default="think_answer")
    parser.add_argument("--original-target-mode", type=str, choices=["think_answer", "answer_only"], default="answer_only")
    parser.add_argument("--include-pvrd-distill-meta", action="store_true")
    parser.add_argument("--include-cvsa-meta", action="store_true")
    parser.add_argument("--prompt-answer-only-kinds", type=str, default="")
    parser.add_argument("--prompt-short-think-kinds", type=str, default="single_letter,number,boolean,short_text")
    parser.add_argument("--allow-sources", type=str, default=",".join(DEFAULT_ALLOW_SOURCES))
    parser.add_argument("--view-block-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def sanitize_prompt_text(text: Any) -> str:
    value = normalize_text(text)
    value = re.sub(r"</?image>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"</?video>", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip()


def build_user_content(prompt_text: str, num_images: int) -> str:
    return ("<image>" * num_images) + prompt_text


def extract_tagged_span(text: str, tag: str) -> str | None:
    m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL)
    if not m:
        return None
    return normalize_text(m.group(1))


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed if trimmed else text[:max_chars].strip()


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _alias_names(dataset_name: str) -> list[str]:
    aliases = [dataset_name]
    prefix = "pvrd_sg_math_thinking_sft_"
    if dataset_name.startswith(prefix):
        aliases.append("pvrd_sg_math_thinking_sft")
    return aliases


def upsert_dataset_info(dataset_info_path: Path, dataset_name: str, dataset_file_name: str) -> None:
    if dataset_info_path.exists():
        data = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    else:
        data = {}

    entry = {
        "file_name": dataset_file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
            "system": "system",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    for name in _alias_names(dataset_name):
        data[name] = entry
    dataset_info_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_rows(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    if n >= len(rows):
        copied = list(rows)
        rng.shuffle(copied)
        return copied
    return rng.sample(rows, n)


def interleave_view_blocks(
    prompt_examples: List[Dict[str, Any]],
    original_examples: List[Dict[str, Any]],
    *,
    block_size: int,
) -> List[Dict[str, Any]]:
    if block_size <= 0:
        return prompt_examples + original_examples

    ordered: List[Dict[str, Any]] = []
    prompt_index = 0
    original_index = 0

    while prompt_index < len(prompt_examples) or original_index < len(original_examples):
        if prompt_index < len(prompt_examples):
            ordered.extend(prompt_examples[prompt_index : prompt_index + block_size])
            prompt_index += block_size
        if original_index < len(original_examples):
            ordered.extend(original_examples[original_index : original_index + block_size])
            original_index += block_size

    return ordered


def parse_allow_sources(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_kind_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def classify_gold_kind(row: Dict[str, Any]) -> str:
    gold = normalize_text(row.get("gold_canonical", ""))
    if re.fullmatch(r"[A-Z]", gold):
        return "single_letter"
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", gold):
        return "number"
    if gold.lower() in {"yes", "no", "true", "false"}:
        return "boolean"
    if len(gold.split()) <= 2 and len(gold) <= 12:
        return "short_text"
    return "long_text"


def build_prompt_in_image_target(
    row: Dict[str, Any],
    max_think_chars: int,
    short_think_chars: int,
    answer_only_kinds: set[str],
    short_think_kinds: set[str],
    prompt_target_mode: str,
) -> str:
    gold = normalize_text(row.get("gold_canonical", ""))
    if prompt_target_mode == "answer_only":
        return gold
    teacher = normalize_text(row.get("teacher_response", ""))
    kind = classify_gold_kind(row)
    if kind in answer_only_kinds:
        return gold
    think = extract_tagged_span(teacher, "think")
    budget = short_think_chars if kind in short_think_kinds else max_think_chars
    if think is None:
        think = truncate_text(teacher, budget)
    else:
        think = truncate_text(think, budget)
    return f"<think>{think}</think><answer>{gold}</answer>"


def build_gold_only_target(row: Dict[str, Any]) -> str:
    gold = normalize_text(row.get("gold_canonical", ""))
    return gold


def build_original_target(
    row: Dict[str, Any],
    original_target_mode: str,
    max_think_chars: int,
) -> str:
    if original_target_mode == "answer_only":
        return build_gold_only_target(row)
    teacher = normalize_text(row.get("teacher_response", ""))
    gold = normalize_text(row.get("gold_canonical", ""))
    think = extract_tagged_span(teacher, "think")
    if think is None:
        think = truncate_text(teacher, max_think_chars)
    else:
        think = truncate_text(think, max_think_chars)
    return f"<think>{think}</think><answer>{gold}</answer>"


def embedding_path_for(row: Dict[str, Any], embedding_dir: Path) -> Path:
    return embedding_dir / f"{row['sample_id']}.pt"


def read_png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        if f.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"expected PNG file at {path}")
        _ = struct.unpack(">I", f.read(4))[0]
        if f.read(4) != b"IHDR":
            raise ValueError(f"missing IHDR in {path}")
        width, height = struct.unpack(">II", f.read(8))
    return width, height


def build_example(
    row: Dict[str, Any],
    view: str,
    max_think_chars: int,
    short_think_chars: int,
    answer_only_kinds: set[str],
    short_think_kinds: set[str],
    embedding_dir: Path,
    prompt_target_mode: str,
    original_target_mode: str,
    include_pvrd_distill_meta: bool,
    include_cvsa_meta: bool,
) -> Dict[str, Any]:
    if view == "original":
        prompt_text = sanitize_prompt_text(row["original_prompt"])
        image_paths = list(row["original_image_paths"])
        assistant_text = build_original_target(
            row,
            original_target_mode,
            max_think_chars,
        )
        sg_enabled = False
        sg_crop_images: list[str] = []
        sg_full_images: list[str] = []
        sg_prompt_box: list[int] = []
        sg_prompt_image_size: list[int] = []
        sg_text_embedding_path = ""
        pvrd_distill_enabled = False
        pvrd_sample_id = ""
        cvsa_enabled = False
        cvsa_sample_id = ""
        cvsa_original_prompt = ""
        cvsa_original_images: list[str] = []
        cvsa_crop_images: list[str] = []
        cvsa_answer_text = ""
        cvsa_prompt_box: list[int] = []
        cvsa_prompt_image_size: list[int] = []
    else:
        prompt_text = sanitize_prompt_text(row["prompt_in_image_prompt"])
        image_paths = list(row["prompt_in_image_image_paths"])
        assistant_text = build_prompt_in_image_target(
            row,
            max_think_chars,
            short_think_chars,
            answer_only_kinds,
            short_think_kinds,
            prompt_target_mode,
        )
        sg_enabled = True
        sg_crop_images = list(row.get("prompt_in_image_panel_crop_image_paths") or [])
        sg_full_images = list(image_paths)
        sg_prompt_box = list(row.get("prompt_in_image_panel_crop_box") or [])
        sg_prompt_image_size = list(read_png_size(Path(image_paths[0]))) if image_paths else []
        sg_text_embedding_path = str(embedding_path_for(row, embedding_dir))
        pvrd_distill_enabled = include_pvrd_distill_meta
        pvrd_sample_id = row["sample_id"] if include_pvrd_distill_meta else ""
        cvsa_enabled = include_cvsa_meta
        cvsa_sample_id = row["sample_id"] if include_cvsa_meta else ""
        cvsa_original_prompt = sanitize_prompt_text(row["original_prompt"]) if include_cvsa_meta else ""
        cvsa_original_images = list(row["original_image_paths"]) if include_cvsa_meta else []
        cvsa_crop_images = list(row.get("prompt_in_image_panel_crop_image_paths") or []) if include_cvsa_meta else []
        cvsa_answer_text = build_gold_only_target(row) if include_cvsa_meta else ""
        cvsa_prompt_box = list(row.get("prompt_in_image_panel_crop_box") or []) if include_cvsa_meta else []
        cvsa_prompt_image_size = (
            list(read_png_size(Path(image_paths[0]))) if include_cvsa_meta and image_paths else []
        )

    return {
        "messages": [
            {"role": "user", "content": build_user_content(prompt_text, len(image_paths))},
            {"role": "assistant", "content": assistant_text},
        ],
        "images": image_paths,
        "system": "",
        "sg_enabled": sg_enabled,
        "sg_sample_id": row["sample_id"],
        "sg_crop_images": sg_crop_images,
        "sg_full_images": sg_full_images,
        "sg_prompt_box": sg_prompt_box,
        "sg_prompt_image_size": sg_prompt_image_size,
        "sg_text_embedding_path": sg_text_embedding_path,
        "pvrd_distill_enabled": pvrd_distill_enabled,
        "pvrd_sample_id": pvrd_sample_id,
        "cvsa_enabled": cvsa_enabled,
        "cvsa_sample_id": cvsa_sample_id,
        "cvsa_original_prompt": cvsa_original_prompt,
        "cvsa_original_images": cvsa_original_images,
        "cvsa_crop_images": cvsa_crop_images,
        "cvsa_answer_text": cvsa_answer_text,
        "cvsa_prompt_box": cvsa_prompt_box,
        "cvsa_prompt_image_size": cvsa_prompt_image_size,
    }


def is_teacher_usable(row: Dict[str, Any]) -> bool:
    teacher = normalize_text(row.get("teacher_response", ""))
    return bool(teacher) and (extract_tagged_span(teacher, "think") is not None)


def is_gold_usable(row: Dict[str, Any], max_gold_chars: int) -> bool:
    gold = normalize_text(row.get("gold_canonical", ""))
    return bool(gold) and len(gold) <= max_gold_chars


def has_grounding_assets(row: Dict[str, Any], embedding_dir: Path) -> bool:
    crop_images = row.get("prompt_in_image_panel_crop_image_paths") or []
    if not crop_images:
        return False
    return embedding_path_for(row, embedding_dir).is_file()


def main() -> None:
    args = parse_args()
    args.dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = args.dataset_dir / f"{args.dataset_name}.json"
    dataset_info_path = args.dataset_dir / "dataset_info.json"
    summary_path = args.dataset_dir / f"{args.dataset_name}_summary.json"
    if dataset_file.exists() and not args.overwrite:
        raise FileExistsError(f"{dataset_file} already exists, pass --overwrite to replace it")

    allow_sources = parse_allow_sources(args.allow_sources)
    answer_only_kinds = parse_kind_set(args.prompt_answer_only_kinds)
    short_think_kinds = parse_kind_set(args.prompt_short_think_kinds)
    rows = load_manifest(args.manifest)

    source_counter = Counter()
    allowed_rows: List[Dict[str, Any]] = []
    for row in rows:
        source = normalize_text(row.get("source", ""))
        if source not in allow_sources:
            continue
        source_counter[source] += 1
        if not is_gold_usable(row, args.max_gold_chars):
            continue
        allowed_rows.append(row)

    needs_teacher = args.prompt_target_mode != "answer_only" or args.original_target_mode != "answer_only"
    if needs_teacher:
        teacher_rows = [row for row in allowed_rows if is_teacher_usable(row)]
    else:
        teacher_rows = list(allowed_rows)
    grounded_rows = [row for row in teacher_rows if has_grounding_assets(row, args.embedding_dir)]

    rng = random.Random(args.seed)
    prompt_examples: List[Dict[str, Any]] = []
    original_examples: List[Dict[str, Any]] = []
    exported_by_view = Counter()

    for row in sample_rows(grounded_rows, args.max_prompt_in_image, rng):
        prompt_examples.append(
            build_example(
                row,
                "prompt_in_image",
                args.max_think_chars,
                args.short_think_chars,
                answer_only_kinds,
                short_think_kinds,
                args.embedding_dir,
                args.prompt_target_mode,
                args.original_target_mode,
                args.include_pvrd_distill_meta,
                args.include_cvsa_meta,
            )
        )
        exported_by_view["prompt_in_image"] += 1
    original_source_rows = teacher_rows if args.original_target_mode != "answer_only" else allowed_rows
    for row in sample_rows(original_source_rows, args.max_original, rng):
        original_examples.append(
            build_example(
                row,
                "original",
                args.max_think_chars,
                args.short_think_chars,
                answer_only_kinds,
                short_think_kinds,
                args.embedding_dir,
                args.prompt_target_mode,
                args.original_target_mode,
                args.include_pvrd_distill_meta,
                args.include_cvsa_meta,
            )
        )
        exported_by_view["original"] += 1

    if args.view_block_size > 0:
        examples = interleave_view_blocks(
            prompt_examples,
            original_examples,
            block_size=args.view_block_size,
        )
    else:
        examples = prompt_examples + original_examples
        rng.shuffle(examples)
    dataset_file.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    upsert_dataset_info(dataset_info_path, args.dataset_name, dataset_file.name)

    summary = {
        "manifest": str(args.manifest),
        "embedding_dir": str(args.embedding_dir),
        "dataset_name": args.dataset_name,
        "dataset_file": str(dataset_file),
        "num_rows_seen": len(rows),
        "num_allowed_sources": sum(source_counter.values()),
        "num_gold_eligible": len(allowed_rows),
        "num_teacher_eligible": len(teacher_rows),
        "num_grounding_eligible": len(grounded_rows),
        "allow_sources": sorted(allow_sources),
        "source_counter": dict(source_counter),
        "max_prompt_in_image": args.max_prompt_in_image,
        "max_original": args.max_original,
        "max_gold_chars": args.max_gold_chars,
        "max_think_chars": args.max_think_chars,
        "short_think_chars": args.short_think_chars,
        "prompt_target_mode": args.prompt_target_mode,
        "original_target_mode": args.original_target_mode,
        "include_pvrd_distill_meta": args.include_pvrd_distill_meta,
        "include_cvsa_meta": args.include_cvsa_meta,
        "prompt_answer_only_kinds": sorted(answer_only_kinds),
        "prompt_short_think_kinds": sorted(short_think_kinds),
        "view_block_size": args.view_block_size,
        "num_exported": len(examples),
        "exported_by_view": dict(exported_by_view),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
