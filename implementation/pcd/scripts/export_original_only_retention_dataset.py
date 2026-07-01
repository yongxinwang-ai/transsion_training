#!/usr/bin/env python3
"""Export original-view-only retention-control data from a paired VTS SFT file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

AUX_PREFIXES = ("sg_", "pvrd_", "cvsa_")
AUX_KEYS = {"sg_enabled", "pvrd_distill_enabled", "cvsa_enabled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data/pvrd_sg_math_thinking_sft_prmlp_scale48k_bal1to1_symthink_20260423_052441.json",
        help="Paired original/VTS SFT JSON file.",
    )
    parser.add_argument(
        "--dataset-dir",
        default="/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data",
        help="LLaMA-Factory data directory that contains dataset_info.json.",
    )
    parser.add_argument(
        "--dataset-name",
        default="pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609",
        help="Output dataset name without .json suffix.",
    )
    parser.add_argument("--max-original", type=int, default=24000)
    parser.add_argument("--update-dataset-info", action="store_true")
    return parser.parse_args()


def is_original_row(row: dict[str, Any]) -> bool:
    return not bool(row.get("sg_enabled"))


def strip_aux_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in AUX_KEYS and not any(key.startswith(prefix) for prefix in AUX_PREFIXES)
    }


def update_dataset_info(dataset_dir: Path, dataset_name: str, file_name: str) -> None:
    info_path = dataset_dir / "dataset_info.json"
    info = json.loads(info_path.read_text())
    info[dataset_name] = {
        "file_name": file_name,
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
    tmp_path = info_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    tmp_path.replace(info_path)


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    dataset_dir = Path(args.dataset_dir)
    output_path = dataset_dir / f"{args.dataset_name}.json"

    rows = json.loads(source.read_text())
    original_rows = [strip_aux_fields(row) for row in rows if is_original_row(row)]
    if args.max_original > 0:
        original_rows = original_rows[: args.max_original]

    if not original_rows:
        raise SystemExit("No original rows found.")
    if args.max_original > 0 and len(original_rows) != args.max_original:
        raise SystemExit(f"Expected {args.max_original} original rows, found {len(original_rows)}.")

    output_path.write_text(json.dumps(original_rows, ensure_ascii=False, indent=2) + "\n")
    if args.update_dataset_info:
        update_dataset_info(dataset_dir, args.dataset_name, output_path.name)

    print(json.dumps({
        "source": str(source),
        "output": str(output_path),
        "dataset_name": args.dataset_name,
        "rows": len(original_rows),
        "update_dataset_info": bool(args.update_dataset_info),
    }, indent=2))


if __name__ == "__main__":
    main()
