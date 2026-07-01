#!/usr/bin/env python3
"""Register a Transsion multimodal JSONL dataset in LLaMA-Factory dataset_info.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llamafactory-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-jsonl", required=True)
    args = parser.parse_args()

    lf_dir = Path(args.llamafactory_dir).resolve()
    dataset_jsonl = Path(args.dataset_jsonl).resolve()
    if not dataset_jsonl.exists():
        raise FileNotFoundError(f"DATASET_JSONL does not exist: {dataset_jsonl}")

    info_path = lf_dir / "data" / "dataset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"dataset_info.json not found: {info_path}")

    info = json.loads(info_path.read_text())
    info[args.dataset_name] = {
        "file_name": str(dataset_jsonl),
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    print(f"Registered {args.dataset_name} -> {dataset_jsonl}")


if __name__ == "__main__":
    main()
