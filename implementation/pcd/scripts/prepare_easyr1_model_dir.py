#!/usr/bin/env python3
"""Create an EasyR1-compatible model directory without copying model weights."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

DEFAULT_QWEN3VL4B_INSTRUCT = Path(
    "/mnt/weka/home/yongxin.wang/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-4B-Instruct/snapshots/"
    "ebb281ec70b05090aa6165b016eac8ec08e71b17"
)

SUPPLEMENTAL_FILENAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
    "chat_template.jinja",
    "generation_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a symlinked EasyR1-compatible model directory")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fallback-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory to use for missing processor/tokenizer sidecar files. Can be passed multiple times.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def patch_tokenizer_config(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    extra = data.get("extra_special_tokens")
    if isinstance(extra, list):
        existing = data.get("additional_special_tokens") or []
        merged = list(dict.fromkeys([*existing, *extra]))
        data["additional_special_tokens"] = merged
        data["extra_special_tokens"] = {}
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def patch_model_config(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))
    text_config = data.get("text_config")
    if isinstance(text_config, dict) and text_config.get("rope_scaling") is None:
        rope_params = text_config.get("rope_parameters")
        if isinstance(rope_params, dict) and "mrope_section" in rope_params:
            text_config["rope_scaling"] = {
                "rope_type": rope_params.get("rope_type", "default"),
                "mrope_section": rope_params["mrope_section"],
                "mrope_interleaved": rope_params.get("mrope_interleaved", True),
            }
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def fallback_dirs(source: Path, cli_dirs: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(cli_dirs)
    env_dirs = os.environ.get("EASYR1_MODEL_FALLBACK_DIRS") or os.environ.get("EASYR1_MODEL_FALLBACK_DIR")
    if env_dirs:
        candidates.extend(Path(p) for p in env_dirs.split(":") if p)
    candidates.extend([source.parent, source.parent.parent])
    if DEFAULT_QWEN3VL4B_INSTRUCT.exists():
        candidates.append(DEFAULT_QWEN3VL4B_INSTRUCT)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for item in candidates:
        resolved = item.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def supplement_missing_sidecars(output: Path, candidates: list[Path]) -> None:
    for name in SUPPLEMENTAL_FILENAMES:
        dst = output / name
        if dst.exists() or dst.is_symlink():
            continue
        for candidate in candidates:
            src = candidate / name
            if src.exists():
                link_or_copy(src, dst)
                break


def main() -> None:
    args = parse_args()
    source = args.source_model.resolve()
    output = args.output_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("output-dir must differ from source-model")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for item in source.iterdir():
        target = output / item.name
        if item.name == "tokenizer_config.json":
            patch_tokenizer_config(item, target)
        elif item.name == "config.json":
            patch_model_config(item, target)
        else:
            link_or_copy(item, target)

    supplement_missing_sidecars(output, fallback_dirs(source, args.fallback_dir))

    print(output)


if __name__ == "__main__":
    main()
