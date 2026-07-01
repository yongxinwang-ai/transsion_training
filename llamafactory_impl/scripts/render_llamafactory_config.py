#!/usr/bin/env python3
"""Render a LLaMA-Factory YAML config from a small placeholder template."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    values = {
        "MODEL_NAME_OR_PATH": env("MODEL_NAME_OR_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
        "DATASET_NAME": env("DATASET_NAME", "transsion_math_sft"),
        "DATASET_DIR": env("DATASET_DIR", str(Path(env("LLAMA_FACTORY_DIR", ".")).resolve() / "data")),
        "CUTOFF_LEN": env("CUTOFF_LEN", "32768"),
        "PREPROCESSING_NUM_WORKERS": env("PREPROCESSING_NUM_WORKERS", "16"),
        "PACKING": env("PACKING", "true"),
        "OUTPUT_DIR": env("OUTPUT_DIR", str(Path.cwd() / "outputs" / "qwen3vl-8b-transsion-sft")),
        "LOGGING_DIR": env("LOGGING_DIR", str(Path(env("OUTPUT_DIR", str(Path.cwd() / "outputs" / "qwen3vl-8b-transsion-sft"))) / "logs")),
        "LOGGING_STEPS": env("LOGGING_STEPS", "1"),
        "SAVE_STEPS": env("SAVE_STEPS", "500"),
        "SAVE_TOTAL_LIMIT": env("SAVE_TOTAL_LIMIT", "1"),
        "REPORT_TO": env("REPORT_TO", "none"),
        "RUN_NAME": env("RUN_NAME", "qwen3vl-8b-transsion-sft"),
        "PER_DEVICE_TRAIN_BATCH_SIZE": env("PER_DEVICE_TRAIN_BATCH_SIZE", "1"),
        "GRADIENT_ACCUMULATION_STEPS": env("GRADIENT_ACCUMULATION_STEPS", "32"),
        "LEARNING_RATE": env("LEARNING_RATE", "1.0e-5"),
        "NUM_TRAIN_EPOCHS": env("NUM_TRAIN_EPOCHS", "3"),
        "WARMUP_RATIO": env("WARMUP_RATIO", "0.03"),
        "WEIGHT_DECAY": env("WEIGHT_DECAY", "0.0"),
        "IMAGE_MAX_PIXELS": env("IMAGE_MAX_PIXELS", "589824"),
        "IMAGE_MIN_PIXELS": env("IMAGE_MIN_PIXELS", "1024"),
        "ENABLE_LIGER_KERNEL": env("ENABLE_LIGER_KERNEL", "true"),
        "FLASH_ATTN": env("FLASH_ATTN", "fa2"),
        "DEEPSPEED_CONFIG": env("DEEPSPEED_CONFIG", "examples/deepspeed/ds_z3_config.json"),
    }

    text = Path(args.template).read_text()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
