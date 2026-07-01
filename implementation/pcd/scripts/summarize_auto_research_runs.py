#!/usr/bin/env python3
"""Summarize active auto-research PVRD-SG ablation runs.

The script is intentionally file-based: it does not require Slurm access and can
be run while jobs are pending, training, evaluating, or completed.  It reports
latest training loss, checkpoint availability, and completed lmms-eval metrics.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = [
    {
        "name": "VTS loss 0.25 + PVRD-SG",
        "train_job": "1692899",
        "eval_job": "1692900",
        "train_dir": "runs/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153",
        "eval_dir": "lmms_eval_results/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153_vllm_parsefix",
        "log_glob": "logs/vtsloss_vtsw025_pvrdsg_1692899.log",
        "err_glob": "logs/vtsloss_vtsw025_pvrdsg_1692899.err",
    },
    {
        "name": "Early eval: VTS loss 0.25 @ ckpt1000",
        "train_job": "1692899",
        "eval_job": "1692946",
        "train_dir": "runs/auto_research_eval_snapshots/vtsw025_pvrdsg_checkpoint-1000_20260529_034439",
        "checkpoint": "checkpoint-1000 snapshot",
        "eval_dir": "lmms_eval_results/auto_research_vtsw025_pvrdsg_checkpoint-1000_20260529_034439_vllm_parsefix",
        "log_glob": "logs/vtsloss_vtsw025_pvrdsg_1692899.log",
        "err_glob": "logs/vtsloss_vtsw025_pvrdsg_1692899.err",
    },
    {
        "name": "VTS answer-only + PVRD-SG",
        "train_job": "1692904",
        "eval_job": "1692905",
        "train_dir": "runs/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829",
        "eval_dir": "lmms_eval_results/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829_vllm_parsefix",
        "log_glob": "logs/vtsans_vtsanswer_pvrdsg_1692904.log",
        "err_glob": "logs/vtsans_vtsanswer_pvrdsg_1692904.err",
    },
    {
        "name": "Early eval: VTS answer-only @ ckpt1000",
        "train_job": "1692904",
        "eval_job": "1692947/1693005/1693011 retry",
        "train_dir": "runs/auto_research_eval_snapshots/vtsanswer_pvrdsg_checkpoint-1000_20260529_034439",
        "checkpoint": "checkpoint-1000 snapshot",
        "eval_dir": "lmms_eval_results/auto_research_vtsanswer_pvrdsg_checkpoint-1000_20260529_034439_retry2_vllm_parsefix",
        "log_glob": "logs/vtsans_vtsanswer_pvrdsg_1692904.log",
        "err_glob": "logs/vtsans_vtsanswer_pvrdsg_1692904.err",
    },
    {
        "name": "VTS LM off + PVRD-SG",
        "train_job": "1719148",
        "eval_job": "1719149",
        "train_dir": "runs/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001",
        "eval_dir": "lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001_vllm_parsefix",
        "log_glob": "logs/vtsloss_vtslmoff_pvrdsg_1719148.log",
        "err_glob": "logs/vtsloss_vtslmoff_pvrdsg_1719148.err",
    },
    {
        "name": "Early eval: VTS LM off @ ckpt1000",
        "train_job": "1719148",
        "eval_job": "1719154",
        "train_dir": "runs/auto_research_eval_snapshots/vtslmoff_pvrdsg_checkpoint-1000_20260609_062112",
        "checkpoint": "checkpoint-1000 snapshot",
        "eval_dir": "lmms_eval_results/auto_research_vtslmoff_pvrdsg_checkpoint-1000_20260609_062112_vllm_parsefix",
        "log_glob": "logs/vtsloss_vtslmoff_pvrdsg_1719148.log",
        "err_glob": "logs/vtsloss_vtslmoff_pvrdsg_1719148.err",
    },
    {
        "name": "VTS LM off + PVRD-SG + PRMLP",
        "train_job": "pending",
        "eval_job": "pending",
        "train_glob": "runs/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_prmlp_*",
        "eval_glob": "lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_prmlp_*_vllm_parsefix",
        "log_glob": "logs/vtsloss_vtslmoff_pvrdsg_prmlp_*.log",
        "err_glob": "logs/vtsloss_vtslmoff_pvrdsg_prmlp_*.err",
    },
    {
        "name": "Original-only 24K retention control",
        "train_job": "1719307",
        "eval_job": "1719308",
        "train_glob": "runs/original_only_retention_origonly24k_sft_retention_*",
        "eval_glob": "lmms_eval_results/original_only_retention_origonly24k_sft_retention_*_vllm_parsefix",
        "log_glob": "logs/vtsloss_origonly24k_sft_retention_*.log",
        "err_glob": "logs/vtsloss_origonly24k_sft_retention_*.err",
    },
    {
        "name": "Early eval: original-only 24K @ ckpt1000",
        "train_job": "1719307",
        "eval_job": "1719386",
        "train_dir": "runs/auto_research_eval_snapshots/origonly24k_sft_retention_checkpoint-1000_20260609_072512",
        "checkpoint": "checkpoint-1000 snapshot",
        "eval_dir": "lmms_eval_results/auto_research_origonly24k_sft_retention_checkpoint-1000_20260609_072512_vllm_parsefix",
        "log_glob": "logs/vtsloss_origonly24k_sft_retention_1719307.log",
        "err_glob": "logs/vtsloss_origonly24k_sft_retention_1719307.err",
    },
    {
        "name": "Original-only 24K low-LR retention control",
        "train_job": "1719410",
        "eval_job": "1719411",
        "train_dir": "runs/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926",
        "eval_dir": "lmms_eval_results/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926_vllm_parsefix",
        "log_glob": "logs/vtsloss_origonly24k_sft_lowlr_retention_1719410.log",
        "err_glob": "logs/vtsloss_origonly24k_sft_lowlr_retention_1719410.err",
    },
    {
        "name": "Early eval: original-only 24K low-LR @ ckpt1000",
        "train_job": "1719410",
        "eval_job": "1719462",
        "train_dir": "runs/auto_research_eval_snapshots/origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858",
        "checkpoint": "checkpoint-1000 snapshot",
        "eval_dir": "lmms_eval_results/auto_research_origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858_vllm_parsefix",
        "log_glob": "logs/vtsloss_origonly24k_sft_lowlr_retention_1719410.log",
        "err_glob": "logs/vtsloss_origonly24k_sft_lowlr_retention_1719410.err",
    },
    {
        "name": "Prepared: low-LR balanced PVRD-SG + PRMLP",
        "train_job": "prepared",
        "eval_job": "prepared",
        "train_glob": "runs/pvrd_sg_lowlr_scale48k_lowlr_balanced_pvrdsg_prmlp_lam003_*",
        "eval_glob": "lmms_eval_results/pvrd_sg_lowlr_scale48k_lowlr_balanced_pvrdsg_prmlp_lam003_*_vllm_parsefix",
        "log_glob": "logs/vtsloss_lowlr_balanced_pvrdsg_prmlp_lam003_*.log",
        "err_glob": "logs/vtsloss_lowlr_balanced_pvrdsg_prmlp_lam003_*.err",
    },
    {
        "name": "Original-only 24K freeze-LM retention control",
        "train_job": "1719505",
        "eval_job": "1719506",
        "train_dir": "runs/original_only_retention_origonly24k_sft_freezelm_retention_20260609_085414",
        "eval_dir": "lmms_eval_results/original_only_retention_origonly24k_sft_freezelm_retention_20260609_085414_vllm_parsefix",
        "log_glob": "logs/vtsloss_origonly24k_sft_freezelm_retention_1719505.log",
        "err_glob": "logs/vtsloss_origonly24k_sft_freezelm_retention_1719505.err",
    },
]

TASK_FIELDS = {
    "MV O": ("mathvision_testmini", "mathvision_standard_eval,none"),
    "MV V": ("mathvision_testmini_prompt_in_image", "mathvision_standard_eval,none"),
    "Vista O": ("mathvista_testmini_cot", "llm_as_judge_eval,none"),
    "Vista V": ("mathvista_testmini_prompt_in_image", "llm_as_judge_eval,none"),
}

ERROR_PATTERNS = (
    "traceback",
    "runtimeerror",
    "cuda error",
    "out of memory",
    "oom",
    "nan",
    "invalid access",
    "exception",
)


def latest_result_json(eval_dir: Path) -> Path | None:
    if not eval_dir.exists():
        return None
    files = sorted(eval_dir.rglob("*_results.json"), key=lambda path: path.stat().st_mtime)
    return files[-1] if files else None


def latest_matching_path(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    return matches[-1] if matches else None


def resolve_path(root: Path, run: dict[str, str], key: str, glob_key: str) -> Path | None:
    if key in run:
        return root / run[key]
    if glob_key in run:
        return latest_matching_path(root, run[glob_key])
    return None


def metric(results: dict[str, Any], task: str, key: str) -> float | None:
    value = results.get(task, {}).get(key)
    return value if isinstance(value, (int, float)) else None


def fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def latest_checkpoint(train_dir: Path | None) -> str:
    if train_dir is None or not train_dir.exists():
        return "--"
    checkpoints = sorted(
        [path for path in train_dir.glob("checkpoint-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
    )
    return checkpoints[-1].name if checkpoints else "--"


def latest_loss(log_path: Path | None) -> dict[str, Any] | None:
    if log_path is None or not log_path.exists():
        return None
    last: dict[str, Any] | None = None
    with log_path.open(errors="replace") as handle:
        for line in handle:
            if "{'loss'" not in line:
                continue
            payload = line.split(":", 1)[-1].strip()
            try:
                parsed = ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                continue
            if isinstance(parsed, dict) and "loss" in parsed:
                last = parsed
    return last


def count_serious_errors(err_path: Path | None) -> int:
    if err_path is None or not err_path.exists():
        return 0
    count = 0
    with err_path.open(errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            # Avoid counting harmless informational lines containing words like
            # "error handling" from NCCL deprecation warnings.
            if "error_handling" in lowered or "async_error_handling" in lowered:
                continue
            if any(pattern in lowered for pattern in ERROR_PATTERNS):
                count += 1
    return count


def eval_metrics(result_path: Path | None) -> dict[str, float | None]:
    values = {field: None for field in TASK_FIELDS}
    if result_path is None:
        return values
    with result_path.open() as handle:
        data = json.load(handle)
    results = data.get("results", {})
    for field, (task, key) in TASK_FIELDS.items():
        values[field] = metric(results, task, key)
    return values


def summarize(root: Path, run: dict[str, str]) -> dict[str, Any]:
    train_dir = resolve_path(root, run, "train_dir", "train_glob")
    eval_dir = resolve_path(root, run, "eval_dir", "eval_glob")
    log_path = resolve_path(root, run, "log_path", "log_glob")
    err_path = resolve_path(root, run, "err_path", "err_glob")
    result_path = latest_result_json(eval_dir) if eval_dir else None
    loss = latest_loss(log_path)
    metrics = eval_metrics(result_path)
    mv_gap = None
    vista_gap = None
    if metrics["MV O"] is not None and metrics["MV V"] is not None:
        mv_gap = metrics["MV O"] - metrics["MV V"]
    if metrics["Vista O"] is not None and metrics["Vista V"] is not None:
        vista_gap = metrics["Vista O"] - metrics["Vista V"]
    avg_o = None
    avg_v = None
    avg_gap = None
    if metrics["MV O"] is not None and metrics["Vista O"] is not None:
        avg_o = (metrics["MV O"] + metrics["Vista O"]) / 2
    if metrics["MV V"] is not None and metrics["Vista V"] is not None:
        avg_v = (metrics["MV V"] + metrics["Vista V"]) / 2
    if mv_gap is not None and vista_gap is not None:
        avg_gap = (mv_gap + vista_gap) / 2

    return {
        "Method": run["name"],
        "Train job": run["train_job"],
        "Eval job": run["eval_job"],
        "Checkpoint": run.get("checkpoint") or latest_checkpoint(train_dir),
        "Loss": loss.get("loss") if loss else None,
        "Epoch": loss.get("epoch") if loss else None,
        "Grad norm": loss.get("grad_norm") if loss else None,
        "Serious errors": count_serious_errors(err_path),
        "MV O": metrics["MV O"],
        "MV V": metrics["MV V"],
        "MV G": mv_gap,
        "Vista O": metrics["Vista O"],
        "Vista V": metrics["Vista V"],
        "Vista G": vista_gap,
        "Avg O": avg_o,
        "Avg V": avg_v,
        "Avg G": avg_gap,
        "Result JSON": str(result_path) if result_path else "--",
    }


def print_markdown(rows: list[dict[str, Any]]) -> None:
    print("# Auto-Research Run Summary")
    print()
    print("| Method | Train | Eval | Ckpt | Loss | Epoch | Grad | Errs | MV O | MV V | MV G | Vista O | Vista V | Vista G | Avg O | Avg V | Avg G | Result JSON |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            "| {Method} | {Train job} | {Eval job} | {Checkpoint} | {Loss} | {Epoch} | {Grad norm} | {Serious errors} | {MV O} | {MV V} | {MV G} | {Vista O} | {Vista V} | {Vista G} | {Avg O} | {Avg V} | {Avg G} | `{Result JSON}` |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of markdown.")
    args = parser.parse_args()

    rows = [summarize(args.root, run) for run in DEFAULT_RUNS]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print_markdown(rows)


if __name__ == "__main__":
    main()
