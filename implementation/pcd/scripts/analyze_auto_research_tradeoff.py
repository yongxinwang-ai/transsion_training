#!/usr/bin/env python3
"""Analyze active VTS auto-research evals against fixed baselines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE = {
    "MV O": 51.6,
    "MV V": 34.8,
    "Vista O": 73.7,
    "Vista V": 61.9,
}

KNOWN_GOOD = {
    "MV O": 52.4,
    "MV V": 49.8,
    "Vista O": 74.2,
    "Vista V": 70.8,
}

RUNS = [
    ("vtsw025@ckpt1000", "lmms_eval_results/auto_research_vtsw025_pvrdsg_checkpoint-1000_20260529_034439_vllm_parsefix"),
    ("vtsanswer@ckpt1000", "lmms_eval_results/auto_research_vtsanswer_pvrdsg_checkpoint-1000_20260529_034439_retry2_vllm_parsefix"),
    ("vtsw025@final", "lmms_eval_results/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153_vllm_parsefix"),
    ("vtsanswer@final", "lmms_eval_results/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829_vllm_parsefix"),
    ("vtslmoff@ckpt1000", "lmms_eval_results/auto_research_vtslmoff_pvrdsg_checkpoint-1000_20260609_062112_vllm_parsefix"),
    ("vtslmoff@final", "lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001_vllm_parsefix"),
    ("vtslmoff_prmlp@final", "lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_prmlp_*_vllm_parsefix"),
    ("origonly24k_retention@ckpt1000", "lmms_eval_results/auto_research_origonly24k_sft_retention_checkpoint-1000_20260609_072512_vllm_parsefix"),
    ("origonly24k_retention@final", "lmms_eval_results/original_only_retention_origonly24k_sft_retention_*_vllm_parsefix"),
    ("origonly24k_lowlr_retention@ckpt1000", "lmms_eval_results/auto_research_origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858_vllm_parsefix"),
    ("origonly24k_lowlr_retention@final", "lmms_eval_results/original_only_retention_origonly24k_sft_lowlr_retention_*_vllm_parsefix"),
    ("lowlr_balanced_pvrdsg_prmlp@final", "lmms_eval_results/pvrd_sg_lowlr_scale48k_lowlr_balanced_pvrdsg_prmlp_lam003_*_vllm_parsefix"),
    ("origonly24k_freezelm_retention@final", "lmms_eval_results/original_only_retention_origonly24k_sft_freezelm_retention_*_vllm_parsefix"),
]

TASK_FIELDS = {
    "MV O": ("mathvision_testmini", "mathvision_standard_eval,none"),
    "MV V": ("mathvision_testmini_prompt_in_image", "mathvision_standard_eval,none"),
    "Vista O": ("mathvista_testmini_cot", "llm_as_judge_eval,none"),
    "Vista V": ("mathvista_testmini_prompt_in_image", "llm_as_judge_eval,none"),
}


def latest_result_json(eval_dir: Path) -> Path | None:
    if not eval_dir.exists():
        return None
    files = sorted(eval_dir.rglob("*_results.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def resolve_eval_dir(pattern: str) -> Path | None:
    if any(ch in pattern for ch in "*?[]"):
        matches = sorted(Path.cwd().glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0)
        return matches[-1] if matches else None
    return Path(pattern)


def load_metrics(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    results = data.get("results", {})
    metrics: dict[str, float] = {}
    for field, (task, key) in TASK_FIELDS.items():
        value = results.get(task, {}).get(key)
        if isinstance(value, (int, float)):
            metrics[field] = float(value)
    return metrics


def summarize(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in TASK_FIELDS:
        value = metrics.get(key)
        out[key] = value
        out[f"{key} delta"] = None if value is None else value - reference[key]
    if all(key in metrics for key in TASK_FIELDS):
        out["Avg O"] = (metrics["MV O"] + metrics["Vista O"]) / 2
        out["Avg V"] = (metrics["MV V"] + metrics["Vista V"]) / 2
        out["Avg G"] = ((metrics["MV O"] - metrics["MV V"]) + (metrics["Vista O"] - metrics["Vista V"])) / 2
        out["Avg O delta"] = out["Avg O"] - ((reference["MV O"] + reference["Vista O"]) / 2)
        out["Avg V delta"] = out["Avg V"] - ((reference["MV V"] + reference["Vista V"]) / 2)
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def main() -> None:
    print("# Auto-Research Tradeoff Analysis")
    print()
    rows: list[dict[str, Any]] = []
    for name, eval_pattern in RUNS:
        eval_dir = resolve_eval_dir(eval_pattern)
        result = latest_result_json(eval_dir) if eval_dir else None
        if result is None:
            rows.append({"Run": name, "Status": "pending", "Result": "--"})
            continue
        metrics = load_metrics(result)
        row = {"Run": name, "Status": "done", "Result": str(result)}
        row.update(summarize(metrics, BASELINE))
        row["Avg V delta vs known good"] = None
        if "Avg V" in row:
            known_good_avg_v = (KNOWN_GOOD["MV V"] + KNOWN_GOOD["Vista V"]) / 2
            row["Avg V delta vs known good"] = row["Avg V"] - known_good_avg_v
        rows.append(row)

    headers = [
        "Run",
        "Status",
        "MV O",
        "MV V",
        "Vista O",
        "Vista V",
        "Avg O",
        "Avg V",
        "Avg G",
        "Avg O delta",
        "Avg V delta",
        "Avg V delta vs known good",
        "Result",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")

    completed = [r for r in rows if r.get("Status") == "done"]
    if not completed:
        return
    best = max(completed, key=lambda r: r.get("Avg V", float("-inf")))
    print()
    print(f"Best completed Avg V so far: {best['Run']} ({fmt(best.get('Avg V'))}).")
    for row in completed:
        avg_o_delta = row.get("Avg O delta")
        avg_v_delta = row.get("Avg V delta")
        if isinstance(avg_o_delta, float) and isinstance(avg_v_delta, float):
            if avg_o_delta < -5 and avg_v_delta < -5:
                print(f"Warning: {row['Run']} is dominated by the baseline on both Avg O and Avg V.")
            elif avg_o_delta < -5:
                print(f"Warning: {row['Run']} hurts original-view retention substantially.")


if __name__ == "__main__":
    main()
