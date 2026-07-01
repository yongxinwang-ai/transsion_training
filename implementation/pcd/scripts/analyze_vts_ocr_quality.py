#!/usr/bin/env python3
"""Measure how much prompt text survives OCR in VTS benchmark images.

This is a CPU-side diagnostic.  It crops the rendered task-semantics panel from
each composite VTS image, runs Tesseract on the panel, and compares the OCR
output to the original textual question.  The summary is useful for separating
"the model cannot use visualized task semantics" from "OCR cannot recover the
task semantics".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset


DEFAULT_PCD_ROOT = Path(
    "/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd"
)
DEFAULT_LMMS_ROOT = Path("/mnt/weka/home/yongxin.wang/workspace/lmms-eval")
DEFAULT_TESSERACT = Path("/mnt/weka/home/yongxin.wang/workspace/runze/miniconda3/envs/retriever/bin/tesseract")
DEFAULT_TESSDATA_PREFIX = Path(
    "/mnt/weka/home/yongxin.wang/workspace/runze/miniconda3/envs/retriever/share/tessdata"
)
DEFAULT_ARROW = {
    "mathvision": Path(
        "/mnt/weka/home/yongxin.wang/.cache/huggingface/datasets/"
        "YongxinWang___math-prompt-in-image/mathvision_testmini_prompt_in_image/"
        "0.0.0/6cf2d280a04d7db7505904c7c6146628597b27a1/"
        "math-prompt-in-image-testmini.arrow"
    ),
    "mathvista": Path(
        "/mnt/weka/home/yongxin.wang/.cache/huggingface/datasets/"
        "YongxinWang___math-prompt-in-image/mathvista_testmini_prompt_in_image/"
        "0.0.0/6cf2d280a04d7db7505904c7c6146628597b27a1/"
        "math-prompt-in-image-testmini.arrow"
    ),
}


def _add_lmms_to_path(lmms_root: Path) -> None:
    sys.path.insert(0, str(lmms_root))


def normalize_for_edit_distance(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_words(text: str) -> list[str]:
    text = normalize_for_edit_distance(text)
    return re.findall(r"[a-z0-9]+|[^\w\s]", text)


def levenshtein(a: str | list[str], b: str | list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]


def rate(distance: int, denom: int) -> float:
    if denom <= 0:
        return 0.0 if distance == 0 else 1.0
    return distance / denom


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
    return values[idx]


def run_tesseract(panel, tesseract_bin: Path, tessdata_prefix: Path, psm: str, lang: str) -> str:
    with TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "panel.png"
        panel.save(image_path)
        cmd = [str(tesseract_bin), str(image_path), "stdout", "--psm", psm, "-l", lang]
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = str(tessdata_prefix)
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract failed with code {proc.returncode}: {proc.stderr[:500]}")
        return proc.stdout.strip()


@dataclass
class Row:
    dataset: str
    index: int
    question_id: str
    image_width: int
    image_height: int
    panel_height: int
    ref_chars: int
    ocr_chars: int
    cer: float
    wer: float
    empty_ocr: bool
    reference: str
    ocr_text: str


def analyze_dataset(
    dataset_name: str,
    arrow_path: Path,
    *,
    limit: int | None,
    tesseract_bin: Path,
    tessdata_prefix: Path,
    psm: str,
    lang: str,
):
    from lmms_eval.tasks._task_utils.prompt_in_image import _build_text_panel

    ds = Dataset.from_file(str(arrow_path))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    rows: list[Row] = []
    for idx, doc in enumerate(ds):
        image = (doc.get("decoded_image") or doc.get("image")).convert("RGB")
        question = doc.get("question", "")
        panel = _build_text_panel(image.width, question)
        panel_height = max(0, min(panel.height, image.height))
        panel_crop = image.crop((0, 0, image.width, panel_height)) if panel_height else panel

        ocr_text = run_tesseract(panel_crop, tesseract_bin, tessdata_prefix, psm=psm, lang=lang)
        ref_norm = normalize_for_edit_distance(question)
        ocr_norm = normalize_for_edit_distance(ocr_text)
        ref_words = normalize_words(question)
        ocr_words = normalize_words(ocr_text)
        cer = rate(levenshtein(ref_norm, ocr_norm), len(ref_norm))
        wer = rate(levenshtein(ref_words, ocr_words), len(ref_words))
        rows.append(
            Row(
                dataset=dataset_name,
                index=idx,
                question_id=str(doc.get("question_id", "")),
                image_width=image.width,
                image_height=image.height,
                panel_height=panel_height,
                ref_chars=len(ref_norm),
                ocr_chars=len(ocr_norm),
                cer=cer,
                wer=wer,
                empty_ocr=not bool(ocr_norm),
                reference=question,
                ocr_text=ocr_text,
            )
        )
        if (idx + 1) % 100 == 0:
            print(f"[{dataset_name}] processed {idx + 1}/{len(ds)}", flush=True)
    return rows


def summarize(rows: list[Row]) -> dict:
    by_dataset = {}
    for dataset in sorted({row.dataset for row in rows}):
        subset = [row for row in rows if row.dataset == dataset]
        cers = [row.cer for row in subset]
        wers = [row.wer for row in subset]
        by_dataset[dataset] = {
            "n": len(subset),
            "cer_mean": sum(cers) / max(1, len(cers)),
            "cer_median": percentile(cers, 0.5),
            "cer_p90": percentile(cers, 0.9),
            "wer_mean": sum(wers) / max(1, len(wers)),
            "wer_median": percentile(wers, 0.5),
            "wer_p90": percentile(wers, 0.9),
            "empty_ocr_rate": sum(row.empty_ocr for row in subset) / max(1, len(subset)),
            "mean_panel_fraction": sum(row.panel_height / max(1, row.image_height) for row in subset) / max(1, len(subset)),
        }
    return {
        "overall": {
            "n": len(rows),
            "cer_mean": sum(row.cer for row in rows) / max(1, len(rows)),
            "wer_mean": sum(row.wer for row in rows) / max(1, len(rows)),
            "empty_ocr_rate": sum(row.empty_ocr for row in rows) / max(1, len(rows)),
        },
        "by_dataset": by_dataset,
        "worst_by_cer": [
            {
                "dataset": row.dataset,
                "index": row.index,
                "question_id": row.question_id,
                "cer": row.cer,
                "wer": row.wer,
                "reference": row.reference[:500],
                "ocr_text": row.ocr_text[:500],
            }
            for row in sorted(rows, key=lambda r: r.cer, reverse=True)[:20]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmms-root", type=Path, default=DEFAULT_LMMS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PCD_ROOT / "analysis" / "vts_ocr_quality")
    parser.add_argument("--tesseract-bin", type=Path, default=DEFAULT_TESSERACT)
    parser.add_argument("--tessdata-prefix", type=Path, default=DEFAULT_TESSDATA_PREFIX)
    parser.add_argument("--psm", default=os.getenv("VTS_OCR_PSM", "6"))
    parser.add_argument("--lang", default=os.getenv("VTS_OCR_LANG", "eng"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mathvision-arrow", type=Path, default=DEFAULT_ARROW["mathvision"])
    parser.add_argument("--mathvista-arrow", type=Path, default=DEFAULT_ARROW["mathvista"])
    args = parser.parse_args()

    _add_lmms_to_path(args.lmms_root)
    if not args.tesseract_bin.exists():
        raise FileNotFoundError(f"missing tesseract binary: {args.tesseract_bin}")
    if not (args.tessdata_prefix / "eng.traineddata").exists():
        raise FileNotFoundError(f"missing eng.traineddata under: {args.tessdata_prefix}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[Row] = []
    for dataset_name, arrow_path in {
        "mathvision": args.mathvision_arrow,
        "mathvista": args.mathvista_arrow,
    }.items():
        if not arrow_path.exists():
            raise FileNotFoundError(f"missing arrow file for {dataset_name}: {arrow_path}")
        all_rows.extend(
            analyze_dataset(
                dataset_name,
                arrow_path,
                limit=args.limit,
                tesseract_bin=args.tesseract_bin,
                tessdata_prefix=args.tessdata_prefix,
                psm=args.psm,
                lang=args.lang,
            )
        )

    suffix = f"limit{args.limit}" if args.limit else "full"
    csv_path = args.output_dir / f"vts_ocr_quality_{suffix}.csv"
    json_path = args.output_dir / f"vts_ocr_quality_{suffix}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(Row.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row.__dict__)
    summary = summarize(all_rows)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
