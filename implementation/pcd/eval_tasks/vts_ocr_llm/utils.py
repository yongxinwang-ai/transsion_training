"""External lmms-eval tasks for OCR+LLM VTS diagnostics.

The VTS dataset stores a composite image where the rendered prompt panel is
placed above the original benchmark image.  This task runs OCR on that rendered
panel, then gives the OCR text plus the cropped original image to the solver.
It is a diagnostic baseline, not a deployment recipe.
"""

from __future__ import annotations

import hashlib
import csv
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from datasets import Dataset

from lmms_eval.tasks._task_utils.prompt_in_image import _build_text_panel
from lmms_eval.tasks.mathvision.utils import (
    mathvision_aggregate_results_eval,
    mathvision_process_results,
)
from lmms_eval.tasks.mathvista.mathvista_evals import MathVistaEvaluator
from lmms_eval.tasks.mathvista.utils import (
    mathvista_aggregate_results,
    mathvista_process_results,
)

MINIMAL_SOLVE_PROMPT = "Help me solve the problem"
TESSERACT_BIN = os.getenv(
    "TESSERACT_BIN",
    "/mnt/weka/home/yongxin.wang/workspace/runze/miniconda3/envs/retriever/bin/tesseract",
)
TESSDATA_PREFIX = os.getenv(
    "TESSDATA_PREFIX",
    "/mnt/weka/home/yongxin.wang/workspace/runze/miniconda3/envs/retriever/share/tessdata",
)
OCR_CACHE_DIR = Path(
    os.getenv(
        "VTS_OCR_CACHE_DIR",
        "/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd/analysis/vts_ocr_cache",
    )
)
OCR_QUALITY_CSV = Path(
    os.getenv(
        "VTS_OCR_QUALITY_CSV",
        "/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd/analysis/vts_ocr_quality/vts_ocr_quality_full.csv",
    )
)
ARROW_BY_DATASET = {
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

_mathvista_evaluator = MathVistaEvaluator()
_ocr_lookup = None
_arrow_lookup = {}


def _load_ocr_lookup():
    global _ocr_lookup
    if _ocr_lookup is not None:
        return _ocr_lookup
    lookup = {}
    if OCR_QUALITY_CSV.exists():
        with OCR_QUALITY_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                lookup[(row.get("dataset", ""), str(row.get("question_id", "")))] = row.get("ocr_text", "")
    _ocr_lookup = lookup
    return lookup


def _load_arrow_lookup(dataset_name: str):
    if dataset_name in _arrow_lookup:
        return _arrow_lookup[dataset_name]
    path = ARROW_BY_DATASET.get(dataset_name)
    if path is None or not path.exists():
        _arrow_lookup[dataset_name] = {}
        return _arrow_lookup[dataset_name]
    ds = Dataset.from_file(str(path))
    mapping = {str(row.get("question_id", "")): row for row in ds}
    _arrow_lookup[dataset_name] = mapping
    return mapping


def _get_image(doc, dataset_name: str | None = None) -> Image.Image:
    image = doc.get("decoded_image") or doc.get("image")
    if image is None and dataset_name:
        fallback = _load_arrow_lookup(dataset_name).get(str(doc.get("question_id", "")))
        if fallback is not None:
            image = fallback.get("decoded_image") or fallback.get("image")
    if image is None:
        raise KeyError(f"Expected `decoded_image` or `image` in dataset document; keys={sorted(doc.keys())}")
    return image.convert("RGB")


def _panel_height(doc, dataset_name: str | None = None) -> int:
    image = _get_image(doc, dataset_name)
    question = doc.get("question", "")
    panel = _build_text_panel(image.width, question)
    return max(0, min(panel.height, image.height))


def _crop_panel_and_original(doc, dataset_name: str | None = None) -> tuple[Image.Image, Image.Image]:
    image = _get_image(doc, dataset_name)
    h = _panel_height(doc, dataset_name)
    panel = image.crop((0, 0, image.width, h)) if h > 0 else Image.new("RGB", (image.width, 1), "white")
    original = image.crop((0, h, image.width, image.height)) if h < image.height else image
    return panel.convert("RGB"), original.convert("RGB")


def _ocr_cache_key(doc) -> str:
    qid = str(doc.get("question_id") or doc.get("pid") or "")
    question = str(doc.get("question", ""))
    payload = f"{qid}\n{question}".encode("utf-8", errors="ignore")
    return hashlib.sha1(payload).hexdigest()


def _normalize_ocr_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _run_tesseract(panel: Image.Image) -> str:
    if not Path(TESSERACT_BIN).exists() and not shutil_which(TESSERACT_BIN):
        raise FileNotFoundError(f"tesseract binary not found: {TESSERACT_BIN}")
    with TemporaryDirectory() as tmpdir:
        image_path = Path(tmpdir) / "panel.png"
        panel.save(image_path)
        cmd = [TESSERACT_BIN, str(image_path), "stdout", "--psm", os.getenv("VTS_OCR_PSM", "6"), "-l", os.getenv("VTS_OCR_LANG", "eng")]
        env = os.environ.copy()
        env["TESSDATA_PREFIX"] = TESSDATA_PREFIX
        proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"tesseract failed with code {proc.returncode}: {proc.stderr[:500]}")
        return _normalize_ocr_text(proc.stdout)


def shutil_which(binary: str) -> str | None:
    paths = os.getenv("PATH", "").split(os.pathsep)
    for path in paths:
        candidate = Path(path) / binary
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _ocr_prompt_text(doc, dataset_name: str | None = None) -> str:
    if dataset_name:
        cached = _load_ocr_lookup().get((dataset_name, str(doc.get("question_id", ""))))
        if cached is not None:
            return cached.strip()
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OCR_CACHE_DIR / f"{_ocr_cache_key(doc)}.txt"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8").strip()
    panel, _ = _crop_panel_and_original(doc, dataset_name)
    text = _run_tesseract(panel)
    cache_path.write_text(text + "\n", encoding="utf-8")
    return text


def mathvision_doc_to_visual_ocr_llm(doc):
    _, original = _crop_panel_and_original(doc, "mathvision")
    return [original]


def mathvision_doc_to_text_ocr_llm(doc, lmms_eval_specific_kwargs=None):
    ocr_question = _ocr_prompt_text(doc, "mathvision")
    if not ocr_question:
        ocr_question = MINIMAL_SOLVE_PROMPT

    choices = doc.get("options") or []
    choices_str = ""
    if choices:
        labels = [chr(ord("A") + i) for i in range(len(choices))]
        choices_str = "\nChoices: " + "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))

    mc_prompt = ""
    if lmms_eval_specific_kwargs is not None and choices:
        mc_prompt = "\n" + lmms_eval_specific_kwargs.get("mc_prompt", "")
    short_prompt = ""
    if lmms_eval_specific_kwargs is not None and not choices:
        short_prompt = "\n" + lmms_eval_specific_kwargs.get("short_answer_prompt", "")

    return f'Please solve the problem step by step and put your answer in one "\\boxed{{}}".\n{ocr_question}{choices_str}{mc_prompt}{short_prompt}'


def mathvista_doc_to_visual_ocr_llm(doc):
    _, original = _crop_panel_and_original(doc, "mathvista")
    return [original]


def mathvista_doc_to_text_ocr_llm(doc, lmms_eval_specific_kwargs=None):
    ocr_question = _ocr_prompt_text(doc, "mathvista")
    if not ocr_question:
        ocr_question = doc.get("question", "") or MINIMAL_SOLVE_PROMPT

    kwargs = lmms_eval_specific_kwargs or {}
    problem = {
        "question_type": doc["question_type"],
        "answer_type": doc["answer_type"],
        "question": ocr_question,
        "unit": doc["unit"] if "unit" in doc else "",
        "caption": doc["caption"] if "caption" in doc else "",
        "ocr": doc["ocr"] if "ocr" in doc else "",
        "choices": doc["choices"],
        "answer": doc["answer"] if "answer" in doc else None,
        "precision": doc["precision"] if "precision" in doc else 0,
    }
    return _mathvista_evaluator.create_one_query(
        problem,
        shot_num=kwargs.get("shot", 0),
        shot_type=kwargs.get("shot_type", "step-by-step"),
        use_caption=kwargs.get("use_caption", False),
        use_ocr=kwargs.get("use_ocr", False),
    )
