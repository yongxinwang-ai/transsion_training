"""OCR-only VTS diagnostics.

This task transcribes the rendered prompt panel and gives only the recovered
text to the solver.  Unlike OCR+MLLM channel surgery, it does not pass the
cropped original image.  The baseline separates OCR transcription from visual
reasoning over the diagram.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_BASE_UTILS = Path(__file__).resolve().parents[1] / "vts_ocr_llm" / "utils.py"
_SPEC = importlib.util.spec_from_file_location("_vts_ocr_llm_utils", _BASE_UTILS)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load OCR+MLLM utilities from {_BASE_UTILS}")
_ocr_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ocr_utils)

# Reuse the same answer processing and aggregation as the source benchmarks.
mathvision_process_results = _ocr_utils.mathvision_process_results
mathvision_aggregate_results_eval = _ocr_utils.mathvision_aggregate_results_eval
mathvista_process_results = _ocr_utils.mathvista_process_results
mathvista_aggregate_results = _ocr_utils.mathvista_aggregate_results


def mathvision_doc_to_visual_ocr_only(doc):
    return []


def mathvision_doc_to_text_ocr_only(doc, lmms_eval_specific_kwargs=None):
    return _ocr_utils.mathvision_doc_to_text_ocr_llm(doc, lmms_eval_specific_kwargs)


def mathvista_doc_to_visual_ocr_only(doc):
    return []


def mathvista_doc_to_text_ocr_only(doc, lmms_eval_specific_kwargs=None):
    return _ocr_utils.mathvista_doc_to_text_ocr_llm(doc, lmms_eval_specific_kwargs)
