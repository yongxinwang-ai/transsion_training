"""VTS prompt-region occlusion tasks.

The HF prompt-in-image split stores a composite image whose top panel contains
rendered task semantics. This task keeps the same composite canvas but blanks
out the rendered prompt panel, while keeping the minimal text prompt. It is a
causal input ablation for whether the prompt panel pixels matter.
"""

from __future__ import annotations

from PIL import Image

from lmms_eval.tasks._task_utils.prompt_in_image import MINIMAL_SOLVE_PROMPT, _build_text_panel
from lmms_eval.tasks.mathvision.utils import mathvision_aggregate_results_eval, mathvision_process_results
from lmms_eval.tasks.mathvista.utils import mathvista_aggregate_results, mathvista_process_results


def _get_doc_image(doc) -> Image.Image:
    image = doc.get("decoded_image") or doc.get("image")
    if image is None:
        raise KeyError(f"Expected `decoded_image` or `image`; keys={sorted(doc.keys())}")
    return image.convert("RGB")


def _panel_height(doc, image: Image.Image) -> int:
    question = doc.get("question", "")
    panel = _build_text_panel(image.width, question)
    return max(0, min(panel.height, image.height))


def doc_to_visual_prompt_occluded(doc, lmms_eval_specific_kwargs=None):
    image = _get_doc_image(doc)
    h = _panel_height(doc, image)
    if h <= 0:
        return [image]
    occluded = image.copy()
    blank = Image.new("RGB", (image.width, h), "white")
    occluded.paste(blank, (0, 0))
    return [occluded]


def doc_to_visual_prompt_band_only(doc, lmms_eval_specific_kwargs=None):
    """Keep only the prompt panel; blank out the original visual evidence."""
    image = _get_doc_image(doc)
    h = _panel_height(doc, image)
    if h <= 0 or h >= image.height:
        return [image]
    masked = image.copy()
    blank = Image.new("RGB", (image.width, image.height - h), "white")
    masked.paste(blank, (0, h))
    return [masked]


def doc_to_text_minimal(doc, lmms_eval_specific_kwargs=None):
    return MINIMAL_SOLVE_PROMPT
