"""VTS text-vs-image conflict-control tasks.

The image contains the rendered original benchmark question. The text channel
contains a conflicting instruction that asks for a fixed sentinel answer. The
metrics estimate whether the model follows the image prompt semantics, the text
channel semantics, or neither.
"""

from __future__ import annotations

import re

from lmms_eval.tasks.mathvision.utils import mathvision_process_results

TEXT_SENTINEL = "999999"
CONFLICT_PROMPT = (
    "This is a channel-conflict diagnostic. The image may contain a different "
    "problem statement, but for this test you must follow the text instruction. "
    f"Do not solve the problem shown in the image. Return exactly \\boxed{{{TEXT_SENTINEL}}}."
)


def _get_doc_image(doc):
    image = doc.get("decoded_image") or doc.get("image")
    if image is None:
        raise KeyError(f"Expected `decoded_image` or `image`; keys={sorted(doc.keys())}")
    return image.convert("RGB")


def doc_to_visual(doc):
    # The prompt-in-image HF split already stores the composite image.
    return [_get_doc_image(doc)]


def doc_to_text_conflict(doc, lmms_eval_specific_kwargs=None):
    return CONFLICT_PROMPT


def doc_to_target_image_answer(doc):
    return str(doc.get("answer", ""))


def _extract_final_answer(text: str) -> str:
    text = "" if text is None else str(text).strip()
    if not text:
        return ""

    tag_matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if tag_matches:
        return tag_matches[-1].strip()

    boxed_matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    labeled_matches = re.findall(r"(?:final answer|answer)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if labeled_matches:
        return labeled_matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    return text


def _normalize_answer(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip(" .,:;!?$\\")


def _strict_matches_target(response: str, target: str) -> bool:
    pred = _normalize_answer(_extract_final_answer(response)).lower()
    gold = _normalize_answer(target).lower()
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pred_compact = re.sub(r"\s+", "", pred)
    gold_compact = re.sub(r"\s+", "", gold)
    if pred_compact == gold_compact:
        return True
    try:
        return abs(float(pred_compact) - float(gold_compact)) < 1e-6
    except ValueError:
        return False


def _follows_text_sentinel(response: str) -> bool:
    extracted = _normalize_answer(_extract_final_answer(response))
    if extracted == TEXT_SENTINEL:
        return True
    # Fallback for models that obey the diagnostic but omit a clean final-answer marker.
    tail = _normalize_answer(response[-200:])
    return bool(re.search(rf"(?:boxed\s*)?{re.escape(TEXT_SENTINEL)}\b", tail))


def _format_result(image_follow: bool, text_follow: bool, response: str):
    return {
        "image_follow_rate": bool(image_follow and not text_follow),
        "text_follow_rate": bool(text_follow and not image_follow),
        "both_rate": bool(image_follow and text_follow),
        "neither_rate": bool((not image_follow) and (not text_follow)),
        "conflict_eval": {
            "image_follow": bool(image_follow),
            "text_follow": bool(text_follow),
            "both": bool(image_follow and text_follow),
            "neither": bool((not image_follow) and (not text_follow)),
            "sentinel": TEXT_SENTINEL,
            "extracted_final": _extract_final_answer(response),
            "response": response,
        },
    }


def mathvision_process_results_conflict(doc, results):
    response = results[0].strip() if results else ""
    image_follow = bool(mathvision_process_results(doc, [response])["mathvision_standard_eval"]["scores"][0])
    text_follow = _follows_text_sentinel(response)
    return _format_result(image_follow, text_follow, response)


def mathvista_process_results_conflict(doc, results):
    response = results[0].strip() if results else ""
    # Do not reuse the normal MathVista judge here. Under a sentinel conflict,
    # the local normalizer can map an invalid numeric answer to a permissive
    # true/false or multiple-choice value. For channel following, we only need
    # a strict check against the original image-side answer.
    image_follow = _strict_matches_target(response, str(doc.get("answer", "")))
    text_follow = _follows_text_sentinel(response)
    return _format_result(image_follow, text_follow, response)


def aggregate_percent(results):
    if not results:
        return 0.0
    return round(100.0 * sum(bool(x) for x in results) / len(results), 2)
