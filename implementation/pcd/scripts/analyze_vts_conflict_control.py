#!/usr/bin/env python3
"""Strict post-processing for VTS text-vs-image conflict-control samples."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PCD = Path('/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd')
OUT = PCD / 'analysis' / 'vts_conflict_control_20260503'
OUT.mkdir(parents=True, exist_ok=True)
TEXT_SENTINEL = '999999'
RUNS = {
    'Base': PCD / 'lmms_eval_results/base_qwen3vl4b_conflict_control_vllm_20260503',
    'PVRD-SG': PCD / 'lmms_eval_results/pvrd_best_conflict_control_vllm_20260503',
    'RLVR': PCD / 'lmms_eval_results/easyr1_rlvr_step750_conflict_control_vllm_20260503',
}
TASKS = {
    'VTS-MATH-Vision': '*samples_mathvision_testmini_vts_text_image_conflict.jsonl',
    'VTS-MathVista': '*samples_mathvista_testmini_vts_text_image_conflict.jsonl',
}


def extract_final(text: str) -> str:
    text = '' if text is None else str(text).strip()
    tag = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, flags=re.I | re.S)
    if tag:
        return tag[-1].strip()
    boxed = re.findall(r'\\boxed\{([^{}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    labeled = re.findall(r'(?:final answer|answer)\s*[:：]\s*(.+)', text, flags=re.I)
    if labeled:
        return labeled[-1].strip()
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return lines[-1] if lines else text


def norm(text: str) -> str:
    text = '' if text is None else str(text)
    text = text.replace(',', '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip().strip(' .,:;!?$\\').lower()


def strict_match(pred_text: str, target: str) -> bool:
    pred = norm(extract_final(pred_text))
    gold = norm(target)
    if not pred or not gold:
        return False
    if pred == gold:
        return True
    pred_c = re.sub(r'\s+', '', pred)
    gold_c = re.sub(r'\s+', '', gold)
    if pred_c == gold_c:
        return True
    try:
        return abs(float(pred_c) - float(gold_c)) < 1e-6
    except ValueError:
        return False


def text_follow(resp: str) -> bool:
    final = norm(extract_final(resp))
    if final == TEXT_SENTINEL:
        return True
    tail = norm(resp[-200:])
    return bool(re.search(rf'(?:boxed\s*)?{re.escape(TEXT_SENTINEL)}\b', tail))


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(f'**/{pattern}'))
    return matches[-1] if matches else None


def main() -> None:
    rows = []
    summary = []
    for model, root in RUNS.items():
        for task, pattern in TASKS.items():
            path = find_one(root, pattern)
            if path is None:
                summary.append({'model': model, 'task': task, 'status': 'missing'})
                continue
            counts = {'n': 0, 'image_follow': 0, 'text_follow': 0, 'both': 0, 'neither': 0}
            with path.open() as f:
                for line in f:
                    obj = json.loads(line)
                    resp = (obj.get('filtered_resps') or [''])[0]
                    target = str(obj.get('target', ''))
                    img = strict_match(resp, target)
                    txt = text_follow(resp)
                    counts['n'] += 1
                    counts['image_follow'] += bool(img and not txt)
                    counts['text_follow'] += bool(txt and not img)
                    counts['both'] += bool(img and txt)
                    counts['neither'] += bool((not img) and (not txt))
                    rows.append({
                        'model': model,
                        'task': task,
                        'doc_id': obj.get('doc_id'),
                        'target': target,
                        'final': extract_final(resp),
                        'image_follow': img,
                        'text_follow': txt,
                        'both': img and txt,
                        'neither': (not img) and (not txt),
                        'response': resp[:1000],
                    })
            item = {'model': model, 'task': task, 'status': 'ok', **counts}
            for key in ['image_follow', 'text_follow', 'both', 'neither']:
                item[f'{key}_rate'] = round(100.0 * counts[key] / counts['n'], 2) if counts['n'] else 0.0
            summary.append(item)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    with (OUT / 'rows.csv').open('w', newline='') as f:
        fields = ['model','task','doc_id','target','final','image_follow','text_follow','both','neither','response']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
