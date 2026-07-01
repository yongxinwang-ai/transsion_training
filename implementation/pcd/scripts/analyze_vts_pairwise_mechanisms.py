#!/usr/bin/env python3
"""Pairwise diagnostics for Visualized Task Semantics evaluation outputs.

The OCR condition is OCR+VLM, not text-only OCR+LLM: the rendered prompt panel
is transcribed into text tokens, the original image is cropped back out, and
the same VLM solves from recovered text plus image.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path('/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd')
BASE = ROOT / 'lmms_eval_results/base_qwen3vl4b_vllm_vts4_20260430_retry1/snapshots__ebb281ec70b05090aa6165b016eac8ec08e71b17'
OCR_VLM = ROOT / 'lmms_eval_results/base_qwen3vl4b_ocr_llm_vllm_20260430_retry2/snapshots__ebb281ec70b05090aa6165b016eac8ec08e71b17'
OCR_QUALITY = ROOT / 'analysis/vts_ocr_quality/vts_ocr_quality_full.csv'
OUT_DIR = ROOT / 'analysis/vts_pairwise_mechanisms'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'mathvision': {
        'original': BASE / '20260430_214023_samples_mathvision_testmini.jsonl',
        'vts': BASE / '20260430_214023_samples_mathvision_testmini_prompt_in_image.jsonl',
        'ocr_vlm': OCR_VLM / '20260430_214758_samples_mathvision_testmini_vts_ocr_llm.jsonl',
    },
    'mathvista': {
        'original': BASE / '20260430_214023_samples_mathvista_testmini_cot.jsonl',
        'vts': BASE / '20260430_214023_samples_mathvista_testmini_prompt_in_image.jsonl',
        'ocr_vlm': OCR_VLM / '20260430_214758_samples_mathvista_testmini_vts_ocr_llm.jsonl',
    },
}


def is_correct(dataset: str, obj: dict) -> bool:
    if dataset == 'mathvision':
        scores = obj.get('mathvision_standard_eval', {}).get('scores', [])
        return bool(scores and scores[0])
    if dataset == 'mathvista':
        ev = obj.get('llm_as_judge_eval') or obj.get('submission') or {}
        return bool(ev.get('true_false'))
    raise KeyError(dataset)


def pred(dataset: str, obj: dict) -> str:
    if dataset == 'mathvision':
        ev = obj.get('mathvision_standard_eval', {})
        response = ev.get('response') or obj.get('filtered_resps') or ['']
        return str(response[0] if isinstance(response, list) else response)
    ev = obj.get('llm_as_judge_eval') or obj.get('submission') or {}
    return str(ev.get('prediction') or obj.get('filtered_resps', [''])[0])


def load_samples(dataset: str, path: Path) -> dict[int, dict]:
    rows = {}
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            doc_id = int(obj['doc_id'])
            rows[doc_id] = {
                'doc_id': doc_id,
                'target': obj.get('target'),
                'correct': is_correct(dataset, obj),
                'prediction': pred(dataset, obj),
                'response_chars': len((obj.get('filtered_resps') or [''])[0]),
                'input_chars': len(obj.get('input') or ''),
            }
    return rows


def load_ocr_quality() -> dict[tuple[str, int], dict]:
    out = {}
    with OCR_QUALITY.open() as f:
        for row in csv.DictReader(f):
            dataset = row['dataset']
            idx = int(row['index'])
            out[(dataset, idx)] = {
                'cer': float(row['cer']),
                'wer': float(row['wer']),
                'empty_ocr': row['empty_ocr'].lower() == 'true',
                'ref_chars': int(row['ref_chars']),
                'ocr_chars': int(row['ocr_chars']),
            }
    return out


def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def summarize_dataset(dataset: str, q: dict) -> tuple[dict, list[dict]]:
    samples = {name: load_samples(dataset, path) for name, path in FILES[dataset].items()}
    ids = sorted(set.intersection(*(set(x) for x in samples.values())))
    rows = []
    for idx in ids:
        row = {'dataset': dataset, 'doc_id': idx}
        for name in ['original', 'vts', 'ocr_vlm']:
            row[f'{name}_correct'] = samples[name][idx]['correct']
            row[f'{name}_prediction'] = samples[name][idx]['prediction'][:500]
            row[f'{name}_response_chars'] = samples[name][idx]['response_chars']
        row.update(q.get((dataset, idx), {}))
        rows.append(row)

    n = len(rows)
    c = Counter()
    for r in rows:
        o, v, h = r['original_correct'], r['vts_correct'], r['ocr_vlm_correct']
        c['original_correct'] += o
        c['vts_correct'] += v
        c['ocr_vlm_correct'] += h
        c['orig_ok_vts_bad'] += o and not v
        c['orig_bad_vts_ok'] += (not o) and v
        c['vts_bad_ocr_vlm_ok'] += (not v) and h
        c['vts_ok_ocr_vlm_bad'] += v and not h
        c['orig_ok_vts_bad_ocr_vlm_ok'] += o and (not v) and h
        c['orig_ok_vts_bad_ocr_vlm_bad'] += o and (not v) and (not h)
        c['all_correct'] += o and v and h
        c['all_wrong'] += (not o) and (not v) and (not h)

    bins = {
        'cer_le_0p05': lambda r: r.get('cer', 1) <= 0.05,
        'cer_0p05_0p15': lambda r: 0.05 < r.get('cer', 1) <= 0.15,
        'cer_gt_0p15': lambda r: r.get('cer', 1) > 0.15,
    }
    bin_summary = {}
    for name, fn in bins.items():
        subset = [r for r in rows if 'cer' in r and fn(r)]
        bin_summary[name] = {
            'n': len(subset),
            'vts_acc': pct(sum(r['vts_correct'] for r in subset), len(subset)),
            'original_acc': pct(sum(r['original_correct'] for r in subset), len(subset)),
            'ocr_vlm_acc': pct(sum(r['ocr_vlm_correct'] for r in subset), len(subset)),
            'orig_ok_vts_bad_rate': pct(sum(r['original_correct'] and not r['vts_correct'] for r in subset), len(subset)),
            'vts_bad_ocr_vlm_ok_rate': pct(sum((not r['vts_correct']) and r['ocr_vlm_correct'] for r in subset), len(subset)),
        }

    summary = {
        'dataset': dataset,
        'n': n,
        'accuracy': {
            'original': pct(c['original_correct'], n),
            'vts': pct(c['vts_correct'], n),
            'ocr_vlm': pct(c['ocr_vlm_correct'], n),
        },
        'transitions': {k: {'count': int(v), 'rate': pct(int(v), n)} for k, v in c.items()},
        'conditional': {
            'among_original_correct_vts_fails': {
                'n': int(c['orig_ok_vts_bad']),
                'ocr_vlm_recovers_rate': pct(c['orig_ok_vts_bad_ocr_vlm_ok'], c['orig_ok_vts_bad']),
                'ocr_vlm_still_fails_rate': pct(c['orig_ok_vts_bad_ocr_vlm_bad'], c['orig_ok_vts_bad']),
            },
            'among_vts_failures': {
                'n': int(sum(not r['vts_correct'] for r in rows)),
                'ocr_vlm_recovers_rate': pct(c['vts_bad_ocr_vlm_ok'], sum(not r['vts_correct'] for r in rows)),
            },
        },
        'ocr_quality_bins': bin_summary,
        'ocr_quality': {
            'cer_mean': round(mean(r['cer'] for r in rows if 'cer' in r), 4),
            'cer_median': round(median(r['cer'] for r in rows if 'cer' in r), 4),
            'wer_mean': round(mean(r['wer'] for r in rows if 'wer' in r), 4),
        },
        'response_chars': {
            name: round(mean(samples[name][idx]['response_chars'] for idx in ids), 1)
            for name in ['original', 'vts', 'ocr_vlm']
        },
    }
    return summary, rows


def main() -> None:
    q = load_ocr_quality()
    summaries = []
    all_rows = []
    for dataset in ['mathvision', 'mathvista']:
        summary, rows = summarize_dataset(dataset, q)
        summaries.append(summary)
        all_rows.extend(rows)

    total_n = len(all_rows)
    overall = {'dataset': 'overall', 'n': total_n}
    for key in ['original_correct', 'vts_correct', 'ocr_vlm_correct']:
        overall[key.replace('_correct', '_accuracy')] = pct(sum(r[key] for r in all_rows), total_n)
    orig_ok_vts_bad = sum(r['original_correct'] and not r['vts_correct'] for r in all_rows)
    orig_ok_vts_bad_ocr_vlm_ok = sum(
        r['original_correct'] and (not r['vts_correct']) and r['ocr_vlm_correct'] for r in all_rows
    )
    vts_bad = sum(not r['vts_correct'] for r in all_rows)
    vts_bad_ocr_vlm_ok = sum((not r['vts_correct']) and r['ocr_vlm_correct'] for r in all_rows)
    overall['orig_ok_vts_bad_count'] = int(orig_ok_vts_bad)
    overall['orig_ok_vts_bad_rate'] = pct(orig_ok_vts_bad, total_n)
    overall['orig_ok_vts_bad_ocr_vlm_ok_count'] = int(orig_ok_vts_bad_ocr_vlm_ok)
    overall['orig_ok_vts_bad_ocr_vlm_ok_rate'] = pct(orig_ok_vts_bad_ocr_vlm_ok, orig_ok_vts_bad)
    overall['vts_bad_count'] = int(vts_bad)
    overall['vts_bad_ocr_vlm_ok_count'] = int(vts_bad_ocr_vlm_ok)
    overall['vts_bad_ocr_vlm_ok_rate_total'] = pct(vts_bad_ocr_vlm_ok, total_n)
    overall['vts_bad_ocr_vlm_ok_rate_conditional'] = pct(vts_bad_ocr_vlm_ok, vts_bad)
    summaries.append(overall)

    (OUT_DIR / 'pairwise_summary.json').write_text(json.dumps(summaries, indent=2, ensure_ascii=False))
    with (OUT_DIR / 'pairwise_rows.csv').open('w', newline='') as f:
        fields = sorted(set().union(*(r.keys() for r in all_rows)))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))
    print(f'wrote {OUT_DIR}')

if __name__ == '__main__':
    main()
