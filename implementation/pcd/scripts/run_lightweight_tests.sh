#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/mnt/weka/home/yongxin.wang/miniconda3/envs/llamafactory/bin/python}

cd "$ROOT"

"$PYTHON" - <<'PY'
import importlib.util
from pathlib import Path

root = Path.cwd()
for test_file, functions in {
    'tests/test_pvrd_answer_only_labels.py': [
        'test_mask_labels_to_answer_span_keeps_only_answer_tags_and_content',
        'test_mask_labels_to_answer_span_falls_back_when_answer_tag_missing',
    ],
    'tests/test_prepare_easyr1_model_dir.py': [
        'test_prepare_easyr1_model_dir_supplements_processor_sidecars',
    ],
    'tests/test_prompt_region_ssl_utils.py': [
        'test_mask_prompt_region_only_changes_inside_box',
        'test_build_prompt_band_mask_marks_first_prompt_row_tokens',
        'test_temporary_image_pixel_limits_override_and_restore',
        'test_maybe_build_prmlp_image_views_skips_disabled_sample_without_loading',
        'test_maybe_build_prmlp_image_views_masks_first_full_image_and_loads_crop',
    ],
    'tests/test_shallow_mm_memory_utils.py': [
        'test_resolve_layer_list_parses_unique_indices',
        'test_pool_hidden_states_mean_respects_attention_mask',
        'test_shallow_memory_adapter_gate_controls_update_strength',
    ],
}.items():
    path = root / test_file
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    for name in functions:
        getattr(mod, name)()
        print(f'PASS {test_file}::{name}')
PY
