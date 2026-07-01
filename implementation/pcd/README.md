# Prompt-Region Grounding Implementation

This directory is the sanitized implementation delta for the paper method. It is designed to sit beside a LLaMA-Factory installation rather than replace the full framework.

## What Is Implemented

1. **VTS paired export**
   - Converts source multimodal samples into balanced original/VTS LLaMA-Factory rows.
   - Preserves the same assistant target on both views.
   - Stores prompt-panel metadata needed for region-local objectives.

2. **PVRD-SG**
   - Pools final decoder states corresponding to rendered prompt-panel visual tokens.
   - Aligns them to a stop-gradient text embedding of the source question.
   - Does not pool answer or reasoning target tokens.

3. **PRMLP**
   - Uses a stop-gradient clean prompt-crop representation as an image-side target.
   - Avoids OCR-character or token reconstruction supervision.
   - Can be enabled as a lightweight auxiliary regularizer.

4. **Diagnostics**
   - OCR-only and OCR+MLLM channel surgery.
   - Text-versus-visual prompt conflict controls.
   - Prompt occlusion and prompt-band-only controls.

## Typical Run Order

```bash
# 1. Export paired LLaMA-Factory data.
python scripts/export_llamafactory_pvrd_sg_dataset.py --help

# 2. Launch the main supervised chain.
bash scripts/submit_pvrd_sg_prmlp_scale48k_chain.sh

# 3. Snapshot a checkpoint and run vLLM/lmms-eval diagnostics.
bash scripts/snapshot_and_submit_auto_research_eval.sh <run_key>

# 4. Summarize method tradeoffs.
python scripts/summarize_auto_research_runs.py --root .
python scripts/analyze_auto_research_tradeoff.py
```

## Important Configs

- `configs/qwen3vl_pvrd_sg_prmlp_llamafactory_nodz_instruct_scale48k_stable.yaml`: stable full-SFT configuration.
- `configs/qwen3vl_pvrd_sg_prmlp_llamafactory_nodz_instruct_pilot50.yaml`: small smoke/pilot configuration.
- `configs/qwen3vl_pvrd_sg_prmlp_debug_1gpu_2step.yaml`: 1-GPU debug run.
- `configs/qwen3vl_pvrd_sg_prmlp_debug_8gpu_2step.yaml`: 8-GPU debug run.

## Tests

```bash
bash scripts/run_lightweight_tests.sh
```

The lightweight tests cover prompt-region utility behavior, answer-only label masking, EasyR1-compatible model directory preparation, and shallow memory utilities.
