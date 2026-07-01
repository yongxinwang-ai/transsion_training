# Auto-Research Method Optimization Log

Updated: 2026-05-29 UTC

## Current bottleneck

The strongest open issue is not whether the model can learn the VTS format at all.  Existing runs show that prompt-region objectives and replay can improve VTS on MATH-Vision, but full 48K supervised runs can damage MathVista VTS relative to the base model.  This suggests that VTS rows are doing two things at once:

1. providing useful prompt-region grounding signal, and
2. overwriting the base model with noisy or distribution-shifted teacher reasoning targets.

The second effect is especially risky on benchmarks where the base model already has good original/VTS behavior.

## Selected next method: VTS SFT loss rebalancing

The next experiment keeps the 1:1 source mixture and PVRD-SG grounding, but downweights the autoregressive SFT loss on VTS rows:

\[
\mathcal{L} = w_{\mathrm{vts}}\mathcal{L}_{\mathrm{sft}} + \lambda_{\mathrm{sg}}\mathcal{L}_{\mathrm{sg}},
\]

where the weighting is applied only when `sg_enabled` marks a VTS row.  Original replay rows keep full SFT weight.  The current pilot uses:

- `PVRD_VTS_SFT_WEIGHT=0.25`
- `PVRD_SG_LAMBDA=0.15`
- `PRMLP_ENABLED=0`
- 48K balanced SymThink dataset
- 1 epoch, 8 GPUs, `save_total_limit=1`

The motivation is to use VTS rows primarily as prompt-region alignment data while reducing damage from their generated reasoning traces.

## Implementation

Files:

- `scripts/run_pvrd_sg_llamafactory.py`
- `scripts/llamafactory_qwen3vl_pvrd_sg_sft.slurm.sh`
- `scripts/submit_pvrd_sg_vtsloss_scale48k_chain.sh`

The trainer patch scales `base_loss` by `PVRD_VTS_SFT_WEIGHT` only on VTS rows before adding PVRD-SG.  This is valid for the current per-device batch size of 1.  If future runs increase per-device batch size, this should become example-wise loss weighting.

## Submitted run

- Train job: `1692899`
- Eval job: `1692900`
- Train dir: `runs/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153`
- Eval dir: `lmms_eval_results/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153_vllm_parsefix`
- Status update: job `1692899` started on `fs-mbz-gpu-814` and entered the training loop; early loss/grad norm logs are normal.  Eval job `1692900` remains dependency-pending.

## Expected evidence of success

The run is useful only if it improves the tradeoff, not just one metric.  Desired outcome:

- MATH-Vision VTS stays above the base model and near prior PVRD-SG runs.
- MathVista VTS recovers relative to previous 48K SymThink/PRMLP runs.
- Original MATH-Vision and MathVista remain close to base or balanced replay.
- The average original/VTS gap shrinks without external OCR.

If this succeeds, the next ablation is `w_vts=0.5` to test whether some VTS target supervision is still beneficial.  If it fails, the next direction is answer-only or rationale-dropout on VTS rows rather than continuous loss weighting.

## Monitoring

Use:

```bash
python scripts/summarize_auto_research_runs.py --root /mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
```

The monitor reads only local files, so it works even when Slurm commands are unavailable.  It reports latest loss, epoch, checkpoint availability, serious error count, and completed eval metrics for both active ablations.

## Early checkpoint evaluation

Both active runs reached `checkpoint-1000` without serious errors.  Because the training jobs use `save_total_limit=1`, the checkpoint can disappear after later saves.  To make early evaluation robust, `scripts/snapshot_and_submit_auto_research_eval.sh` first creates a hardlink snapshot under `runs/auto_research_eval_snapshots/` and then submits vLLM eval on the snapshot.

Submitted early evals:

- `vtsw025 checkpoint-1000`: eval job `1692946`, snapshot `runs/auto_research_eval_snapshots/vtsw025_pvrdsg_checkpoint-1000_20260529_034439`, output `lmms_eval_results/auto_research_vtsw025_pvrdsg_checkpoint-1000_20260529_034439_vllm_parsefix`.
- `vtsanswer checkpoint-1000`: eval job `1692947`, snapshot `runs/auto_research_eval_snapshots/vtsanswer_pvrdsg_checkpoint-1000_20260529_034439`, output `lmms_eval_results/auto_research_vtsanswer_pvrdsg_checkpoint-1000_20260529_034439_vllm_parsefix`.

These early evals are for direction only; the main comparison remains the checkpoint-6000 eval jobs (`1692900`, `1692905`).

Status update: the full training jobs reached `checkpoint-2000` without serious errors.  Early eval job `1692946` is running normally and generating requests; job `1692947` is still pending on `QOSGrpNodeLimit`.  No early eval result JSON has been written yet.

## Parallel ablation: VTS answer-only supervision

The second ablation tests the same hypothesis more directly.  Instead of scaling the entire VTS autoregressive loss, it masks VTS-row labels outside the `<answer>...</answer>` span:

\[
\mathcal{L}_{\mathrm{vts}} =
-\sum_{t \in \mathrm{answer\ span}}\log p_\theta(y_t\mid y_{<t}, u_i^{\mathrm{vts}}).
\]

Original replay rows still train the full `<think>...</think><answer>...</answer>` target.  VTS rows still receive PVRD-SG.  This asks whether VTS examples should teach final answer behavior and prompt-region binding, while leaving chain-of-thought style mostly anchored by the original replay rows.

Implementation:

- `PVRD_VTS_ANSWER_ONLY_LOSS=1`
- `PVRD_VTS_ANSWER_INCLUDE_TAGS=1`
- `PVRD_VTS_SFT_WEIGHT=1.0`
- `PVRD_SG_LAMBDA=0.15`
- `PRMLP_ENABLED=0`
- Local validation: `tests/test_pvrd_answer_only_labels.py` verifies that only the `<answer>...</answer>` span remains supervised on VTS rows and that missing answer tags fall back to the original labels.
- Test runner: `scripts/run_lightweight_tests.sh` runs the answer-only label mask tests plus prompt-region utility tests without requiring `pytest`.

Submitted run:

- Train job: `1692904`
- Eval job: `1692905`
- Train dir: `runs/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829`
- Eval dir: `lmms_eval_results/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829_vllm_parsefix`
- Status update: job `1692904` started on `fs-mbz-gpu-449` and entered the training loop; early loss/grad norm logs are normal.  Eval job `1692905` remains dependency-pending.

Success criterion:

- If answer-only VTS beats `w_vts=0.25`, the main damage was likely VTS rationale imitation.
- If `w_vts=0.25` beats answer-only, some VTS reasoning supervision is useful but needs reduced strength.
- If both fail on MathVista, the next step should move away from VTS target imitation entirely and rely on prompt-region contrastive/latent objectives plus original replay.

## Eval infrastructure fix: EasyR1/vLLM processor sidecars

The first `vtsanswer checkpoint-1000` early eval (`1692947`) failed before generation.  The failure was not a model result: vLLM could not load the Qwen3-VL image processor from the EasyR1-compatible checkpoint directory because the snapshot lacked processor sidecar files such as `preprocessor_config.json`.

Fix:

- `scripts/prepare_easyr1_model_dir.py` now supplements missing processor/tokenizer sidecar files from explicit fallback directories, the checkpoint parents, or the local Qwen3-VL-4B-Instruct HF snapshot.
- `scripts/pvrd_lmms_eval_vllm.slurm.sh` passes the Qwen3-VL-4B-Instruct HF snapshot as the default fallback directory when preparing an EasyR1-compatible model path.
- The eval script no longer hardcodes a W&B key; it uses the caller environment instead.
- Local validation created a compatibility directory from the broken `vtsanswer checkpoint-1000` snapshot and confirmed that `preprocessor_config.json`, `video_preprocessor_config.json`, `vocab.json`, `merges.txt`, tokenizer files, and configs are present.

Retry:

- Failed eval: `1692947`
- Retry eval: `1693005`
- Retry output: `lmms_eval_results/auto_research_vtsanswer_pvrdsg_checkpoint-1000_20260529_034439_retry_vllm_parsefix`

This is an infrastructure fix only.  It should not change model weights or training behavior, but it should make snapshot and final checkpoint evals robust when checkpoint directories are missing processor files.

Follow-up: retry job `1693005` was cancelled because Slurm `--export` parsed the comma-separated TASKS list and truncated evaluation to only `mathvista_testmini_cot`.  To avoid this class of error, `scripts/resubmit_early_vtsanswer_retry_eval.sh` now exports TASKS in the shell environment and calls `sbatch` without `--export`.  The corrected full early-eval retry is job `1693011`, with output `lmms_eval_results/auto_research_vtsanswer_pvrdsg_checkpoint-1000_20260529_034439_retry2_vllm_parsefix`.

## Early result: VTS loss 0.25 checkpoint-1000

The first early result is available for `vtsw025@checkpoint-1000` (`1692946`).  This checkpoint is very early (`global_step=1000`, epoch `0.1667`), so it should not be treated as the final result.  However, the direction is currently poor:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `w_vts=0.25 + PVRD-SG`, ckpt1000 | 19.1 | 14.1 | 66.1 | 48.9 | 42.6 | 31.5 | 11.1 |

Interpretation:

- This early checkpoint is dominated by the base model on both average original accuracy and average VTS accuracy.
- The strongest damage is on MATH-Vision original, so the issue is not merely VTS channel transfer; early training disrupts the original reasoning interface.
- Because the checkpoint is only 0.167 epoch, the full-run eval is still needed before killing the run.  But this early signal lowers the priority of the `w_vts=0.25` direction unless later checkpoints recover strongly.

A helper script now tracks this automatically:

```bash
python3 scripts/analyze_auto_research_tradeoff.py
```

The pending comparison is `vtsanswer@checkpoint-1000` (`1693011`).  If answer-only does not show the same early collapse, the next method direction should favor answer-only VTS supervision over uniformly downweighted VTS SFT.  If both early variants collapse, the next run should reduce or remove VTS language-modeling loss entirely and rely on original replay plus prompt-region representation objectives.

## Prepared next ablation: VTS LM off + PVRD-SG

A follow-up chain is prepared but not submitted yet:

```bash
bash scripts/submit_pvrd_sg_vtslmoff_scale48k_chain.sh
```

This sets:

- `PVRD_VTS_SFT_WEIGHT=0.0`
- `PVRD_SG_LAMBDA=0.15`
- `PRMLP_ENABLED=0`
- original replay rows still use full SFT loss
- VTS rows contribute prompt-region grounding loss but no autoregressive LM loss

Purpose: if `vtsanswer@checkpoint-1000` also collapses, the next hypothesis is that any VTS target imitation is hurting the base reasoning interface.  This ablation makes VTS rows act only as representation-alignment examples while original replay anchors answer format and reasoning behavior.

## Decision from completed vtsw025 / vtsanswer runs

Both mitigation variants completed, including early and final evals.  Both are dominated by the base model and by the known-good PVRD-SG run.

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `w_vts=0.25 + PVRD-SG`, final | 29.6 | 17.4 | 67.7 | 48.4 | 48.7 | 32.9 | 15.7 |
| `answer-only VTS + PVRD-SG`, final | 29.0 | 16.1 | 67.4 | 48.8 | 48.2 | 32.5 | 15.7 |

Interpretation:

- Downweighting VTS SFT is not sufficient; it still damages original reasoning and does not recover VTS.
- Answer-only VTS supervision is also not sufficient; the collapse is not only caused by imitating noisy VTS rationales.
- The common failure source is likely any autoregressive target pressure on VTS rows in this data/export path, or a mismatch between these target rows and the prompt-region objective.

Action taken:

Submitted `VTS LM off + PVRD-SG`, which keeps original replay full SFT but sets VTS-row LM weight to zero.  VTS rows now contribute prompt-region semantic grounding only.

- Train job: `1719148`
- Eval job: `1719149`
- Train dir: `runs/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001`
- Eval dir: `lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001_vllm_parsefix`
- Config: `PVRD_VTS_SFT_WEIGHT=0.0`, `PVRD_SG_LAMBDA=0.15`, `PRMLP_ENABLED=0`

Early checkpoint snapshot:

- Source checkpoint: `runs/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001/checkpoint-1000`
- Snapshot: `runs/auto_research_eval_snapshots/vtslmoff_pvrdsg_checkpoint-1000_20260609_062112`
- Early eval job: `1719154`
- Early eval dir: `lmms_eval_results/auto_research_vtslmoff_pvrdsg_checkpoint-1000_20260609_062112_vllm_parsefix`

Success criterion:

- It must recover original-view accuracy much closer to base, because VTS rows no longer train target text.
- If VTS accuracy improves over base while original is preserved, this supports the hypothesis that prompt-region representation alignment is useful but VTS target imitation is harmful.
- If VTS remains weak but original recovers, the next step should increase/reshape the representation objective rather than reintroduce VTS LM loss.

## Prepared next ablation: VTS LM off + PVRD-SG + PRMLP

The next representation-only variant is prepared but should be submitted only after inspecting the `VTS LM off + PVRD-SG` early or final eval:

```bash
bash scripts/submit_pvrd_sg_vtslmoff_prmlp_scale48k_chain.sh
```

Configuration:

- `PVRD_VTS_SFT_WEIGHT=0.0`
- `PVRD_SG_LAMBDA=0.15`
- `PRMLP_ENABLED=1`
- `PRMLP_ONLINE_VIEW=main`
- `PRMLP_LAMBDA=0.003`
- `PRMLP_EVERY_N_STEPS=2`
- `PRMLP_MAX_MAIN_TOKENS=4096`
- `PRMLP_MAX_EXTRA_TOKENS=2048`
- `PRMLP_IMAGE_MAX_PIXELS=262144`

Rationale:

- If LM-off recovers original accuracy but leaves VTS below the base/PVRD-SG target, then the method needs stronger representation pressure without returning to VTS autoregressive target imitation.
- PRMLP is deliberately set to the medium-weight regularizer from the paper ablation rather than the older `0.05` pilot value, because strong/frequent latent matching can compete with SFT.
- This variant tests whether the prompt crop latent target helps recover VTS when VTS rows are used only as alignment data.

Snapshot support:

- `scripts/snapshot_and_submit_auto_research_eval.sh` now accepts `vtslmoff_prmlp`, with `TRAIN_DIR` supplied explicitly once the run is submitted.

## New retention diagnostic: original-only 24K SFT

The `VTS LM off + PVRD-SG` early checkpoint improves Avg V over the two failed VTS-target variants, but it is still dominated by the base model and shows severe original-view retention loss:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `VTS LM off + PVRD-SG`, ckpt1000 | 22.0 | 17.8 | 63.6 | 56.4 | 42.8 | 37.1 | 5.7 |

This suggests that removing VTS autoregressive loss is not enough.  The next required diagnostic is whether the original replay rows and generated reasoning targets alone already damage original benchmark behavior under the current full-SFT recipe.

Action implemented:

```bash
python3 scripts/export_original_only_retention_dataset.py --update-dataset-info
bash scripts/submit_original_only_retention_control_chain.sh
```

Generated dataset:

- Dataset name: `pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609`
- File: `/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data/pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609.json`
- Rows: `24,000`
- Fields: `messages`, `images`, `system`
- Removed fields: all `sg_*`, `pvrd_*`, and `cvsa_*` metadata

Run configuration:

- Same stable Qwen3-VL-4B full-SFT config as the failed variants
- Dataset: original replay rows only
- `PVRD_SG_ENABLED=0`
- `PVRD_SG_LAMBDA=0.0`
- `PRMLP_ENABLED=0`
- `num_train_epochs=1.0`
- `save_total_limit=1`
- Expected checkpoint: `checkpoint-3000`

Decision rule:

- If original-only retains original-view accuracy close to base, the retention collapse is caused by VTS rows and/or prompt-region objectives; next test `VTS LM off + PVRD-SG + PRMLP` with representation-only VTS rows.
- If original-only also collapses, the main failure is the 24K original replay target distribution or full-SFT recipe; next fixes should be lower learning rate, LoRA/frozen language model, stricter target filtering, or a smaller supervised update before adding VTS objectives.

Submitted original-only retention control:

- Train job: `1719307`
- Eval job: `1719308`
- Train dir: `runs/original_only_retention_origonly24k_sft_retention_20260609_071141`
- Eval dir: `lmms_eval_results/original_only_retention_origonly24k_sft_retention_20260609_071141_vllm_parsefix`
- Model path for eval: `checkpoint-3000`

This run is intentionally no-aux: `PVRD_SG_ENABLED=0`, `PVRD_SG_LAMBDA=0.0`, `PRMLP_ENABLED=0`.

Early eval helper update:

`scripts/snapshot_and_submit_auto_research_eval.sh` now accepts `origonly24k`:

```bash
TRAIN_DIR=runs/original_only_retention_origonly24k_sft_retention_20260609_071141 \
  bash scripts/snapshot_and_submit_auto_research_eval.sh origonly24k checkpoint-1000
```

Use this once `checkpoint-1000` exists to get a faster retention signal before the final `checkpoint-3000` eval finishes.

Submitted original-only early eval:

- Source checkpoint: `runs/original_only_retention_origonly24k_sft_retention_20260609_071141/checkpoint-1000`
- Snapshot: `runs/auto_research_eval_snapshots/origonly24k_sft_retention_checkpoint-1000_20260609_072512`
- Eval job: `1719386`
- Eval dir: `lmms_eval_results/auto_research_origonly24k_sft_retention_checkpoint-1000_20260609_072512_vllm_parsefix`

This early eval is the fastest retention signal for whether the current original-only full-SFT recipe already damages the base benchmark interface.

## Result: VTS LM off + PVRD-SG final

Final eval for `VTS LM off + PVRD-SG` completed:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `VTS LM off + PVRD-SG`, ckpt1000 | 22.0 | 17.8 | 63.6 | 56.4 | 42.8 | 37.1 | 5.7 |
| `VTS LM off + PVRD-SG`, final | 25.0 | 22.0 | 67.9 | 55.3 | 46.5 | 38.7 | 7.8 |

Result JSON:

`lmms_eval_results/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001_vllm_parsefix/easyr1_model_compat__pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001__checkpoint-6000/20260609_150751_results.json`

Interpretation:

- Removing VTS-row LM loss improves Avg V relative to the two failed VTS-target variants, but the run is still dominated by the base model and known-good PVRD-SG.
- Original-view retention remains badly damaged: Avg O is `46.5`, which is `-16.2` below base.
- The main open diagnostic is now `original-only 24K retention control`.  If it also collapses, the current 24K original replay targets or full-SFT recipe are the cause; adding PRMLP will not solve the retention problem.

Prepared fallback retention repair: original-only low-LR full SFT

Script:

```bash
bash scripts/submit_original_only_lowlr_retention_control_chain.sh
```

Configuration:

- Dataset: `pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609`
- No auxiliary objectives: `PVRD_SG_ENABLED=0`, `PRMLP_ENABLED=0`
- Same full-SFT recipe except `learning_rate=5.0e-7`
- Expected checkpoint: `checkpoint-3000`

Do not submit this before reading the original-only full-SFT eval.  It should be used only if the current original-only control also damages original-view benchmark performance, because then update magnitude is the next simplest hypothesis.

## Submitted fallback retention repair: original-only low-LR full SFT

The original-only early eval already shows severe retention collapse:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `Original-only 24K`, ckpt1000 | 23.7 | 21.1 | 65.8 | 55.1 | 44.7 | 38.1 | 6.7 |

This means the current retention failure is not specific to VTS rows, PRMLP, or PVRD-SG.  Original replay targets alone can damage the base benchmark interface under the current full-SFT update.  Therefore the next useful diagnostic is lower update magnitude, not another representation objective.

Submitted low-LR original-only retention control:

- Train job: `1719410`
- Eval job: `1719411`
- Train dir: `runs/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926`
- Eval dir: `lmms_eval_results/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926_vllm_parsefix`
- Dataset: `pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609`
- Auxiliary objectives: none (`PVRD_SG_ENABLED=0`, `PVRD_SG_LAMBDA=0.0`, `PRMLP_ENABLED=0`)
- Only recipe change versus the original-only control: `learning_rate=5.0e-7`
- Checkpoint policy: `save_total_limit=1`, expected eval checkpoint `checkpoint-3000`

Decision rule:

- If low LR restores original-view accuracy close to base, the next method run should use low LR or reduced trainable scope for balanced replay + PVRD-SG/PRMLP.
- If low LR still collapses, the target distribution itself is suspect; next steps should be target filtering, LoRA/frozen language backbone, or replaying true base-format answers instead of generated SymThink traces.
- Do not submit `VTS LM off + PVRD-SG + PRMLP` until the retention repair is understood; PRMLP cannot fix a base-interface collapse caused by original-only SFT.

## Result: original-only 24K full-SFT retention control final

The final original-only retention eval completed and confirms the early signal:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Known good PVRD-SG | 52.4 | 49.8 | 74.2 | 70.8 | 63.3 | 60.3 | 3.0 |
| `Original-only 24K`, ckpt1000 | 23.7 | 21.1 | 65.8 | 55.1 | 44.7 | 38.1 | 6.7 |
| `Original-only 24K`, final | 24.3 | 20.4 | 67.0 | 57.2 | 45.7 | 38.8 | 6.9 |

Result JSON:

`lmms_eval_results/original_only_retention_origonly24k_sft_retention_20260609_071141_vllm_parsefix/easyr1_model_compat__original_only_retention_origonly24k_sft_retention_20260609_071141__checkpoint-3000/20260609_154154_results.json`

Interpretation:

- The original-only control is dominated by the base model on both Avg O and Avg V.
- This rules out VTS-row LM loss, PVRD-SG, and PRMLP as the primary cause of the latest collapse.
- The failure is now localized to the current 24K original replay targets plus full-SFT update recipe.
- The low-LR original-only control (`1719410` train, `1719411` eval) is now the right next experiment.  If it recovers retention, future PVRD-SG/PRMLP runs should inherit the lower LR or a smaller trainable scope.

## Submitted early eval: original-only low-LR checkpoint-1000

The low-LR original-only run reached `checkpoint-1000` with stable training dynamics:

- Loss around checkpoint-1000: roughly `0.44--0.48`
- Grad norm around checkpoint-1000: roughly `3.4--4.4`
- No serious errors in the train log at the time of snapshot

Submitted early eval:

- Source checkpoint: `runs/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926/checkpoint-1000`
- Snapshot: `runs/auto_research_eval_snapshots/origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858`
- Eval job: `1719462`
- Eval dir: `lmms_eval_results/auto_research_origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858_vllm_parsefix`

Fast decision rule:

- If checkpoint-1000 already restores Avg O near the base model, low LR is enough to prevent the catastrophic original-interface collapse.
- If checkpoint-1000 remains around the failed full-LR original-only result, lower LR alone is unlikely to fix the target-distribution problem, and the next repair should reduce trainable scope or filter/rewrite the original replay targets.

## Prepared decision branches after low-LR retention result

Two follow-up scripts are prepared but not submitted.  They are mutually exclusive decision branches depending on the low-LR original-only result.

### Branch A: low-LR balanced PVRD-SG + PRMLP

Script:

```bash
bash scripts/submit_pvrd_sg_prmlp_lowlr_scale48k_chain.sh
```

Use this only if low-LR original-only restores original-view retention.  It migrates the safer update magnitude to the real balanced method run:

- Dataset: `pvrd_sg_math_thinking_sft_prmlp_scale48k_bal1to1_symthink_20260423_052441`
- `PVRD_VTS_SFT_WEIGHT=1.0`
- `PVRD_SG_LAMBDA=0.15`
- `PRMLP_ENABLED=1`
- `PRMLP_LAMBDA=0.003`
- `PRMLP_EVERY_N_STEPS=2`
- `learning_rate=5.0e-7`
- `save_total_limit=1`

Rationale: if update magnitude was the cause, keep the method but lower the update strength.

### Branch B: original-only freeze-LM retention control

Script:

```bash
bash scripts/submit_original_only_freezelm_retention_control_chain.sh
```

Use this only if low-LR original-only still collapses.  It freezes the language model while keeping the same original-only dataset and eval path:

- Dataset: `pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609`
- No PVRD-SG / PRMLP auxiliary objectives
- `freeze_language_model=true`
- `learning_rate=1.0e-5`
- `save_total_limit=1`

Rationale: if lower LR does not fix retention, the next clean test is whether updating the LLM backbone itself is the destructive component.  This avoids LoRA merge/eval complications and keeps the current vLLM eval path unchanged.

## Low-LR original-only training completed

The low-LR original-only train job completed successfully:

- Train job: `1719410`
- Final checkpoint: `runs/original_only_retention_origonly24k_sft_lowlr_retention_20260609_080926/checkpoint-3000`
- Final eval job: `1719411` started automatically after training
- Early eval job: `1719462` is still running on checkpoint-1000
- Final logged loss: `0.411`
- Final grad norm: `3.351`
- Serious errors: none detected by the summary script

The next decision still waits on eval metrics.  The key comparison is against the failed original-only full-LR result (`Avg O=45.7`, `Avg V=38.8`) and the base model (`Avg O=62.7`, `Avg V=48.4`).

## Result: low-LR original-only checkpoint-1000 early eval

The low-LR checkpoint-1000 eval completed:

| Run | MV O | MV V | Vista O | Vista V | Avg O | Avg V | Avg G |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Qwen3-VL-4B-Instruct | 51.6 | 34.8 | 73.7 | 61.9 | 62.7 | 48.4 | 14.3 |
| Full-LR original-only final | 24.3 | 20.4 | 67.0 | 57.2 | 45.7 | 38.8 | 6.9 |
| Low-LR original-only ckpt1000 | 24.3 | 21.1 | 68.1 | 57.1 | 46.2 | 39.1 | 7.1 |

Result JSON:

`lmms_eval_results/auto_research_origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858_vllm_parsefix/easyr1_model_compat__auto_research_eval_snapshots__origonly24k_sft_lowlr_retention_checkpoint-1000_20260609_081858/20260609_161948_results.json`

Interpretation:

- Lower LR gives only a small improvement over the failed full-LR original-only run.
- It remains dominated by the base model on both Avg O and Avg V.
- Therefore the current collapse is not fixed by update magnitude alone.
- The next clean hypothesis is trainable scope: updating the language backbone on these SymThink original replay targets may be the destructive component.

Submitted freeze-LM original-only retention control:

- Train job: `1719505`
- Eval job: `1719506`
- Train dir: `runs/original_only_retention_origonly24k_sft_freezelm_retention_20260609_085414`
- Eval dir: `lmms_eval_results/original_only_retention_origonly24k_sft_freezelm_retention_20260609_085414_vllm_parsefix`
- Dataset: `pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609`
- No auxiliary objectives
- Key change: `freeze_language_model=true`
- LR: `1.0e-5`
- Expected checkpoint: `checkpoint-3000`

Decision rule:

- If freeze-LM preserves original-view accuracy, the next method recipe should avoid updating the LLM backbone and put prompt-region objectives on projector/visual-side trainable parameters.
- If freeze-LM still collapses or cannot improve, the target distribution/eval format itself is the main suspect; next step should be rewriting/filtering original replay targets rather than changing optimization knobs.

## Sanity check: freeze-LM control is using the intended trainable scope

The freeze-LM train log confirms the intended parameter scope:

- `freeze_language_model=true` is present in the launched command.
- LLaMA-Factory reports: `Set language model not trainable: ['language_model', 'lm_head']`.
- LLaMA-Factory reports: `trainable params: 27,271,680 || all params: 4,437,815,808 || trainable%: 0.6145`.
- Early training is stable: initial grad norms are around `0.16--0.31`, with no serious errors in the visible log.

This is a clean trainable-scope diagnostic: if this run preserves benchmark performance, the previous collapse is likely caused by updating the LLM backbone on the current SymThink replay targets.  If it still collapses, the problem is more likely target/eval-format mismatch rather than update magnitude or trainable scope alone.
