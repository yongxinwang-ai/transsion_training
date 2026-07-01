# Prompt-Region Grounding LLaMA-Factory Delta

This sanitized package contains the method code used for prompt-region grounding on top of LLaMA-Factory. It includes the VTS paired-data export path, PVRD-SG/PRMLP loss utilities, Slurm launchers, diagnostic lmms-eval task logic, and lightweight tests.

It intentionally excludes checkpoints, run logs, datasets, caches, private credentials, and obsolete failed-run artifacts.

## Main Method

- Balanced same-target replay: each source problem contributes an original-view row and a VTS-view row with the same `<think>...</think><answer>...</answer>` target.
- PVRD-SG: aligns pooled hidden states from the rendered prompt panel with a frozen text embedding of the source question.
- PRMLP: aligns the prompt region in the full VTS image with a stop-gradient clean prompt-crop representation.
- Optional RLVR: answer-verifiable refinement after the supervised prompt-region grounding stage.

## Primary Entry Points

Training and export:

```bash
cd implementation/pcd
python scripts/export_llamafactory_pvrd_sg_dataset.py --help
bash scripts/submit_pvrd_sg_prmlp_scale48k_chain.sh
```

Core implementation files:

- `implementation/pcd/scripts/run_pvrd_sg_llamafactory.py`
- `implementation/pcd/scripts/prompt_region_ssl_utils.py`
- `implementation/pcd/scripts/llamafactory_qwen3vl_pvrd_sg_sft.slurm.sh`
- `implementation/pcd/configs/qwen3vl_pvrd_sg_prmlp_llamafactory_nodz_instruct_scale48k_stable.yaml`

Diagnostics and controls:

- `implementation/pcd/eval_tasks/vts_ocr_only/`
- `implementation/pcd/eval_tasks/vts_ocr_llm/`
- `implementation/pcd/eval_tasks/vts_conflict_control/`
- `implementation/pcd/eval_tasks/vts_prompt_occlusion/`
- `implementation/pcd/scripts/analyze_vts_pairwise_mechanisms.py`

Retention controls:

- `implementation/pcd/scripts/export_original_only_retention_dataset.py`
- `implementation/pcd/scripts/submit_original_only_retention_control_chain.sh`
- `implementation/pcd/scripts/submit_original_only_lowlr_retention_control_chain.sh`
- `implementation/pcd/scripts/submit_original_only_freezelm_retention_control_chain.sh`

Prepared method branch after retention repair:

- `implementation/pcd/scripts/submit_pvrd_sg_prmlp_lowlr_scale48k_chain.sh`

## Lightweight Verification

```bash
cd implementation/pcd
bash scripts/run_lightweight_tests.sh
```

## Notes

The scripts assume the original cluster layout and a local LLaMA-Factory checkout. Paths should be edited in the Slurm launchers before running on another machine. Long runs should use Slurm; tests and data-schema checks can be run locally.

# transsion_training

## Portable LLaMA-Factory Training Entry Points

The `llamafactory_impl/` folder contains sanitized LLaMA-Factory configs and launch scripts for running Qwen3-VL Transsion SFT on another cluster. It does not include data, checkpoints, cache files, or private credentials.
