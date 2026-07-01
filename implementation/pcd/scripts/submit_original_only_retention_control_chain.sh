#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
TS=${TS:-$(date -u +%Y%m%d_%H%M%S)}

DATASET_DIR=${DATASET_DIR:-/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data}
SOURCE_DATASET=${SOURCE_DATASET:-$DATASET_DIR/pvrd_sg_math_thinking_sft_prmlp_scale48k_bal1to1_symthink_20260423_052441.json}
DATASET_NAME=${DATASET_NAME:-pvrd_sg_math_thinking_sft_prmlp_scale48k_origonly24k_symthink_20260609}
ABLATION_NAME=${ABLATION_NAME:-origonly24k_sft_retention}
TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/original_only_retention_${ABLATION_NAME}_${TS}}
EVAL_DIR=${EVAL_DIR:-$ROOT/lmms_eval_results/original_only_retention_${ABLATION_NAME}_${TS}_vllm_parsefix}
MODEL_PATH=${MODEL_PATH:-$TRAIN_DIR/checkpoint-3000}

python3 "$ROOT/scripts/export_original_only_retention_dataset.py" \
  --source "$SOURCE_DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --dataset-name "$DATASET_NAME" \
  --max-original 24000 \
  --update-dataset-info

export DATASET_DIR DATASET_NAME ABLATION_NAME TRAIN_DIR EVAL_DIR MODEL_PATH
export PVRD_SG_ENABLED=0
export PVRD_SG_LAMBDA=0.0
export PVRD_VTS_SFT_WEIGHT=1.0
export PVRD_DISTILL_ENABLED=0
export PRMLP_ENABLED=0
export EVAL_BACKEND=${EVAL_BACKEND:-vllm}
export TASKS=${TASKS:-mathvista_testmini_cot,mathvista_testmini_prompt_in_image,mathvision_testmini,mathvision_testmini_prompt_in_image}
export EXTRA_ARGS=${EXTRA_ARGS:-"num_train_epochs=1.0 save_steps=1000 save_total_limit=1 report_to=none"}

bash "$ROOT/scripts/submit_pvrd_sg_vtsloss_scale48k_chain.sh"
