#!/usr/bin/env bash
set -euo pipefail

# Prepared branch: use only if original-only low-LR retention recovers the base
# interface.  This migrates the safer update magnitude to the actual balanced
# VTS/original method run.

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
TS=${TS:-$(date -u +%Y%m%d_%H%M%S)}

DATASET_NAME=${DATASET_NAME:-pvrd_sg_math_thinking_sft_prmlp_scale48k_bal1to1_symthink_20260423_052441}
ABLATION_NAME=${ABLATION_NAME:-lowlr_balanced_pvrdsg_prmlp_lam003}
TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_lowlr_scale48k_${ABLATION_NAME}_${TS}}
EVAL_DIR=${EVAL_DIR:-$ROOT/lmms_eval_results/pvrd_sg_lowlr_scale48k_${ABLATION_NAME}_${TS}_vllm_parsefix}
MODEL_PATH=${MODEL_PATH:-$TRAIN_DIR/checkpoint-6000}

export DATASET_NAME ABLATION_NAME TRAIN_DIR EVAL_DIR MODEL_PATH
export PVRD_SG_ENABLED=${PVRD_SG_ENABLED:-1}
export PVRD_SG_LAMBDA=${PVRD_SG_LAMBDA:-0.15}
export PVRD_VTS_SFT_WEIGHT=${PVRD_VTS_SFT_WEIGHT:-1.0}
export PVRD_DISTILL_ENABLED=${PVRD_DISTILL_ENABLED:-0}
export PRMLP_ENABLED=${PRMLP_ENABLED:-1}
export PRMLP_ONLINE_VIEW=${PRMLP_ONLINE_VIEW:-main}
export PRMLP_LAMBDA=${PRMLP_LAMBDA:-0.003}
export PRMLP_EVERY_N_STEPS=${PRMLP_EVERY_N_STEPS:-2}
export PRMLP_MAX_MAIN_TOKENS=${PRMLP_MAX_MAIN_TOKENS:-4096}
export PRMLP_MAX_EXTRA_TOKENS=${PRMLP_MAX_EXTRA_TOKENS:-2048}
export PRMLP_IMAGE_MAX_PIXELS=${PRMLP_IMAGE_MAX_PIXELS:-262144}
export EVAL_BACKEND=${EVAL_BACKEND:-vllm}
export TASKS=${TASKS:-mathvista_testmini_cot,mathvista_testmini_prompt_in_image,mathvision_testmini,mathvision_testmini_prompt_in_image}
export EXTRA_ARGS=${EXTRA_ARGS:-"num_train_epochs=1.0 learning_rate=5.0e-7 save_steps=1000 save_total_limit=1 report_to=none"}

bash "$ROOT/scripts/submit_pvrd_sg_vtsloss_scale48k_chain.sh"
