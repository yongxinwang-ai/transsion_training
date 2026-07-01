#!/usr/bin/env bash
set -euo pipefail

# Representation-only VTS ablation with both local prompt-region objectives:
# original replay keeps full SFT; VTS rows contribute PVRD-SG + PRMLP but no LM loss.
# Submit after inspecting the VTS-LM-off early/final eval, unless there is spare
# capacity and the next question is whether latent prompt-region regularization
# recovers VTS without reintroducing VTS target imitation.

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
TS=${TS:-$(date -u +%Y%m%d_%H%M%S)}

ABLATION_NAME=${ABLATION_NAME:-vtslmoff_pvrdsg_prmlp}
PVRD_VTS_SFT_WEIGHT=${PVRD_VTS_SFT_WEIGHT:-0.0}
PVRD_SG_LAMBDA=${PVRD_SG_LAMBDA:-0.15}

PRMLP_ENABLED=${PRMLP_ENABLED:-1}
PRMLP_ONLINE_VIEW=${PRMLP_ONLINE_VIEW:-main}
PRMLP_LAMBDA=${PRMLP_LAMBDA:-0.003}
PRMLP_MASK_RATIO=${PRMLP_MASK_RATIO:-0.35}
PRMLP_BLOCK_SIZE=${PRMLP_BLOCK_SIZE:-32}
PRMLP_CUTOFF_LEN=${PRMLP_CUTOFF_LEN:-4096}
PRMLP_EVERY_N_STEPS=${PRMLP_EVERY_N_STEPS:-2}
PRMLP_MAX_MAIN_TOKENS=${PRMLP_MAX_MAIN_TOKENS:-4096}
PRMLP_MAX_EXTRA_TOKENS=${PRMLP_MAX_EXTRA_TOKENS:-2048}
PRMLP_LOG_EVERY_N_STEPS=${PRMLP_LOG_EVERY_N_STEPS:-50}
PRMLP_IMAGE_MAX_PIXELS=${PRMLP_IMAGE_MAX_PIXELS:-262144}
PRMLP_IMAGE_MIN_PIXELS=${PRMLP_IMAGE_MIN_PIXELS:-1024}
PRMLP_DEBUG=${PRMLP_DEBUG:-0}

TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_vtslmoff_scale48k_${ABLATION_NAME}_${TS}}
EVAL_DIR=${EVAL_DIR:-$ROOT/lmms_eval_results/pvrd_sg_vtslmoff_scale48k_${ABLATION_NAME}_${TS}_vllm_parsefix}

export ABLATION_NAME PVRD_VTS_SFT_WEIGHT PVRD_SG_LAMBDA
export PRMLP_ENABLED PRMLP_ONLINE_VIEW PRMLP_LAMBDA PRMLP_MASK_RATIO PRMLP_BLOCK_SIZE PRMLP_CUTOFF_LEN
export PRMLP_EVERY_N_STEPS PRMLP_MAX_MAIN_TOKENS PRMLP_MAX_EXTRA_TOKENS PRMLP_LOG_EVERY_N_STEPS
export PRMLP_IMAGE_MAX_PIXELS PRMLP_IMAGE_MIN_PIXELS PRMLP_DEBUG
export TRAIN_DIR EVAL_DIR

bash "$ROOT/scripts/submit_pvrd_sg_vtsloss_scale48k_chain.sh"
