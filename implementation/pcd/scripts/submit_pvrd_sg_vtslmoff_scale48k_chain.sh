#!/usr/bin/env bash
set -euo pipefail

# Pure prompt-region grounding ablation:
# original replay keeps full SFT; VTS rows contribute PVRD-SG but no LM loss.
# Submit this only if answer-only VTS supervision also shows early original/VTS
# collapse, because it removes VTS target imitation entirely.

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
TS=${TS:-$(date -u +%Y%m%d_%H%M%S)}

ABLATION_NAME=${ABLATION_NAME:-vtslmoff_pvrdsg}
PVRD_VTS_SFT_WEIGHT=${PVRD_VTS_SFT_WEIGHT:-0.0}
PVRD_SG_LAMBDA=${PVRD_SG_LAMBDA:-0.15}
PRMLP_ENABLED=${PRMLP_ENABLED:-0}

TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_vtslmoff_scale48k_${ABLATION_NAME}_${TS}}
EVAL_DIR=${EVAL_DIR:-$ROOT/lmms_eval_results/pvrd_sg_vtslmoff_scale48k_${ABLATION_NAME}_${TS}_vllm_parsefix}

export ABLATION_NAME PVRD_VTS_SFT_WEIGHT PVRD_SG_LAMBDA PRMLP_ENABLED TRAIN_DIR EVAL_DIR

bash "$ROOT/scripts/submit_pvrd_sg_vtsloss_scale48k_chain.sh"
