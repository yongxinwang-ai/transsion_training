#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
RUN_KEY=${1:?Usage: $0 <vtsw025|vtsanswer|vtslmoff|vtslmoff_prmlp|origonly24k> [checkpoint-name, default checkpoint-1000]}
CKPT_NAME=${2:-checkpoint-1000}
TS=${TS:-$(date -u +%Y%m%d_%H%M%S)}

case "$RUN_KEY" in
  vtsw025)
    TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_vtsloss_scale48k_vtsw025_pvrdsg_20260529_032153}
    LABEL=${LABEL:-vtsw025_pvrdsg}
    ;;
  vtsanswer)
    TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_vtsanswer_scale48k_vtsanswer_pvrdsg_20260529_032829}
    LABEL=${LABEL:-vtsanswer_pvrdsg}
    ;;
  vtslmoff)
    TRAIN_DIR=${TRAIN_DIR:-$ROOT/runs/pvrd_sg_vtslmoff_scale48k_vtslmoff_pvrdsg_20260609_061001}
    LABEL=${LABEL:-vtslmoff_pvrdsg}
    ;;
  vtslmoff_prmlp)
    TRAIN_DIR=${TRAIN_DIR:?TRAIN_DIR must point to the vtslmoff+PRMLP run directory}
    LABEL=${LABEL:-vtslmoff_pvrdsg_prmlp}
    ;;
  origonly24k)
    TRAIN_DIR=${TRAIN_DIR:?TRAIN_DIR must point to the original-only retention run directory}
    LABEL=${LABEL:-origonly24k_sft_retention}
    ;;
  *)
    echo "Unknown RUN_KEY=$RUN_KEY; expected vtsw025, vtsanswer, vtslmoff, vtslmoff_prmlp, or origonly24k" >&2
    exit 2
    ;;
esac

SRC_CKPT=$TRAIN_DIR/$CKPT_NAME
if [[ ! -d "$SRC_CKPT" ]]; then
  echo "Checkpoint does not exist yet: $SRC_CKPT" >&2
  exit 2
fi

SNAPSHOT_ROOT=${SNAPSHOT_ROOT:-$ROOT/runs/auto_research_eval_snapshots}
SNAPSHOT_DIR=${SNAPSHOT_DIR:-$SNAPSHOT_ROOT/${LABEL}_${CKPT_NAME}_${TS}}
EVAL_DIR=${EVAL_DIR:-$ROOT/lmms_eval_results/auto_research_${LABEL}_${CKPT_NAME}_${TS}_vllm_parsefix}
TASKS=${TASKS:-mathvista_testmini_cot,mathvista_testmini_prompt_in_image,mathvision_testmini,mathvision_testmini_prompt_in_image}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-iq}
SBATCH_EVAL_TIME=${SBATCH_EVAL_TIME:-720:00:00}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GEN_MAX_NEW_TOKENS=${GEN_MAX_NEW_TOKENS:-1024}
LOG_SAMPLES=${LOG_SAMPLES:-0}
PARSE_REASONING_ANSWER=${PARSE_REASONING_ANSWER:-1}

mkdir -p "$SNAPSHOT_ROOT" "$ROOT/lmms_eval_results" "$ROOT/slurm"
if [[ ! -d "$SNAPSHOT_DIR" ]]; then
  # Hardlink first to avoid duplicating model weights; fall back to normal copy
  # only when hardlinks are unsupported across filesystems.
  if ! cp -al "$SRC_CKPT" "$SNAPSHOT_DIR" 2>/dev/null; then
    cp -a "$SRC_CKPT" "$SNAPSHOT_DIR"
  fi
fi

export MODEL_PATH="$SNAPSHOT_DIR"
export TASKS OUTPUT_DIR="$EVAL_DIR"
export TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE BATCH_SIZE MAX_MODEL_LEN GEN_MAX_NEW_TOKENS LOG_SAMPLES PARSE_REASONING_ANSWER

EVAL_JOB_ID=$(
  sbatch --parsable \
    --account="$SBATCH_ACCOUNT" \
    --job-name="early_${LABEL}_${CKPT_NAME}" \
    --time="$SBATCH_EVAL_TIME" \
    "$ROOT/scripts/pvrd_lmms_eval_vllm.slurm.sh"
)

cat <<OUT
RUN_KEY=${RUN_KEY}
CKPT_NAME=${CKPT_NAME}
SRC_CKPT=${SRC_CKPT}
SNAPSHOT_DIR=${SNAPSHOT_DIR}
EVAL_DIR=${EVAL_DIR}
EVAL_JOB_ID=${EVAL_JOB_ID}
OUT
