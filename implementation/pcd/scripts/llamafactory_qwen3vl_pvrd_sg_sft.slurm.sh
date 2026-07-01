#!/usr/bin/env bash
#SBATCH --job-name=lf_qwen3vl_pvrd_sg_sft
#SBATCH --output=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd/logs/%x_%j.log
#SBATCH --error=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd/logs/%x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00

set -euo pipefail

ROOT=/mnt/weka/home/yongxin.wang/workspace/Auto-claude-code-research-in-sleep/implementation/pcd
LLAMAFACTORY_ROOT=/mnt/weka/home/yongxin.wang/workspace/LlamaFactory
CONFIG=${CONFIG:-$ROOT/configs/qwen3vl_pvrd_sg_sft_llamafactory.yaml}
DATASET_DIR=${DATASET_DIR:-/mnt/weka/home/yongxin.wang/workspace/LlamaFactory/data}
DATASET_NAME=${DATASET_NAME:-pvrd_sg_math_thinking_sft}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runs/llamafactory_qwen3vl_4b_thinking_pvrd_sg_sft}
EXTRA_ARGS=${EXTRA_ARGS:-}
HF_HOME=${HF_HOME:-$ROOT/.cache/huggingface}
CONDA_ENV=${CONDA_ENV:-llamafactory}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}

mkdir -p "$ROOT/logs"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/.bashrc" ]]; then
  source "$HOME/.bashrc"
fi

conda activate "$CONDA_ENV"

export HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRITON_CACHE_DIR="$HF_HOME/triton"
export PYTHONPATH="$LLAMAFACTORY_ROOT/src:${PYTHONPATH:-}"
export PVRD_SG_ENABLED="${PVRD_SG_ENABLED:-1}"
export PVRD_SG_LAMBDA="${PVRD_SG_LAMBDA:-0.15}"
export PVRD_VTS_SFT_WEIGHT="${PVRD_VTS_SFT_WEIGHT:-1.0}"
export PVRD_VTS_ANSWER_ONLY_LOSS="${PVRD_VTS_ANSWER_ONLY_LOSS:-0}"
export PVRD_VTS_ANSWER_INCLUDE_TAGS="${PVRD_VTS_ANSWER_INCLUDE_TAGS:-1}"
export PVRD_DISTILL_ENABLED="${PVRD_DISTILL_ENABLED:-0}"
export PVRD_DISTILL_LAMBDA="${PVRD_DISTILL_LAMBDA:-0.10}"
export PVRD_HIDDEN_DISTILL_LAMBDA="${PVRD_HIDDEN_DISTILL_LAMBDA:-0.05}"
export PVRD_DISTILL_TEMPERATURE="${PVRD_DISTILL_TEMPERATURE:-2.0}"
export PVRD_DISTILL_ONLY_EXACT="${PVRD_DISTILL_ONLY_EXACT:-1}"
export PVRD_TEACHER_CACHE_PATH="${PVRD_TEACHER_CACHE_PATH:-}"
export PRMLP_ENABLED="${PRMLP_ENABLED:-0}"
export PRMLP_ONLINE_VIEW="${PRMLP_ONLINE_VIEW:-masked}"
export PRMLP_LAMBDA="${PRMLP_LAMBDA:-0.05}"
export PRMLP_MASK_RATIO="${PRMLP_MASK_RATIO:-0.35}"
export PRMLP_BLOCK_SIZE="${PRMLP_BLOCK_SIZE:-32}"
export PRMLP_CUTOFF_LEN="${PRMLP_CUTOFF_LEN:-8192}"
export PRMLP_EVERY_N_STEPS="${PRMLP_EVERY_N_STEPS:-1}"
export PRMLP_MAX_MAIN_TOKENS="${PRMLP_MAX_MAIN_TOKENS:-0}"
export PRMLP_MAX_EXTRA_TOKENS="${PRMLP_MAX_EXTRA_TOKENS:-0}"
export PRMLP_LOG_EVERY_N_STEPS="${PRMLP_LOG_EVERY_N_STEPS:-0}"
export PRMLP_PROMPT_TEXT="${PRMLP_PROMPT_TEXT:-Help me solve the problem}"
export PRMLP_DUMMY_ANSWER="${PRMLP_DUMMY_ANSWER:-0}"
export PRMLP_IMAGE_MAX_PIXELS="${PRMLP_IMAGE_MAX_PIXELS:-1048576}"
export PRMLP_IMAGE_MIN_PIXELS="${PRMLP_IMAGE_MIN_PIXELS:-1024}"
export PRMLP_DEBUG="${PRMLP_DEBUG:-0}"
export SKIP_ROOT_FINAL_SAVE="${SKIP_ROOT_FINAL_SAVE:-0}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export NCCL_DEBUG="${NCCL_DEBUG:-}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export MASTER_PORT
TORCHRUN_LOG_DIR="${TORCHRUN_LOG_DIR:-}"
TORCHRUN_REDIRECTS="${TORCHRUN_REDIRECTS:-}"
TORCHRUN_TEE="${TORCHRUN_TEE:-}"
cd "$ROOT"

CMD=(torchrun --nnodes 1 --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT")
if [[ -n "$TORCHRUN_LOG_DIR" ]]; then
  mkdir -p "$TORCHRUN_LOG_DIR"
  CMD+=(--log-dir "$TORCHRUN_LOG_DIR")
fi
if [[ -n "$TORCHRUN_REDIRECTS" ]]; then
  CMD+=(--redirects "$TORCHRUN_REDIRECTS")
fi
if [[ -n "$TORCHRUN_TEE" ]]; then
  CMD+=(--tee "$TORCHRUN_TEE")
fi
CMD+=("$ROOT/scripts/run_pvrd_sg_llamafactory.py" "$CONFIG"
  dataset_dir="$DATASET_DIR"
  dataset="$DATASET_NAME"
  output_dir="$OUTPUT_DIR")

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"
