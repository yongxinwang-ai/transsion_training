#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl_transsion_sft
#SBATCH --account=iq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=720:00:00
#SBATCH --output=slurm/qwen3vl_transsion_sft_%j.out
#SBATCH --error=slurm/qwen3vl_transsion_sft_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

: "${LLAMA_FACTORY_DIR:?Set LLAMA_FACTORY_DIR to your LLaMA-Factory checkout}"
: "${DATASET_JSONL:?Set DATASET_JSONL to the LLaMA-Factory multimodal JSONL file}"

DATASET_NAME=${DATASET_NAME:-transsion_math_sft}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/qwen3vl-8b-transsion-sft}
CONFIG_OUT=${CONFIG_OUT:-$LLAMA_FACTORY_DIR/examples/train_full/qwen3vl_8b_transsion_sft.runtime.yaml}
CONDA_ENV=${CONDA_ENV:-llamafactory}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}
MASTER_PORT=${MASTER_PORT:-$((20000 + SLURM_JOB_ID % 20000))}
HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}

mkdir -p slurm "$OUTPUT_DIR" "$HF_HOME"

echo "Job: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "LLaMA-Factory: $LLAMA_FACTORY_DIR"
echo "Dataset: $DATASET_JSONL"
echo "Output: $OUTPUT_DIR"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook)"
fi
conda activate "$CONDA_ENV"

python "$IMPL_DIR/scripts/register_transsion_dataset.py" \
  --llamafactory-dir "$LLAMA_FACTORY_DIR" \
  --dataset-name "$DATASET_NAME" \
  --dataset-jsonl "$DATASET_JSONL"

LLAMA_FACTORY_DIR=$LLAMA_FACTORY_DIR \
DATASET_NAME=$DATASET_NAME \
OUTPUT_DIR=$OUTPUT_DIR \
python "$IMPL_DIR/scripts/render_llamafactory_config.py" \
  --template "$IMPL_DIR/config_templates/qwen3vl_8b_transsion_sft.yaml.in" \
  --output "$CONFIG_OUT"

export MASTER_ADDR MASTER_PORT HF_HOME
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PYTHONUNBUFFERED=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cd "$LLAMA_FACTORY_DIR"
torchrun \
  --nnodes "${SLURM_NNODES:-1}" \
  --node_rank "${SLURM_NODEID:-0}" \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  "$LLAMA_FACTORY_DIR/src/llamafactory/launcher.py" \
  "$CONFIG_OUT"
