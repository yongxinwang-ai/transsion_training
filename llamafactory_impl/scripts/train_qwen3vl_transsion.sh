#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
IMPL_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

: "${LLAMA_FACTORY_DIR:?Set LLAMA_FACTORY_DIR to your LLaMA-Factory checkout}"
: "${DATASET_JSONL:?Set DATASET_JSONL to the LLaMA-Factory multimodal JSONL file}"

DATASET_NAME=${DATASET_NAME:-transsion_math_sft}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/outputs/qwen3vl-8b-transsion-sft}
CONFIG_OUT=${CONFIG_OUT:-$LLAMA_FACTORY_DIR/examples/train_full/qwen3vl_8b_transsion_sft.runtime.yaml}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
CONDA_ENV=${CONDA_ENV:-llamafactory}

mkdir -p "$OUTPUT_DIR" "$(dirname "$CONFIG_OUT")"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
fi

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

cd "$LLAMA_FACTORY_DIR"
if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG_OUT"
else
  llamafactory-cli train "$CONFIG_OUT"
fi
