# LLaMA-Factory Transsion Training Entry Points

This folder contains the portable LLaMA-Factory implementation entry points for Transsion/Qwen3-VL SFT training. It is intentionally separated from the method wrapper code in `implementation/pcd` because the other cluster can install a fresh upstream LLaMA-Factory checkout and apply only these runtime configs.

No dataset files, image files, checkpoints, cache directories, or private credentials are included.

## Expected External Inputs

Set these paths on the target cluster:

```bash
export LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory
export DATASET_JSONL=/path/to/llamafactory_sft.jsonl
export OUTPUT_DIR=/path/to/output/qwen3vl-8b-transsion-sft
export CONDA_ENV=llamafactory
```

The dataset JSONL should use LLaMA-Factory ShareGPT multimodal fields:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "images": ["/abs/path/to/image.png"]}
```

## One-node Training

```bash
cd transsion_training/llamafactory_impl
LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory \
DATASET_JSONL=/path/to/llamafactory_sft.jsonl \
OUTPUT_DIR=/path/to/output/qwen3vl-8b-transsion-sft \
bash scripts/train_qwen3vl_transsion.sh
```

## Slurm Training

```bash
cd transsion_training/llamafactory_impl
LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory \
DATASET_JSONL=/path/to/llamafactory_sft.jsonl \
OUTPUT_DIR=/path/to/output/qwen3vl-8b-transsion-sft \
sbatch scripts/train_qwen3vl_transsion.slurm
```

Optional knobs:

```bash
export MODEL_NAME_OR_PATH=Qwen/Qwen3-VL-8B-Instruct
export DATASET_NAME=transsion_math_sft
export NUM_TRAIN_EPOCHS=3
export LEARNING_RATE=1.0e-5
export CUTOFF_LEN=32768
export SAVE_TOTAL_LIMIT=1
export REPORT_TO=none        # set to wandb only after wandb login on the cluster
export NPROC_PER_NODE=8
```

## What the scripts do

1. Register `DATASET_NAME` in `$LLAMA_FACTORY_DIR/data/dataset_info.json` without overwriting unrelated dataset entries.
2. Render a runtime YAML config from `config_templates/qwen3vl_8b_transsion_sft.yaml.in`.
3. Launch LLaMA-Factory full SFT with Qwen3-VL template and DeepSpeed ZeRO-3.

The scripts do not embed tokens. If using W&B, authenticate outside the script with the cluster's standard secret manager or `wandb login`.
