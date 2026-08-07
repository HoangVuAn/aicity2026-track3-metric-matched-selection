#!/bin/bash
# Qwen3.6-27B LoRA SFT via DeepSpeed ZeRO-3 (real data-parallel + model sharding) over N GPUs.
# ~N x faster than single-process device_map=auto (naive model-parallel).
#   GPUS=0,2,3,4,5,7 bash training/qwen_sft/train_ds.sh --n-frames 32 --max-steps 4000 --lr 1e-4 --cot
# Use >=2 GPUs (single-process DeepSpeed additionally needs mpi4py). For a 1-GPU run, call
# train_sft.py directly without --deepspeed (device_map=auto):
#   CUDA_VISIBLE_DEVICES=0 python training/qwen_sft/train_sft.py --n-frames 32 --max-steps 4000 --lr 1e-4 --cot
set -uo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPUS:-0,2,3,4,5,7}"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
cd "$(dirname "$0")/../.."
echo "[qwen36-ds] DeepSpeed ZeRO-3 on GPUs $CUDA_VISIBLE_DEVICES (N=$NPROC)"
"${PY:-python3}" -m accelerate.commands.launch \
  --num_processes "$NPROC" --num_machines 1 --mixed_precision bf16 \
  training/qwen_sft/train_sft.py --deepspeed "$@"
