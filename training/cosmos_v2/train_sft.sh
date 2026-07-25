#!/bin/bash
# Multi-GPU SFT for Cosmos3-Super (accelerate launches N processes; FSDP/DeepSpeed
# configured in-code). Pass --deepspeed for ZeRO-3 (the v2 recipe) or --fsdp.
#   GPUS=0,3,4,5,6,7 bash training/cosmos_v2/train_sft.sh --config training/cosmos_v2/configs/sft.yaml --deepspeed
set -euo pipefail
export HF_HOME=/mnt/data/anhv10/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPUS:-0,3,4,5,6,7}"
NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
cd "$(dirname "$0")/../.."
/mnt/data/anhv10/cosmos3_env/bin/python -m accelerate.commands.launch \
  --num_processes "$NPROC" --num_machines 1 --mixed_precision bf16 \
  training/cosmos_v2/train_sft.py --fsdp "$@"
