#!/usr/bin/env bash
# 27B box-cue specialist (bcq + mcq only), 64 frames on YOLO-boxed clips, DeepSpeed ZeRO-3.
# Args verified from the original run (wandb btqkny9f): --box-cue --cot --augment, uniform
# sampling, lr 1e-4, grad-accum 10, 600 steps (save every 200), tiers S,A. The original ran on
# 3 GPUs -> eff batch = micro1 x grad-accum10 x 3gpu = 30; keep ~3 GPUs (or scale grad-accum
# inversely) to hold the same eff batch. Override GPUS for your machine.
# Prerequisite: the YOLO box cache (training/data/yolo_boxes/) must exist; box_cue.py overlays it.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec > training/qwen_sft/logs/qwen36_bcqmcq_boxcue_64.log 2>&1
run() {
  GPUS="${GPUS:-4,5,6}" bash training/qwen_sft/train_ds.sh \
    --model Qwen/Qwen3.6-27B --data training/data/merged_sft.jsonl --tasks bcq,mcq \
    --box-cue --cot --augment --n-frames 64 --sampling uniform --grad-accum 10 --lr 1e-4 \
    --output-dir checkpoints/qwen36_27b_bcqmcq_boxcue $1
}
echo "===== SMOKE @64f box-cue $(date) ====="
if run "--max-steps 2 --save-steps 999999 --smoke"; then
  echo "===== SMOKE OK -> FULL (max 600 step, save 200) $(date) ====="
  run "--max-steps 600 --save-steps 200 --logging-steps 10"
  echo "===== BOXCUE_64 TRAIN DONE $(date) — merge (merge.py) + eval to pick peak ====="
else
  echo "===== SMOKE FAILED $(date) ====="
fi
