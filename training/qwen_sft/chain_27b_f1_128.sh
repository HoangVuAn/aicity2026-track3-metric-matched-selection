#!/usr/bin/env bash
# 27B F1 specialist, 128 frames, 6-GPU DeepSpeed ZeRO-3 (cards 0-5, leave 6,7 free), from scratch,
# ~3 epochs (pick peak by eval). eff batch = micro1 x grad-accum8 x 6gpu = 48. 3 epochs over ~46.3K
# weighted F1 corpus ~= 2895 steps. Clean-resume enabled (state_latest); to resume: re-run with
# --resume-from checkpoints/qwen36_27b_f1_128/state_latest. Eval/infer/route done manually after.
set -uo pipefail
cd "$(dirname "$0")/../.."
exec > training/qwen_sft/logs/qwen36_f1_128.log 2>&1
TASKS=bcq_openended,mcq_openended,open_qa,causal_linkage,scene_description,video_summarization,temporal_description
run() {
  GPUS=0,1,2,3,4,5 bash training/qwen_sft/train_ds.sh \
    --model Qwen/Qwen3.6-27B --data training/data/merged_sft.jsonl --tasks "$TASKS" \
    --cot --augment --n-frames 128 --sampling uniform --grad-accum 8 --lr 1e-4 \
    --output-dir checkpoints/qwen36_27b_f1_128 $1
}
echo "===== SMOKE @128f 6-GPU $(date) ====="
if run "--max-steps 2 --save-steps 999999 --smoke"; then
  echo "===== SMOKE OK -> FULL (max 3655 step =3ep tron, save 400) $(date) ====="
  run "--max-steps 3655 --save-steps 400 --logging-steps 10"
  echo "===== F1_128 TRAIN DONE $(date) — eval/infer manually to pick peak ====="
else
  echo "===== SMOKE FAILED $(date) ====="
fi
