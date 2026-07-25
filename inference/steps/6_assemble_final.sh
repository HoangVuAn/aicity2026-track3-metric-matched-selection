#!/usr/bin/env bash
# Step 6 — final assembly: serve the text judge, then run open-ended MBR-BERTScore + cross-task
# consistency judge + 1-Yes-1-No pair-repair over the step-2..5 pools -> submission_repro.csv.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
kg
echo "########## step6 serve judge[$G1] + assemble_faithful (PAIR_REPAIR=$PAIR_REPAIR) ##########"
CUDA_VISIBLE_DEVICES=$G1 nohup $VLLM serve "$JUDGE_MODEL" --port $PORT_JUDGE \
  --tensor-parallel-size 1 --max-model-len $MML_JUDGE --gpu-memory-utilization $GPU_UTIL > "$R/s_judge.log" 2>&1 &
wp $PORT_JUDGE || exit 1
PR_FLAG=""; [ "$PAIR_REPAIR" = "1" ] && PR_FLAG="--pair-repair"
CUDA_VISIBLE_DEVICES=$G2 $LOC "$INF/assemble_faithful.py" --base $SUB/submission_clean_repro.csv \
  --pool $R --judge-url http://localhost:$PORT_JUDGE/v1 $PR_FLAG --out $SUB/submission_repro.csv
kg
echo "########## step6 DONE -> $SUB/submission_repro.csv ##########"
