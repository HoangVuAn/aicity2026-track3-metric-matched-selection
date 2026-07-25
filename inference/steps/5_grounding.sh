#!/usr/bin/env bash
# Step 5 — grounding gens: re-serve f1 (TP2) and generate the task-matched grounding pools
# (locevent for temporal, full-event + mcq-option for summary, evidence for open_qa/causal).
# summco/evidence read submission_clean_repro.csv from step 4. Consumed by step 6.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
kg
echo "########## step5 re-serve f1 TP2[$G1,$G2] + grounding gens ##########"
CUDA_VISIBLE_DEVICES=$G1,$G2 nohup $VLLM serve "$F1_MODEL" --port $PORT_F1 \
  --tensor-parallel-size 2 --allowed-local-media-path / --max-model-len $MML_F1 --gpu-memory-utilization $GPU_UTIL \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_F1}}" > "$R/s_f1b.log" 2>&1 &
wp $PORT_F1 || exit 1
UF=http://localhost:$PORT_F1/v1
$PY "$INF/gen_locevent_temporal_test.py" --base-url $UF --n $NG --out $R/locevent_temporal_test.json
$PY "$INF/gen_summ_fullevent_test.py"    --base-url $UF --n $NG --out $R/summ_fullevent_test.json
$PY "$INF/gen_summco_test.py"            --base-url $UF --n $NG --sub $SUB/submission_clean_repro.csv --out $R/summco_test.json
$PY "$INF/gen_evidence_test.py"          --base-url $UF --n $NG --sub $SUB/submission_clean_repro.csv --out $R/evidence_test.json
kg
echo "########## step5 DONE ##########"
