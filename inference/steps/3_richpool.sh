#!/usr/bin/env bash
# Step 3 — open-ended richpool: serve f1 (TP2) + cosmos_v2 (TP1) and sample the MBR candidate
# pools (f1 open-ended, f1 richpool, cosmos_v2 richpool, orphan votes). Consumed by steps 4-6.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
kg
echo "########## step3 serve f1 TP2[$G1,$G2] + cosmos_v2[$G3] ##########"
CUDA_VISIBLE_DEVICES=$G1,$G2 nohup $VLLM serve "$F1_MODEL" --port $PORT_F1 \
  --tensor-parallel-size 2 --allowed-local-media-path / --max-model-len $MML_F1 --gpu-memory-utilization $GPU_UTIL \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_F1}}" > "$R/s_f1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G3 nohup $VLLM serve "$COSMOS_V2_MODEL" --port $PORT_CV2 --tensor-parallel-size 1 \
  --allowed-local-media-path / --max-model-len $MML_COSMOS --gpu-memory-utilization 0.92 --hf-overrides "$OVR" \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_COSMOS}}" > "$R/s_cv2b.log" 2>&1 &
wp $PORT_F1 && wp $PORT_CV2 || exit 1
UF=http://localhost:$PORT_F1/v1
# richpool: temp-0.7 only (dropping temp-0.9 leaves the MBR pick unchanged -> halves the heaviest gen)
( $PY "$INF/gen_f1_oe.py"    --base-url $UF --n $NV --out $R/gen_f1_oe.json
  $PY "$INF/mbr_test_gen.py" --base-url $UF --n $NR --temp 0.7 --out $R/mbr_test.json
  $PY "$INF/gen_orphan_votes.py" --base-url $UF --kind raw --tag f1 --n $NV --out $R/orphan_votes_f1.json ) > $R/g_f1.log 2>&1 &
Pf=$!
( $PY "$INF/mbr_test_gen.py" --base-url http://localhost:$PORT_CV2/v1 --n $NR --temp 0.7 --out $R/mbr_test_cosmos.json ) > $R/g_cv2rp.log 2>&1 &
Pc=$!
wait $Pf $Pc   # gen sub-shells only, not serves
kg
echo "########## step3 DONE ##########"
