#!/usr/bin/env bash
# Step 2 — closed-task votes: serve cosmos-base, cosmos_v2, box-cue (TP1, in parallel) and
# generate each model's vote/TTA pools + orphan votes. Consumed by step 4 (assemble_clean).
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
kg
echo "########## step2 serve cosmos-base[$G1] cosmos_v2[$G2] box-cue[$G3] ##########"
CUDA_VISIBLE_DEVICES=$G1 nohup $VLLM serve "$COSMOS_BASE_MODEL" --port $PORT_CB --tensor-parallel-size 1 \
  --allowed-local-media-path / --max-model-len $MML_COSMOS --gpu-memory-utilization 0.92 --hf-overrides "$OVR" \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_COSMOS}}" > "$R/s_cb.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G2 nohup $VLLM serve "$COSMOS_V2_MODEL" --port $PORT_CV2 --tensor-parallel-size 1 \
  --allowed-local-media-path / --max-model-len $MML_COSMOS --gpu-memory-utilization 0.92 --hf-overrides "$OVR" \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_COSMOS}}" > "$R/s_cv2.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G3 nohup $VLLM serve "$BOXCUE_MODEL" --port $PORT_BX \
  --tensor-parallel-size 1 --allowed-local-media-path / --max-model-len $MML_COSMOS --gpu-memory-utilization 0.92 \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_BOXCUE}}" > "$R/s_bx.log" 2>&1 &
wp $PORT_CB && wp $PORT_CV2 && wp $PORT_BX || exit 1
( $PY "$INF/gen_cosmos_votetta.py" --base-url http://localhost:$PORT_CB/v1 --n $NV --out $R/cosmos_votetta.json
  $PY "$INF/gen_orphan_votes.py" --base-url http://localhost:$PORT_CB/v1 --kind raw --tag cosmos --n $NV --out $R/orphan_votes_cosmos.json ) > $R/g_cb.log 2>&1 &
P1=$!
( $PY "$INF/gen_cosmos_votetta.py" --base-url http://localhost:$PORT_CV2/v1 --n $NV --out $R/cosmos_v2_votetta.json
  $PY "$INF/gen_orphan_votes.py" --base-url http://localhost:$PORT_CV2/v1 --kind raw --tag cosmos_v2 --n $NV --out $R/orphan_votes_cosmos_v2.json ) > $R/g_cv2.log 2>&1 &
P2=$!
( $PY "$INF/gen_boxcue_votetta.py" --base-url http://localhost:$PORT_BX/v1 --n $NV --out $R/boxcue_votetta.json
  $PY "$INF/gen_orphan_votes.py" --base-url http://localhost:$PORT_BX/v1 --kind boxcue --tag boxcue --n $NV --out $R/orphan_votes_boxcue.json ) > $R/g_bx.log 2>&1 &
P3=$!
wait $P1 $P2 $P3   # wait ONLY for the gen sub-shells, not the vLLM serves
kg
echo "########## step2 DONE ##########"
