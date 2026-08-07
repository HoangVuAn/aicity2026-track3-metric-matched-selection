#!/usr/bin/env bash
# CANONICAL orchestrator: full-flow regeneration (5-model ensemble) then faithful assembly -> ONE submission.
# Parallel model rounds on 3 GPUs (TP1), reduced N, resume-safe. ~5-7h. All paths/models/ports/N in config.sh.
# Models: cosmos-base, cosmos_v2, box-cue, f1 (128f), judge.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cd "$REPO_DIR"
mkdir -p "$R"
OVR="$COSMOS_OVERRIDES"

wp(){ for t in $(seq 1 180); do [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$1/v1/models)" = "200" ] && return 0; sleep 10; done; echo "PORT $1 FAIL"; return 1; }
kg(){ for row in $(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader|awk -F', ' '$3+0>1000{print $1","$2}'); do
  p=$(echo $row|cut -d, -f1); u=$(echo $row|cut -d, -f2); idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader|grep "$u"|cut -d, -f1|tr -d ' ');
  [ "$(ps -o user= -p $p 2>/dev/null|tr -d ' ')" = "$USER" ] && echo "$idx"|grep -qE "^($G1|$G2|$G3)$" && kill -9 $p 2>/dev/null; done; sleep 8; }

# ── box-cue prep: render boxed mp4s (YOLO detect -> draw) if absent. Requires YOLO_WEIGHTS (external 118MB).
#    Heavy on first run (detector pass); resume-safe. Skipped once boxed mp4s exist. ──
if [ -z "$(ls "$BOXCUE_MP4_DIR"/*.mp4 2>/dev/null)" ]; then
  echo "########## box-cue prep: detector + render (BOXCUE_GPU=$BOXCUE_GPU) ##########"
  bash "$HERE/stage_boxcue.sh" || { echo "box-cue prep FAILED (check YOLO_WEIGHTS)"; exit 1; }
fi

echo "########## ROUND 1 (parallel TP1): cosmos-base[$G1] cosmos_v2[$G2] box-cue[$G3] ##########"
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
( $PY "$HERE/gen_cosmos_votetta.py" --base-url http://localhost:$PORT_CB/v1 --n $NV --out $R/cosmos_votetta.json
  $PY "$HERE/gen_orphan_votes.py" --base-url http://localhost:$PORT_CB/v1 --kind raw --tag cosmos --n $NV --out $R/orphan_votes_cosmos.json ) > $R/g_cb.log 2>&1 &
P1=$!
( $PY "$HERE/gen_cosmos_votetta.py" --base-url http://localhost:$PORT_CV2/v1 --n $NV --out $R/cosmos_v2_votetta.json
  $PY "$HERE/gen_orphan_votes.py" --base-url http://localhost:$PORT_CV2/v1 --kind raw --tag cosmos_v2 --n $NV --out $R/orphan_votes_cosmos_v2.json ) > $R/g_cv2.log 2>&1 &
P2=$!
( $PY "$HERE/gen_boxcue_votetta.py" --base-url http://localhost:$PORT_BX/v1 --n $NV --out $R/boxcue_votetta.json
  $PY "$HERE/gen_orphan_votes.py" --base-url http://localhost:$PORT_BX/v1 --kind boxcue --tag boxcue --n $NV --out $R/orphan_votes_boxcue.json ) > $R/g_bx.log 2>&1 &
P3=$!
wait $P1 $P2 $P3   # wait ONLY for gen sub-shells, NOT the vllm serves (bare `wait` hangs on serves)

echo "########## ROUND 1.5: RICHPOOL in PARALLEL -- f1 TP2[$G1,$G2] || cosmos_v2 TP1[$G3] ##########"
kg
CUDA_VISIBLE_DEVICES=$G1,$G2 nohup $VLLM serve "$F1_MODEL" --port $PORT_F1 \
  --tensor-parallel-size 2 --allowed-local-media-path / --max-model-len $MML_F1 --gpu-memory-utilization $GPU_UTIL \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_F1}}" > "$R/s_f1.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$G3 nohup $VLLM serve "$COSMOS_V2_MODEL" --port $PORT_CV2 --tensor-parallel-size 1 \
  --allowed-local-media-path / --max-model-len $MML_COSMOS --gpu-memory-utilization 0.92 --hf-overrides "$OVR" \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_COSMOS}}" > "$R/s_cv2b.log" 2>&1 &
wp $PORT_F1 && wp $PORT_CV2 || exit 1
UF=http://localhost:$PORT_F1/v1
# richpool: temp-0.7 only. prune_sweep showed dropping temp-0.9 leaves the scene MBR pick 100% unchanged
# (40/40) -> temp-0.9 is redundant; dropping it halves the (heaviest) f1 richpool generation, 0 score loss.
( $PY "$HERE/gen_f1_oe.py"    --base-url $UF --n $NV --out $R/gen_f1_oe.json
  $PY "$HERE/mbr_test_gen.py" --base-url $UF --n $NR --temp 0.7 --out $R/mbr_test.json
  $PY "$HERE/gen_orphan_votes.py" --base-url $UF --kind raw --tag f1 --n $NV --out $R/orphan_votes_f1.json ) > $R/g_f1.log 2>&1 &
Pf=$!
( $PY "$HERE/mbr_test_gen.py" --base-url http://localhost:$PORT_CV2/v1 --n $NR --temp 0.7 --out $R/mbr_test_cosmos.json ) > $R/g_cv2rp.log 2>&1 &
Pc=$!
wait $Pf $Pc   # gen sub-shells only, not serves

echo "########## base choice+desc assembly (kill f1 first -> free cards for bert_score) ##########"
kg
CUDA_VISIBLE_DEVICES=$G1 $LOC "$HERE/assemble_clean.py" --dir $R --out $SUB/submission_clean_repro.csv

echo "########## grounding gens (re-serve f1 TP2) ##########"
CUDA_VISIBLE_DEVICES=$G1,$G2 nohup $VLLM serve "$F1_MODEL" --port $PORT_F1 \
  --tensor-parallel-size 2 --allowed-local-media-path / --max-model-len $MML_F1 --gpu-memory-utilization $GPU_UTIL \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_F1}}" > "$R/s_f1b.log" 2>&1 &
wp $PORT_F1 || exit 1
$PY "$HERE/gen_locevent_temporal_test.py" --base-url $UF --n $NG --out $R/locevent_temporal_test.json
$PY "$HERE/gen_summ_fullevent_test.py"    --base-url $UF --n $NG --out $R/summ_fullevent_test.json
$PY "$HERE/gen_summco_test.py"            --base-url $UF --n $NG --sub $SUB/submission_clean_repro.csv --out $R/summco_test.json
$PY "$HERE/gen_evidence_test.py"          --base-url $UF --n $NG --sub $SUB/submission_clean_repro.csv --out $R/evidence_test.json
kg

echo "########## judge + faithful final (PAIR_REPAIR=$PAIR_REPAIR) ##########"
CUDA_VISIBLE_DEVICES=$G1 nohup $VLLM serve "$JUDGE_MODEL" --port $PORT_JUDGE \
  --tensor-parallel-size 1 --max-model-len $MML_JUDGE --gpu-memory-utilization $GPU_UTIL > "$R/s_judge.log" 2>&1 &
wp $PORT_JUDGE || exit 1
PR_FLAG=""; [ "$PAIR_REPAIR" = "1" ] && PR_FLAG="--pair-repair"
CUDA_VISIBLE_DEVICES=$G2 $LOC "$HERE/assemble_faithful.py" --base $SUB/submission_clean_repro.csv \
  --pool $R --judge-url http://localhost:$PORT_JUDGE/v1 $PR_FLAG --out $SUB/submission_repro.csv
kg
echo "########## DONE -> $SUB/submission_repro.csv ##########"
