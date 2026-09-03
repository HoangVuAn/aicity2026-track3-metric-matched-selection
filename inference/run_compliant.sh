#!/usr/bin/env bash
# Exact commands used for the compliant rerun requested in the Track 3 award review (2026-09-02).
# Only compliant models: cosmos_v2 (SFT on `tasks: all`) and the zero-shot nvidia/Cosmos3-Super.
# No f1, no box-cue, no artifact either of them produced; no pair repair; no consistency judge.
#
#   G1=1 G2=2 bash inference/run_compliant.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source inference/config.sh

RC="inference/full_run_compliant"; mkdir -p "$RC"
SUB_BASE="dataset/official_test/submission_compliant.csv"
G1="${G1:-1}"; G2="${G2:-2}"

# ── 1. choice assembly from the cached cosmos pools (no grounding yet) ────────────────
CUDA_VISIBLE_DEVICES=$G1 $PY inference/assemble_compliant.py \
  --dir inference/full_run --out "$SUB_BASE"

# ── 2. regenerate every grounding cue with cosmos_v2 ─────────────────────────────────
CUDA_VISIBLE_DEVICES=$G1 nohup $VLLM serve "$COSMOS_V2_MODEL" --port $PORT_CV2 \
  --tensor-parallel-size 1 --allowed-local-media-path / --max-model-len $MML_COSMOS \
  --gpu-memory-utilization 0.92 --hf-overrides "$COSMOS_OVERRIDES" \
  --media-io-kwargs "{\"video\":{\"num_frames\":$FRAMES_COSMOS}}" > "$RC/s_cv2.log" 2>&1 &
until curl -s "http://localhost:$PORT_CV2/v1/models" >/dev/null 2>&1; do sleep 15; done
U=http://localhost:$PORT_CV2/v1
# NOTE: --sub is the COMPLIANT base from step 1, never the f1/box-cue assembly.
$PY inference/gen_locevent_temporal_test.py --base-url $U --n $NG --out $RC/locevent_temporal_test.json
$PY inference/gen_summ_fullevent_test.py    --base-url $U --n $NG --out $RC/summ_fullevent_test.json
$PY inference/gen_summco_test.py            --base-url $U --n $NG --sub "$SUB_BASE" --out $RC/summco_test.json
$PY inference/gen_evidence_test.py          --base-url $U --n $NG --sub "$SUB_BASE" --out $RC/evidence_test.json
pkill -f "[v]llm serve.*$PORT_CV2" 2>/dev/null; sleep 5

# ── 3. final compliant assembly (the reported submission) ────────────────────────────
CUDA_VISIBLE_DEVICES=$G1 $PY inference/assemble_compliant.py \
  --dir inference/full_run --grounding-dir "$RC" \
  --out dataset/official_test/submission_compliant_grounded.csv
echo "DONE -> dataset/official_test/submission_compliant_grounded.csv"
