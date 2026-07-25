#!/usr/bin/env bash
# Prepare box-cue inputs: (1) YOLO26x detection cache, (2) boxed test mp4s.
# MUST run BEFORE the box-cue serve/gen step in run_full_fast.sh (gen_boxcue_votetta.py
# reads the boxed mp4s produced here).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"
cd "$REPO_DIR"

# 1) Build the YOLO box cache on GPU if it is not already present.
#    Requires yolo26x.pt (provide via YOLO_WEIGHTS; defaults to the api_backend path).
if ! ls "$YOLO_CACHE_DIR"/*.json >/dev/null 2>&1; then
  echo "=== building YOLO box cache -> $YOLO_CACHE_DIR ==="
  CUDA_VISIBLE_DEVICES=${BOXCUE_GPU} $PY "$HERE/detector_pass.py"
else
  echo "=== YOLO box cache present in $YOLO_CACHE_DIR, skipping detector ==="
fi

# 2) Render boxed test mp4s (CPU).
echo "=== rendering boxed mp4s -> $BOXCUE_MP4_DIR ==="
$PY "$HERE/render_boxcue.py"

echo "=== BOXCUE STAGE DONE ==="
