#!/usr/bin/env bash
# Step 1 — box-cue prep: YOLO detect + draw the boxed test mp4s the box-cue model reads.
# Heavy on first run (detector pass); skipped once BOXCUE_MP4_DIR has mp4s. Needs YOLO_WEIGHTS.
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
if [ -z "$(ls "$BOXCUE_MP4_DIR"/*.mp4 2>/dev/null)" ]; then
  echo "########## step1 box-cue prep (BOXCUE_GPU=$BOXCUE_GPU) ##########"
  bash "$INF/stage_boxcue.sh" || { echo "box-cue prep FAILED (check YOLO_WEIGHTS)"; exit 1; }
else
  echo "step1: boxed mp4s already present -> skip"
fi
echo "########## step1 DONE ##########"
