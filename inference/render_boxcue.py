"""Render one boxed mp4 per TEST video that has a bcq item.

Draws cached YOLO boxes (from detector_pass.py -> YOLO_CACHE_DIR) onto 64 uniform frames
per test video and writes a boxed .mp4 that gen_boxcue_votetta.py reads. Resume-safe:
boxed_test_mp4 skips videos whose mp4 already exists.
"""
import os
import sys
import json
import random
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TEST_JSON, TEST_VIDEO_BASE  # noqa: E402
import box_cue as bc  # noqa: E402

MP4 = os.getenv("BOXCUE_MP4_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/boxcue_mp4"))
os.makedirs(MP4, exist_ok=True)


def boxed_test_mp4(video_id, n=64):
    out = f"{MP4}/{video_id.replace('/', '__')}.mp4"
    if os.path.exists(out):
        return out
    cap = cv2.VideoCapture(str((TEST_VIDEO_BASE / video_id).resolve()))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = list(range(total)) if total <= n else [int(round(x)) for x in np.linspace(0, total - 1, n)]
    frames, used = [], []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        s = min(1.0, 640 / max(h, w))
        nw, nh = max(32, int(round(w * s / 32)) * 32), max(32, int(round(h * s / 32)) * 32)
        frames.append(cv2.cvtColor(cv2.resize(fr, (nw, nh)), cv2.COLOR_BGR2RGB))
        used.append(fi)
    cap.release()
    if len(used) % 2:
        frames, used = frames[:-1], used[:-1]
    arr = np.stack(frames)
    arr, _ = bc.draw_box_cue(arr, used, f"dataset/train/videos/{video_id}", "correct", random.Random(0))
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (arr.shape[2], arr.shape[1]))
    for fr in arr:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    items = json.load(open(TEST_JSON))["items"]
    vids = []
    seen = set()
    for it in items:
        if it["task_type"] != "bcq":
            continue
        vid = str(it["video_id"])
        if vid in seen:
            continue
        seen.add(vid)
        vids.append(vid)
    print(f"rendering {len(vids)} bcq videos -> {MP4}", flush=True)
    done = 0
    for vid in vids:
        boxed_test_mp4(vid, a.n)
        done += 1
        if done % 20 == 0:
            print(f"{done}/{len(vids)}", flush=True)
    print(f"DONE: {done}/{len(vids)} -> {MP4}", flush=True)


if __name__ == "__main__":
    main()
