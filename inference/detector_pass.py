"""B2 detector pass: YOLO26x boxes on 64 uniform frames per video (train+val+test).

Caches normalized [0,1] xyxy boxes per (video, frame_idx) so the box-overlay step is
resolution-independent and aligns with `--sampling uniform --n-frames 64` training.
Resume-safe (skips videos already cached). Shard across GPUs with --shard/--nshards.
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import TRAIN_VIDEO_BASE, TEST_VIDEO_BASE, TEST_JSON  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolo26x.pt")  # ultralytics auto-downloads by name
CACHE = os.getenv("YOLO_CACHE_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_run/yolo_boxes"))
CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]  # person bicycle car motorcycle bus truck traffic-light stop-sign
MAX_BOXES = 20        # sanity cap: drop frames that emit an absurd count
DARK_MEAN = 12.0      # brightness gate: near-black frame -> no detection


def video_list():
    vids = {}
    for split in ("dataset/processed/train.jsonl", "dataset/processed/val.jsonl"):
        for line in open(split):
            vid = json.loads(line)["video_id"]
            vids.setdefault(vid, TRAIN_VIDEO_BASE)
    for it in json.load(open(TEST_JSON))["items"]:
        vids.setdefault(str(it["video_id"]), TEST_VIDEO_BASE)
    return sorted(vids.items())


def uniform_idx(total, n):
    if total <= n:
        return list(range(total))
    return [int(round(x)) for x in np.linspace(0, total - 1, n)]


def slug(vid):
    return vid.replace("/", "__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--res", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    model = YOLO(WEIGHTS)
    vids = video_list()
    mine = [vids[i] for i in range(len(vids)) if i % a.nshards == a.shard]
    print(f"shard {a.shard}/{a.nshards}: {len(mine)} videos", flush=True)

    done = 0
    for vid, base in mine:
        out = f"{CACHE}/{slug(vid)}.json"
        if os.path.exists(out):
            done += 1
            continue
        vp = str((base / vid).resolve())
        cap = cv2.VideoCapture(vp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        rec = {"video_id": vid, "n": a.n_frames, "frames": []}
        if total <= 0:
            cap.release()
            json.dump(rec, open(out, "w"))
            continue
        idxs = uniform_idx(total, a.n_frames)
        frames, keep = [], []
        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok:
                continue
            h, w = fr.shape[:2]
            sc = a.res / max(h, w)
            fr = cv2.resize(fr, (int(w * sc), int(h * sc)))
            frames.append(fr)
            keep.append(fi)
        cap.release()
        for b0 in range(0, len(frames), a.batch):
            chunk = frames[b0:b0 + a.batch]
            res = model.predict(chunk, conf=a.conf, classes=CLASSES, verbose=False)
            for j, r in enumerate(res):
                fi = keep[b0 + j]
                fr = chunk[j]
                H, W = fr.shape[:2]
                if float(fr.mean()) < DARK_MEAN:
                    rec["frames"].append({"idx": fi, "boxes": []})
                    continue
                boxes = []
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    boxes.append([int(b.cls[0]), round(x1 / W, 4), round(y1 / H, 4),
                                  round(x2 / W, 4), round(y2 / H, 4), round(float(b.conf[0]), 3)])
                if len(boxes) > MAX_BOXES:
                    boxes = []
                rec["frames"].append({"idx": fi, "boxes": boxes})
        json.dump(rec, open(out, "w"))
        done += 1
        if done % 100 == 0:
            print(f"shard {a.shard}: {done}/{len(mine)}", flush=True)
    print(f"shard {a.shard} DONE: {done}/{len(mine)}", flush=True)


main()
