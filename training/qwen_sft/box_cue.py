"""Draw cached YOLO actor-boxes onto frames as a soft visual cue (B2-robust).

Boxes are a HINT, made robust via cue-dropout+corruption at the caller:
  off (~30%): no boxes (breaks 'boxed=important' shortcut so the model keeps full-image
              attention and still sees the un-boxed accident vehicle),
  correct (~55%): cache boxes (already naturally noisy: YOLO misses/FPs),
  corrupt (~15%): mislabeled colors (covers the label-error case natural YOLO noise lacks).
Injected as DRAWN boxes (spatially interpretable), not text coords. Cache = training/data/yolo_boxes/
(normalized [0,1] xyxy per uniform frame idx). See [[reference-noisy-bbox-cue]], [[b2-yolo-boxes]].
"""
import os
import json
import glob
import functools

import cv2
import numpy as np

CACHE_DIR = os.getenv("YOLO_CACHE_DIR", "training/data/yolo_boxes")
_BASE = "dataset/train/videos/"
# arr is RGB (extract_frames does BGR2RGB). Colors in RGB.
CLS_COLOR = {0: (0, 255, 0), 1: (0, 255, 0),      # person, bicycle -> green (VRU)
             2: (0, 0, 255), 5: (0, 0, 255), 7: (0, 0, 255),  # car, bus, truck -> blue
             3: (255, 165, 0),                     # motorcycle -> orange
             9: (255, 255, 0), 11: (255, 255, 0)}  # traffic-light, stop-sign -> yellow
_PALETTE = [(0, 255, 0), (0, 0, 255), (255, 165, 0), (255, 255, 0)]

LEGEND = ("Colored boxes mark detected road actors (blue=vehicle, green=person/cyclist, "
          "orange=motorcycle, yellow=traffic light/sign); they are hints and may be "
          "incomplete or wrong.\n")


@functools.lru_cache(maxsize=1)
def _cache():
    d = {}
    for f in glob.glob(f"{CACHE_DIR}/*.json"):
        rec = json.load(open(f))
        d[os.path.basename(f)[:-5]] = {fr["idx"]: fr["boxes"] for fr in rec["frames"]}
    return d


def _stem(video_path):
    return video_path.split(_BASE)[-1].replace("/", "__")


def draw_box_cue(arr, frames_indices, video_path, mode, rng):
    """Draw boxes onto arr (in place). Returns (arr, drew_any: bool).

    arr: (N,H,W,3) uint8 RGB. frames_indices: actual frame idx per arr frame (meta.frames_indices).
    mode: 'off' | 'correct' | 'corrupt'. rng: a random.Random.
    """
    if mode == "off":
        return arr, False
    cache = _cache().get(_stem(video_path))
    if not cache:
        return arr, False
    keys = np.array(sorted(cache))
    if keys.size == 0:
        return arr, False
    n, h, w = arr.shape[0], arr.shape[1], arr.shape[2]
    drew = False
    for i in range(n):
        fi = frames_indices[i] if i < len(frames_indices) else int(keys[min(i, keys.size - 1)])
        nk = int(keys[int(np.argmin(np.abs(keys - fi)))])
        frame = np.ascontiguousarray(arr[i])
        for b in cache[nk]:
            cls, x1, y1, x2, y2 = b[0], b[1], b[2], b[3], b[4]
            col = _PALETTE[rng.randrange(len(_PALETTE))] if mode == "corrupt" else CLS_COLOR.get(cls, (200, 200, 200))
            cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, 2)
            drew = True
        arr[i] = frame
    return arr, drew


def sample_mode(rng, p_off=0.30, p_corrupt=0.15):
    r = rng.random()
    if r < p_off:
        return "off"
    if r > 1.0 - p_corrupt:
        return "corrupt"
    return "correct"
