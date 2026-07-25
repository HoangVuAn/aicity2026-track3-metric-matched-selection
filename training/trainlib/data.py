"""Video frame extraction, prompt encoding, and record loading for Cosmos SFT.

torchcodec is broken in cosmos3_env, so we decode N frames with cv2 ourselves and pass a
correct VideoMetadata (real fps + actual frames_indices) so the model still receives accurate
native `<X seconds>` timestamp tokens — equivalent to native video sampling, leak-free.
"""

from __future__ import annotations

import json
import random

import cv2
import numpy as np
from transformers.video_utils import VideoMetadata

from trainlib.frames import N_FRAMES, select_indices, uniform
from trainlib.paths import MERGED_SFT, rel_to_repo

TIER_DUP = {"S": 3, "A": 2, "B": 1}  # tier-weighting via record duplication
# UniTime-style sequence hint for temporal grounding; must match between training data
# (prep_temporal_tiles.py) and any inference on a model trained with it.
TS_HINT = ("This is a video sequence interleaved with timestamps and frames. Identify the "
           "temporal window (start and end timestamps) when the queried event occurs.")
ALL_TASKS = {"bcq", "mcq", "bcq_openended", "mcq_openended", "open_qa", "causal_linkage",
             "scene_description", "temporal_description", "video_summarization",
             "temporal_localization", "anomaly_description"}


# --- frame sampling -------------------------------------------------------------------

def _to_n(idxs: list[int], total: int, n: int) -> list[int]:
    """Resize an index list to exactly n (even) sorted frames: subsample if too many,
    uniform-pad if too few (temporal_patch_size=2 needs an even count)."""
    if n % 2:
        n += 1
    idxs = sorted(set(idxs))
    if len(idxs) > n:
        idxs = sorted({idxs[int(i * len(idxs) / n)] for i in range(n)})
    if len(idxs) < n:
        idxs = sorted(set(idxs + uniform(total, n - len(idxs), exclude=[(j, j) for j in idxs])))
    return sorted(idxs[:n])


def pick_indices(task: str, question: str, answer: str, total: int, fps: float,
                 n: int, sampling: str) -> list[int]:
    """sampling="uniform": evenly spaced (leak-free, test-reproducible).
    sampling="smart": task-aware select_indices (dense around QUESTION timestamps).
    temporal_localization never uses the answer window (leak) -> question/uniform fallback."""
    if sampling == "uniform":
        return [int(i * total / n) for i in range(n if n % 2 == 0 else n + 1)]
    ans = "" if task == "temporal_localization" else answer
    return _to_n(select_indices(task, question, ans, total, fps), total, n)


def _resize_native(fr, max_side: int):
    """Aspect-preserving downscale so the long side <= max_side, rounded to a multiple of 32
    (patch 16 x merge 2); never upscale. Lets Qwen3-VL's NaViT processor keep the TRUE aspect
    ratio (no square distortion), matching the heterogeneous train + 640x360 test natively."""
    h0, w0 = fr.shape[:2]
    s = min(1.0, max_side / max(h0, w0))
    nw = max(32, int(round(w0 * s / 32)) * 32)
    nh = max(32, int(round(h0 * s / 32)) * 32)
    return cv2.resize(fr, (nw, nh))


def extract_frames(rec: dict, n: int, sampling: str, size: int = 640) -> tuple[np.ndarray, VideoMetadata]:
    """Decode the chosen frames (cv2) + VideoMetadata. `size` = long-side cap (aspect-preserved, native)."""
    cap = cv2.VideoCapture(str(rel_to_repo(rec["video_path"])))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idxs = pick_indices(rec["task"], rec["question"], rec.get("answer", ""), total, fps, n, sampling)
    frames, used = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(_resize_native(fr, size), cv2.COLOR_BGR2RGB))
            used.append(i)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded: {rec['video_path']}")
    if len(used) % 2:
        frames, used = frames[:-1], used[:-1]
    meta = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps,
                         frames_indices=used, video_backend="custom")
    return np.stack(frames), meta


def load_frames_dir(rec: dict, n: int, size: int = 448) -> tuple[np.ndarray, VideoMetadata]:
    """Uniformly sample n (even) frames from a DoTA JPG frames dir + correct VideoMetadata."""
    files = sorted(rel_to_repo(rec["frames_dir"]).glob("*.jpg"))
    total = len(files)
    if n % 2:
        n += 1
    frames, used = [], []
    for i in (int(j * total / n) for j in range(n)):
        img = cv2.imread(str(files[i]))
        if img is not None:
            frames.append(cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB))
            used.append(i)
    if len(used) % 2:
        frames, used = frames[:-1], used[:-1]
    fps = rec.get("fps", 10.0)
    meta = VideoMetadata(total_num_frames=total, fps=fps, duration=total / fps,
                         frames_indices=used, video_backend="custom")
    return np.stack(frames), meta


def load_frames(rec: dict, n: int, sampling: str) -> tuple[np.ndarray, VideoMetadata]:
    """Dispatch: DoTA records have 'frames_dir' (JPG) -> uniform; challenge records have
    'video_path' -> SFT smart sampling. Same encoding the SFT model saw."""
    if "frames_dir" in rec:
        return load_frames_dir(rec, n)
    return extract_frames(rec, n, sampling)


# --- prompt encoding ------------------------------------------------------------------

def make_target(rec: dict, use_cot: bool) -> str:
    ans = json.dumps(json.loads(rec["answer"])) if rec["task"] == "temporal_localization" else str(rec["answer"])
    if use_cot and rec.get("reasoning"):
        return f"<think>{rec['reasoning']}</think>\n<answer>{ans}</answer>"
    return ans


def augment_clip(arr: np.ndarray) -> np.ndarray:
    """Appearance-only, clip-consistent augmentation (label-preserving; never touches the time axis).
    Simulates the heterogeneous test distribution (270p-1080p sources, varied CCTV lighting/compression)."""
    import random as _r

    n, h, w, _ = arr.shape
    out = arr
    if _r.random() < 0.5:                                 # resolution jitter: downscale -> upscale
        f = _r.uniform(0.4, 1.0)
        sw, sh = max(8, int(w * f)), max(8, int(h * f))
        out = np.stack([cv2.resize(cv2.resize(fr, (sw, sh)), (w, h), interpolation=cv2.INTER_LINEAR)
                        for fr in out])
    if _r.random() < 0.5:                                 # JPEG compression
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), _r.randint(25, 90)]
        out = np.stack([cv2.cvtColor(cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), enc)[1],
                                                  cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for fr in out])
    if _r.random() < 0.5:                                 # photometric: brightness/contrast/saturation
        br, co, sa = _r.uniform(0.75, 1.25), _r.uniform(0.8, 1.2), _r.uniform(0.7, 1.3)
        f32 = out.astype(np.float32) * br
        mean = f32.mean(axis=(1, 2), keepdims=True)
        f32 = (f32 - mean) * co + mean
        gray = f32.mean(axis=3, keepdims=True)
        f32 = gray + (f32 - gray) * sa
        out = np.clip(f32, 0, 255).astype(np.uint8)
    if _r.random() < 0.3:                                 # mild blur OR gaussian noise
        if _r.random() < 0.5:
            k = _r.choice([3, 5, 7])
            out = np.stack([cv2.GaussianBlur(fr, (k, k), 0) for fr in out])
        else:
            noise = np.random.normal(0, _r.uniform(3, 12), out.shape)
            out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def build_batch(proc, rec: dict, n_frames: int, sampling: str, use_cot: bool, device,
                augment: bool = False) -> tuple[dict, int]:
    """SFT batch: full (user + assistant target) sequence + prompt length for label masking."""
    arr, meta = extract_frames(rec, n_frames, sampling)
    if augment:
        arr = augment_clip(arr)
    user = {"role": "user", "content": [{"type": "video", "video": arr},
                                        {"type": "text", "text": rec["question"]}]}
    plen = proc.apply_chat_template([user], tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt",
                                    video_metadata=[meta])["input_ids"].shape[1]
    full = proc.apply_chat_template(
        [user, {"role": "assistant", "content": make_target(rec, use_cot)}],
        tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt",
        video_metadata=[meta])
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full.items()}
    return batch, plen


def encode(proc, arr, meta, question, device):
    """Build the user turn (video + question) with the generation prompt appended."""
    msgs = [{"role": "user", "content": [{"type": "video", "video": arr},
                                         {"type": "text", "text": question}]}]
    return proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                    return_dict=True, return_tensors="pt",
                                    video_metadata=[meta]).to(device)


# --- timestamp parsing (temporal tasks) -----------------------------------------------

def _to_sec(s: str) -> float:
    p = str(s).split(":")
    if len(p) == 2:           # MM:SS or MM:SS.cc
        return int(p[0]) * 60 + float(p[1])
    if len(p) == 3:           # 3-part is AMBIGUOUS (see to_sec_dur); HH:MM:SS is the more common
        # reading (verified vs durations 2026-06-12: video 5.3s, end "00:00:04" = 4s).
        return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    return float(s)


def to_sec_dur(s: str, dur: float) -> float:
    """Duration-aware timestamp parse. The challenge 3-part GTs MIX two formats:
    HH:MM:SS(.ss) (e.g. "00:00:04" = 4s on a 5.3s clip) and MM:SS:cc (e.g. "00:35:20" = 35.2s
    on a 36s clip). Disambiguate by which reading fits inside the video duration."""
    p = str(s).split(":")
    if len(p) != 3 or dur <= 0:
        return _to_sec(s)
    hh = int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    cc = int(p[0]) * 60 + int(p[1]) + float(p[2]) / 100
    if hh <= dur * 1.05:
        return hh
    if cc <= dur * 1.05:
        return cc
    return hh


# --- record loading -------------------------------------------------------------------


def load_sft_records(tasks: set[str], tiers: set[str], limit: int | None,
                     weighted: bool = True) -> list[dict]:
    """SFT corpus from merged_sft.jsonl, tier-weighted by record duplication (S 3x / A 2x / B 1x)."""
    recs: list[dict] = []
    for line in MERGED_SFT.open():
        r = json.loads(line)
        if r["task"] in tasks and r.get("tier") in tiers and rel_to_repo(r["video_path"]).exists():
            recs.extend([r] * TIER_DUP.get(r.get("tier"), 1) if weighted else [r])
    random.shuffle(recs)
    return recs[:limit] if limit else recs


