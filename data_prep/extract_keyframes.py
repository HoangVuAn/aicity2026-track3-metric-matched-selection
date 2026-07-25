"""Extract task-aware keyframes from training/val videos.

Per-task strategy:
  temporal_localization : 12 dense around answer window (±1s) + 20 uniform
  temporal_description  : 16 dense around question window (±1s) + 16 uniform
  causal_linkage        : dense around timestamp pairs in question + uniform fill
  mcq / mcq_openended   : 16 dense around question timestamp if present + 16 uniform
  all others            : 32 uniform

Outputs:
  dataset/processed/frames/<slug>/frame_NNNNN.jpg  (32 frames per unique record)
  dataset/processed/train_stage1_images.jsonl       (88,536 tier-weighted records)
  dataset/processed/val_images.jsonl                (2,196 records)

Resume-safe: skips records whose frame dir already has 32 JPEGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_BASE = REPO_ROOT / "dataset/train/videos"
VAD_RL_REMAP = REPO_ROOT / "dataset/external/vad_r1/Vad-Reasoning-RL"
VAD_SFT_REMAP = REPO_ROOT / "dataset/external/vad_r1/Vad-Reasoning-SFT"
FRAMES_DIR = REPO_ROOT / "dataset/processed/frames"

SRC_TRAIN = REPO_ROOT / "dataset/processed/train.jsonl"
SRC_VAL = REPO_ROOT / "dataset/processed/val.jsonl"
DST_TRAIN = REPO_ROOT / "dataset/processed/train_stage1_images.jsonl"
DST_VAL = REPO_ROOT / "dataset/processed/val_images.jsonl"

N_FRAMES = 32
DENSE_FRAMES_PRIMARY = 16
DENSE_FRAMES_CAUSAL = 8
JPEG_QUALITY = 60
BUFFER_SEC = 1.0
FLOW_CANDIDATES = 128
TIER_REPEATS = {"S": 3, "A": 2, "B": 1}

logger = logging.getLogger(__name__)


def resolve_video_path(video_id: str) -> Optional[Path]:
    p = VIDEO_BASE / video_id
    if p.exists():
        return p
    if video_id.startswith("Vad-R1/Vad-Reasoning-RL/"):
        rel = Path(video_id).relative_to("Vad-R1/Vad-Reasoning-RL")
        avi = (VAD_RL_REMAP / rel).with_suffix(".avi")
        if avi.exists():
            return avi
    if video_id.startswith("Vad-R1/Vad-Reasoning-SFT/"):
        rel = Path(video_id).relative_to("Vad-R1/Vad-Reasoning-SFT")
        avi = (VAD_SFT_REMAP / rel).with_suffix(".avi")
        if avi.exists():
            return avi
    return None


def parse_time(s: str) -> float:
    """Parse MM:SS.cs (dot) or MM:SS:cs (colon) into seconds."""
    s = s.strip()
    parts_colon = s.split(":")
    if len(parts_colon) == 3:
        # MM:SS:cs format — third part is centiseconds
        return float(parts_colon[0]) * 60 + float(parts_colon[1]) + float(parts_colon[2]) / 100
    if len(parts_colon) == 2:
        # MM:SS or MM:SS.cs (dot decimal)
        return float(parts_colon[0]) * 60 + float(parts_colon[1])
    return float(s)


def parse_timestamps_from_question(text: str) -> list[tuple[float, float]]:
    # Match both MM:SS.cs and MM:SS:cs patterns
    pattern = r"\b(\d{1,2}:\d{2}(?:[:.]\d+)?)\b"
    matches = re.findall(pattern, text)
    times = [parse_time(m) for m in matches]
    pairs: list[tuple[float, float]] = []
    for i in range(0, len(times) - 1, 2):
        pairs.append((times[i], times[i + 1]))
    if len(times) % 2 == 1:
        pairs.append((times[-1], times[-1]))
    return pairs


def parse_answer_window(answer: str) -> Optional[tuple[float, float]]:
    try:
        ans = json.loads(answer)
        if "start" in ans and "end" in ans:
            return parse_time(str(ans["start"])), parse_time(str(ans["end"]))
    except (json.JSONDecodeError, KeyError, ValueError):
        pass
    return None


def frame_at(t: float, fps: float, total: int) -> int:
    return max(0, min(total - 1, round(t * fps)))


def dense(start_f: int, end_f: int, n: int, total: int) -> list[int]:
    start_f, end_f = max(0, start_f), min(total - 1, end_f)
    if start_f > end_f:
        end_f = start_f
    count = min(n, end_f - start_f + 1)
    return sorted(set(np.linspace(start_f, end_f, count).astype(int).tolist()))


def uniform(total: int, n: int, exclude: list[tuple[int, int]] = None) -> list[int]:
    pool = (
        [i for i in range(total) if not any(s <= i <= e for s, e in exclude)]
        if exclude
        else list(range(total))
    )
    if not pool:
        return []
    count = min(n, len(pool))
    idx = np.linspace(0, len(pool) - 1, count).astype(int)
    return sorted(set(pool[i] for i in idx))


def optical_flow_top(cap: cv2.VideoCapture, total: int, n: int) -> list[int]:
    n_cand = min(FLOW_CANDIDATES, total)
    candidates = np.linspace(0, total - 1, n_cand).astype(int).tolist()
    scores: dict[int, float] = {}
    prev_gray: Optional[np.ndarray] = None

    for idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            scores[idx] = 0.0
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            scores[idx] = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
        else:
            scores[idx] = 0.0
        prev_gray = gray

    top = sorted(scores, key=lambda x: scores[x], reverse=True)[:n]
    return sorted(top)


def select_indices(
    task: str, question: str, answer: str,
    cap: cv2.VideoCapture, total: int, fps: float,
) -> list[int]:
    n_flow = N_FRAMES // 2
    n_uniform = N_FRAMES - n_flow

    if task == "temporal_localization":
        window = parse_answer_window(answer)
        if window:
            sf = frame_at(max(0.0, window[0] - BUFFER_SEC), fps, total)
            ef = frame_at(window[1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, 12, total)
            u = uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])
            return sorted(set(d + u))

    elif task == "temporal_description":
        pairs = parse_timestamps_from_question(question)
        if pairs:
            sf = frame_at(max(0.0, pairs[0][0] - BUFFER_SEC), fps, total)
            ef = frame_at(pairs[0][1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, DENSE_FRAMES_PRIMARY, total)
            u = uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])
            return sorted(set(d + u))

    elif task == "causal_linkage":
        pairs = parse_timestamps_from_question(question)
        if pairs:
            d: list[int] = []
            exclude: list[tuple[int, int]] = []
            for t0, t1 in pairs[:2]:
                sf = frame_at(max(0.0, t0 - BUFFER_SEC), fps, total)
                ef = frame_at(t1 + BUFFER_SEC, fps, total)
                d += dense(sf, ef, DENSE_FRAMES_CAUSAL, total)
                exclude.append((sf, ef))
            u = uniform(total, N_FRAMES - len(d), exclude=exclude)
            return sorted(set(d + u))

    elif task in ("mcq", "mcq_openended"):
        pairs = parse_timestamps_from_question(question)
        if pairs:
            sf = frame_at(max(0.0, pairs[0][0] - BUFFER_SEC), fps, total)
            ef = frame_at(pairs[0][1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, DENSE_FRAMES_PRIMARY, total)
            u = uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])
            return sorted(set(d + u))

    # Fallback: uniform
    return uniform(total, N_FRAMES)


def make_slug(video_id: str, question: str) -> str:
    h = hashlib.md5((video_id + question).encode()).hexdigest()[:8]
    safe = video_id.replace("/", "__").replace(".mp4", "").replace(".avi", "")
    return f"{safe}__{h}"


def extract_frames(
    video_path: Path, indices: list[int], out_dir: Path
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths: list[str] = []
    for rank, idx in enumerate(indices):
        out_path = out_dir / f"frame_{rank:05d}.jpg"
        if not out_path.exists():
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.save(str(out_path), quality=JPEG_QUALITY)
        paths.append(str(out_path))
    cap.release()
    return paths


def sec_to_ts(t: float) -> str:
    """Convert seconds to MM:SS.cs format matching dataset timestamps."""
    m = int(t) // 60
    s = t - m * 60
    return f"{m:02d}:{s:05.2f}"


def build_human_text(indices: list[int], fps: float, question: str) -> str:
    total_sec = indices[-1] / fps if indices else 0.0
    header = f"Video duration: {sec_to_ts(total_sec)}. {len(indices)} frames sampled:\n"
    lines = [f"[t={sec_to_ts(idx / fps)}] <image>" for idx in indices]
    return header + "\n".join(lines) + "\n" + question


def make_target(reasoning: str, answer: str) -> str:
    return f"<think>{reasoning}</think>\n<answer>{answer}</answer>"


def process_record(rec: dict) -> Optional[dict]:
    """Return LLaMA-Factory entry or None if video missing."""
    video_path = resolve_video_path(rec["video_id"])
    if video_path is None:
        return None

    slug = make_slug(rec["video_id"], rec["question"])
    out_dir = FRAMES_DIR / slug

    # Check for cached indices file (preserves fps for timestamp labels)
    indices_file = out_dir / "indices.json"
    existing = sorted(out_dir.glob("frame_*.jpg")) if out_dir.exists() else []

    if len(existing) == N_FRAMES and indices_file.exists():
        meta = json.loads(indices_file.read_text())
        indices, fps = meta["indices"], meta["fps"]
        image_paths = [str(p) for p in existing]
    else:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total == 0:
            cap.release()
            return None

        indices = select_indices(rec["task"], rec["question"], rec["answer"], cap, total, fps)
        if len(indices) < N_FRAMES:
            extra = uniform(total, N_FRAMES - len(indices), exclude=[(i, i) for i in indices])
            indices = sorted(set(indices + extra))
        indices = indices[:N_FRAMES]
        cap.release()

        image_paths = extract_frames(video_path, indices, out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        indices_file.write_text(json.dumps({"indices": indices, "fps": fps}))

    return {
        "conversations": [
            {"from": "human", "value": build_human_text(indices, fps, rec["question"])},
            {"from": "gpt", "value": make_target(rec["reasoning"], rec["answer"])},
        ],
        "images": image_paths,
        "tier": rec.get("tier", "B"),
        "task": rec["task"],
        "init_weight": rec.get("init_weight", 1.0),
    }


def _worker(rec: dict) -> Optional[dict]:
    try:
        return process_record(rec)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed %s: %s", rec.get("video_id"), e)
        return None


def process_split(
    records: list[dict],
    dst: Path,
    n_workers: int,
    apply_tier: bool,
    log_interval: int = 1000,
) -> None:
    total = len(records)
    skipped = 0
    done = 0

    with dst.open("w") as f, multiprocessing.Pool(n_workers) as pool:
        for entry in pool.imap_unordered(_worker, records, chunksize=4):
            done += 1
            if entry is None:
                skipped += 1
            else:
                repeats = TIER_REPEATS.get(entry.get("tier", "B"), 1) if apply_tier else 1
                for _ in range(repeats):
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if done % log_interval == 0:
                logger.info("%d/%d done, %d skipped", done, total, skipped)

    logger.info("Done: %d/%d → %s (skipped %d)", total - skipped, total, dst, skipped)


def run(limit: Optional[int] = None, n_workers: int = 32) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    train_records = [json.loads(l) for l in SRC_TRAIN.open() if l.strip()]
    if limit:
        train_records = train_records[:limit]

    logger.info("Train: %d records, %d workers", len(train_records), n_workers)
    process_split(train_records, DST_TRAIN, n_workers, apply_tier=True)

    val_records = [json.loads(l) for l in SRC_VAL.open() if l.strip()]
    logger.info("Val: %d records, %d workers", len(val_records), n_workers)
    process_split(val_records, DST_VAL, n_workers, apply_tier=False, log_interval=200)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    run(limit=args.limit, n_workers=args.workers)
