"""Task-aware frame index selection (vendored from scripts/extract_keyframes.py so the
Cosmos package is self-contained).

Strategy per task: dense sampling around the timestamps in the QUESTION (temporal_description,
causal_linkage, mcq) plus uniform fill; uniform everywhere else. temporal_localization never
uses the answer window here (leak-free, single-pass reproducible) — the caller passes answer="".
"""

from __future__ import annotations

import json
import re

import numpy as np

N_FRAMES = 32
DENSE_FRAMES_PRIMARY = 16
DENSE_FRAMES_CAUSAL = 8
BUFFER_SEC = 1.0


def parse_time(s: str) -> float:
    """MM:SS.cs (dot) or MM:SS:cs (colon centiseconds) or seconds -> seconds."""
    s = s.strip()
    parts = s.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 60 + float(parts[1]) + float(parts[2]) / 100
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(s)


def parse_timestamps_from_question(text: str) -> list[tuple[float, float]]:
    matches = re.findall(r"\b(\d{1,2}:\d{2}(?:[:.]\d+)?)\b", text)
    times = [parse_time(m) for m in matches]
    pairs = [(times[i], times[i + 1]) for i in range(0, len(times) - 1, 2)]
    if len(times) % 2 == 1:
        pairs.append((times[-1], times[-1]))
    return pairs


def parse_answer_window(answer: str) -> tuple[float, float] | None:
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


def uniform(total: int, n: int, exclude: list[tuple[int, int]] | None = None) -> list[int]:
    pool = ([i for i in range(total) if not any(s <= i <= e for s, e in exclude)]
            if exclude else list(range(total)))
    if not pool:
        return []
    count = min(n, len(pool))
    idx = np.linspace(0, len(pool) - 1, count).astype(int)
    return sorted(set(pool[i] for i in idx))


def select_indices(task: str, question: str, answer: str, total: int, fps: float) -> list[int]:
    """Task-aware indices (N_FRAMES total). Note: no cv2.VideoCapture arg (the original took
    one only for an unused optical-flow path) — pure index math from total/fps + timestamps."""
    if task == "temporal_localization":
        window = parse_answer_window(answer)
        if window:
            sf = frame_at(max(0.0, window[0] - BUFFER_SEC), fps, total)
            ef = frame_at(window[1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, 12, total)
            return sorted(set(d + uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])))

    elif task == "temporal_description":
        pairs = parse_timestamps_from_question(question)
        if pairs:
            sf = frame_at(max(0.0, pairs[0][0] - BUFFER_SEC), fps, total)
            ef = frame_at(pairs[0][1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, DENSE_FRAMES_PRIMARY, total)
            return sorted(set(d + uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])))

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
            return sorted(set(d + uniform(total, N_FRAMES - len(d), exclude=exclude)))

    elif task in ("mcq", "mcq_openended"):
        pairs = parse_timestamps_from_question(question)
        if pairs:
            sf = frame_at(max(0.0, pairs[0][0] - BUFFER_SEC), fps, total)
            ef = frame_at(pairs[0][1] + BUFFER_SEC, fps, total)
            d = dense(sf, ef, DENSE_FRAMES_PRIMARY, total)
            return sorted(set(d + uniform(total, N_FRAMES - len(d), exclude=[(sf, ef)])))

    return uniform(total, N_FRAMES)
