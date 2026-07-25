"""Answer parsing / normalization for inference + scoring (vendored from
scripts/infer_test.py so the Cosmos package is self-contained). Pure text helpers,
no torch — usable by the vLLM inference, vote, and eval scripts."""

from __future__ import annotations

import json
import re

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_answer(text: str) -> str:
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_yesno(text: str) -> str | None:
    s = text.strip().lower()
    m = re.match(r"^(yes|no)\b", s) or re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def extract_letter(text: str) -> str | None:
    s = text.strip()
    m = re.match(r"^\(?([A-Za-z])\)?[).\s,:]", s)
    if m:
        return m.group(1).upper()
    if re.fullmatch(r"[A-Da-d]", s):
        return s.upper()
    m = re.search(r"\b([A-D])\b", s)
    return m.group(1) if m else None


def extract_temporal_json(text: str, clip_dur: float) -> str:
    """Return a clean {"start","end"} JSON string; fall back to whole-clip span."""
    s = text.strip()
    obj = None
    m = re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL)
    candidate = m.group(1) if m else s
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, list) and parsed:
            parsed = parsed[0]
        if isinstance(parsed, dict) and "start" in parsed and "end" in parsed:
            obj = {"start": str(parsed["start"]), "end": str(parsed["end"])}
    except (json.JSONDecodeError, ValueError, TypeError):
        obj = None
    if obj is None:
        ts = re.findall(r"\d{1,2}:\d{2}(?:\.\d+)?", s)
        if len(ts) >= 2:
            obj = {"start": ts[0], "end": ts[1]}
    if obj is None:
        m2, sec = divmod(clip_dur, 60)
        obj = {"start": "00:00", "end": f"{int(m2):02d}:{sec:05.2f}"}
    return json.dumps(obj)


def clip_duration(video_path) -> float:
    import cv2
    c = cv2.VideoCapture(str(video_path))
    n = c.get(cv2.CAP_PROP_FRAME_COUNT)
    f = c.get(cv2.CAP_PROP_FPS) or 30.0
    c.release()
    return n / f if f else 0.0
