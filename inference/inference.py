from __future__ import annotations

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
