"""Shared helpers for the AI City Track-3 inference reproduction pipeline.

Self-contained: answer cleaning, MBR-BERTScore selection, timestamp/option parsing, cross-task
grounding-prompt builders, and structural guards. No dependency on the research-time trainlib package.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional

_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_TS = re.compile(r"(\d{1,2}):(\d{2})[.:](\d{2})")
_OPT = re.compile(r"(?:^|\n)\s*([A-E])[\.\)]\s*(.+?)(?=\n\s*[A-E][\.\)]|\Z)", re.DOTALL)
_FMT = re.compile(r"Answer with|Provide the result|Provide the")

OPEN_TASKS = ("open_qa", "causal_linkage", "scene_description",
              "temporal_description", "video_summarization")
CHOICE_TASKS = ("bcq", "mcq", "mcq_openended")
# tasks whose question text is used as a cross-task context source for other tasks (present per video)
CONTEXT_TASKS = ("temporal_localization", "open_qa", "causal_linkage", "scene_description",
                 "video_summarization", "temporal_description", "mcq", "mcq_openended")


def clean(s: Optional[str]) -> str:
    """Strip <think>…</think> and unwrap <answer>…</answer>."""
    s = s or ""
    m = _ANSWER.search(s)
    if m:
        s = m.group(1)
    return _THINK.sub("", s).strip()


def strip_format(q: str) -> str:
    """Drop the trailing answer-format instruction from a question."""
    return _FMT.split(q)[0].strip()


def first_line(q: str) -> str:
    return (q or "").split("\n")[0].strip()


def yn(s: Optional[str]) -> Optional[str]:
    s = (s or "").strip().lower()
    m = re.match(r"^(yes|no)\b", s) or re.search(r"\b(yes|no)\b", s)
    return m.group(1) if m else None


def gt_polarity(a: str) -> str:
    return "yes" if (a or "").strip().lower().startswith("yes") else "no"


def option_text(question: str, letter: str) -> str:
    """Map an MCQ letter (A-E) to its option text within the question."""
    letter = (letter or "").strip().upper()[:1]
    for lt, tx in _OPT.findall(question):
        if lt == letter:
            return re.sub(r"\s+", " ", tx).strip()[:220]
    return ""


def window_seconds(q: str):
    """Return (start, end) seconds parsed from a temporal question, or None."""
    m = _TS.findall(q)
    if len(m) < 2:
        return None
    t = [int(a) * 60 + int(b) + int(c) / 100 for a, b, c in m[:2]]
    return (t[0], t[1]) if t[1] > t[0] else None


# ---------------------------------------------------------------------------
# MBR-BERTScore selection (lazy-loads the scorer once)
# ---------------------------------------------------------------------------
_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        import bert_score
        _SCORER = bert_score.BERTScorer(lang="en", rescale_with_baseline=True)
    return _SCORER


def mbr(candidates, cap: int = 48) -> str:
    """Minimum-Bayes-Risk pick: candidate maximizing mean pairwise BERTScore-F1.

    Dedups (preserving order) and caps the pool for speed (same argmax as full pool).
    """
    cs = [c for c in candidates if c]
    cs = list(dict.fromkeys(cs))[:cap]
    if len(cs) < 2:
        return cs[0] if cs else ""
    sc = _scorer()
    n = len(cs)
    pc, pr = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                pc.append(cs[i]); pr.append(cs[j])
    f = [float(x) for x in sc.score(pc, pr)[2]]
    util = [0.0] * n
    k = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                util[i] += f[k]; k += 1
    return cs[max(range(n), key=lambda i: util[i])]


def vote(votes, margin_too: bool = False):
    """Majority vote over a list of discrete labels. Returns label (and margin if requested)."""
    vs = [v for v in votes if v]
    if not vs:
        return (None, 0.0) if margin_too else None
    c = Counter(vs)
    top, cnt = c.most_common(1)[0]
    if margin_too:
        return top, (2 * cnt - len(vs)) / len(vs)  # 0 = even split, 1 = unanimous
    return top


# ---------------------------------------------------------------------------
# Cross-task grounding prompt builders (the cross-task grounding levers)
# ---------------------------------------------------------------------------
def evidence_prefix(loc_q: str, mcq_opt: str, entity_pair) -> str:
    """Shared evidence pack used by open_qa / summarization grounding."""
    parts = []
    if loc_q:
        parts.append(f"the video shows: {loc_q}")
    if mcq_opt:
        parts.append(f"what happened: {mcq_opt}")
    if entity_pair:
        parts.append(f"the vehicles are a {entity_pair[0]} and a {entity_pair[1]}")
    return ("Context: " + "; ".join(parts) + ".") if parts else ""


def prompt_temporal_locevent(question: str, loc_q: str) -> str:
    return (f"Note: this segment shows: {loc_q}. Describe what happened, consistent with this event.\n\n"
            f"{question}")


def prompt_open_qa(question: str, loc_q: str, mcq_opt: str, entity_pair) -> str:
    ev = evidence_prefix(loc_q, mcq_opt, entity_pair)
    return f"{ev} Answer the question consistent with this context.\n\n{question}"


def prompt_causal(question: str, mcq_opt: str) -> str:
    cause = f"Context: the root cause is: {mcq_opt}." if mcq_opt else ""
    return (f"{cause} Keep your answer focused on the cause-and-effect relationship, "
            f"consistent with this.\n\n{question}")


def prompt_summ_fullevent(question: str, loc_q: str, causal_q: str, mcq_opt: str, entity_pair) -> str:
    fe = f"Note: the video shows: {loc_q}."
    if causal_q:
        fe += f" A related question concerns: {causal_q}."
    if mcq_opt:
        fe += f" What happened: {mcq_opt}."
    if entity_pair:
        fe += f" The vehicles are a {entity_pair[0]} and a {entity_pair[1]}."
    return (fe + " Summarize the full sequence of events including the incident and its consequences, "
            f"consistent with this.\n\n{question}")


def prompt_bcqoe_no(question: str, loc_q: str, mcq_opt: str, entity_pair) -> str:
    ev = evidence_prefix(loc_q, mcq_opt, entity_pair)
    return (f"{ev} The correct answer is No: the queried collision does NOT occur. Answer in the style: "
            f"'No. The actual collision involves ... , not ...' using the true vehicles from the context.\n\n"
            f"{question}")
