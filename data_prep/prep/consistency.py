"""Cross-task consistency check per video.

Each pair returns 1.0 (consistent), 0.0 (conflict), or 0.5 (undecidable).
We use rule-based comparators — no LLM:
  - bcq vs causal_linkage: bcq answer (Yes/No) should match presence of
    anomaly in causal_linkage answer.
  - temporal_localization vs temporal_description: TL window should be
    referenced (within tolerance) in TD answer text.
  - scene_description vs video_summarization: token overlap above floor.
  - mcq vs causal_linkage: shared keyword presence between selected option
    text and causal answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .config import ConsistencyConfig

logger = logging.getLogger(__name__)

UNDECIDABLE = 0.5
TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2})(?:\.\d+)?")
ANOMALY_NEG = (
    "no anomaly",
    "no accident",
    "no collision",
    "no incident",
    "normal traffic",
    "uneventful",
    "no unusual",
)


@dataclass(slots=True)
class ConsistencyReport:
    video_id: str
    pair_scores: dict[tuple[str, str], float]
    mean_score: float
    downweight: float = 1.0


def _yesno(text: str) -> str | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return None


def _answer_implies_anomaly(text: str) -> bool | None:
    low = text.lower()
    if any(p in low for p in ANOMALY_NEG):
        return False
    anomaly_terms = ("collide", "crash", "accident", "violat", "hits", "loses control")
    if any(t in low for t in anomaly_terms):
        return True
    return None


def check_bcq_causal(bcq_items: list[dict], causal_items: list[dict]) -> float:
    if not bcq_items or not causal_items:
        return UNDECIDABLE
    yes_count = sum(1 for it in bcq_items if _yesno(it.get("answer", "")) == "yes")
    no_count = sum(1 for it in bcq_items if _yesno(it.get("answer", "")) == "no")
    if yes_count + no_count == 0:
        return UNDECIDABLE
    bcq_anomaly = yes_count >= no_count
    causal_signal = _answer_implies_anomaly(causal_items[0].get("answer", ""))
    if causal_signal is None:
        return UNDECIDABLE
    return 1.0 if bcq_anomaly == causal_signal else 0.0


def _parse_timestamp(s: str) -> float | None:
    m = TIMESTAMP_RE.search(s)
    if not m:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def check_temporal_pair(tl_items: list[dict], td_items: list[dict], tol_seconds: int = 10) -> float:
    if not tl_items or not td_items:
        return UNDECIDABLE
    tl_answer = tl_items[0].get("answer", "")
    start_match = re.search(r'"start"\s*:\s*"([^"]+)"', tl_answer)
    if not start_match:
        return UNDECIDABLE
    tl_start = _parse_timestamp(start_match.group(1))
    if tl_start is None:
        return UNDECIDABLE
    td_text = td_items[0].get("answer", "") + " " + td_items[0].get("reasoning", "")
    found = TIMESTAMP_RE.findall(td_text)
    if not found:
        return UNDECIDABLE
    for mm, ss in found:
        sec = int(mm) * 60 + int(ss)
        if abs(sec - tl_start) <= tol_seconds:
            return 1.0
    return 0.0


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", text.lower())}


def check_text_overlap(items_a: list[dict], items_b: list[dict], floor: float = 0.15) -> float:
    if not items_a or not items_b:
        return UNDECIDABLE
    a = _tokenize(items_a[0].get("answer", ""))
    b = _tokenize(items_b[0].get("answer", ""))
    if not a or not b:
        return UNDECIDABLE
    overlap = len(a & b) / max(len(a | b), 1)
    return 1.0 if overlap >= floor else 0.0


def check_mcq_causal(mcq_items: list[dict], causal_items: list[dict]) -> float:
    if not mcq_items or not causal_items:
        return UNDECIDABLE
    q = mcq_items[0].get("question", "")
    ans_letter = mcq_items[0].get("answer", "").strip()
    if not ans_letter or len(ans_letter) > 2:
        return UNDECIDABLE
    option_re = re.compile(rf"(?m)^\s*{re.escape(ans_letter)}[.)]\s*(.+)$")
    m = option_re.search(q)
    if not m:
        return UNDECIDABLE
    option_text = m.group(1)
    return check_text_overlap(
        [{"answer": option_text}],
        [{"answer": causal_items[0].get("answer", "")}],
        floor=0.10,
    )


_CHECKERS = {
    ("bcq", "causal_linkage"): check_bcq_causal,
    ("temporal_localization", "temporal_description"): check_temporal_pair,
    ("scene_description", "video_summarization"): check_text_overlap,
    ("mcq", "causal_linkage"): check_mcq_causal,
}


def consistency_for_video(
    items_per_task: dict[str, list[dict]],
    cfg: ConsistencyConfig,
) -> ConsistencyReport:
    pair_scores: dict[tuple[str, str], float] = {}
    for pair in cfg.pairs:
        checker = _CHECKERS.get(pair) or _CHECKERS.get((pair[1], pair[0]))
        a_items = items_per_task.get(pair[0], [])
        b_items = items_per_task.get(pair[1], [])
        score = checker(a_items, b_items) if checker else UNDECIDABLE
        pair_scores[pair] = score
    if pair_scores:
        mean = sum(pair_scores.values()) / len(pair_scores)
    else:
        mean = UNDECIDABLE
    downweight = cfg.conflict_downweight if mean < cfg.conflict_threshold else 1.0
    vid = next(iter(items_per_task.values()))[0].get("video_id", "") if items_per_task else ""
    return ConsistencyReport(video_id=vid, pair_scores=pair_scores, mean_score=mean, downweight=downweight)


def consistency_for_all(
    by_video: dict[str, dict[str, list[dict]]],
    cfg: ConsistencyConfig,
) -> list[ConsistencyReport]:
    reports = [consistency_for_video(items_per_task, cfg) for items_per_task in by_video.values()]
    flagged = sum(1 for r in reports if r.downweight < 1.0)
    logger.info(
        "Consistency: %d videos checked, %d flagged for downweight (threshold=%.2f)",
        len(reports), flagged, cfg.conflict_threshold,
    )
    return reports
