"""Heuristic Tier S/A/B classifier per video.

Tier S detection signal (proxy for NVIDIA human-annotation priming):
  - Reasoning contains explicit timestamps (regex)
  - Spatial reference keywords (lane, crosswalk, frame position)
  - Reasoning length above corpus median
  - Low ratio of generic hedge terms

Tier A vs B is decided by the consistency checker downstream; this module
emits an `s_score` per video (continuous) plus a tier label using percentiles.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import median

from .config import TierConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TierScore:
    video_id: str
    s_score: float
    timestamp_hits: int
    spatial_hits: int
    avg_reasoning_len: float
    generic_ratio: float
    tier: str = "B"


def _count_pattern(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _count_keywords(text: str, kws: tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(kw) for kw in kws)


def _generic_ratio(text: str, generic_terms: tuple[str, ...]) -> float:
    words = text.split()
    if not words:
        return 1.0
    lower = text.lower()
    hits = sum(lower.count(g) for g in generic_terms)
    return hits / max(len(words), 1)


def compute_video_signals(
    items_per_task: dict[str, list[dict]],
    tier_cfg: TierConfig,
) -> tuple[int, int, float, float]:
    """Aggregate signals across all items belonging to one video."""
    pattern = re.compile(tier_cfg.s_heuristic.timestamp_pattern)
    kws = tier_cfg.s_heuristic.spatial_keywords
    generics = tier_cfg.s_heuristic.generic_terms

    ts_total = 0
    sp_total = 0
    lengths: list[int] = []
    gen_ratios: list[float] = []

    for items in items_per_task.values():
        for it in items:
            r = it.get("reasoning", "") or ""
            ts_total += _count_pattern(r, pattern)
            sp_total += _count_keywords(r, kws)
            lengths.append(len(r))
            gen_ratios.append(_generic_ratio(r, generics))

    avg_len = sum(lengths) / len(lengths) if lengths else 0.0
    avg_gen = sum(gen_ratios) / len(gen_ratios) if gen_ratios else 1.0
    return ts_total, sp_total, avg_len, avg_gen


def score_video(
    timestamp_hits: int,
    spatial_hits: int,
    avg_reasoning_len: float,
    generic_ratio: float,
    length_median: float,
) -> float:
    """Linear combination of normalized signals → unbounded score, higher = more S-like."""
    ts = min(timestamp_hits / 6.0, 2.0)
    sp = min(spatial_hits / 4.0, 2.0)
    length_bonus = 1.0 if avg_reasoning_len >= length_median else 0.0
    generic_penalty = generic_ratio * 5.0
    return ts * 1.2 + sp * 1.0 + length_bonus * 0.6 - generic_penalty


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (pct / 100.0) * (len(s) - 1)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def classify_tiers(
    by_video: dict[str, dict[str, list[dict]]],
    tier_cfg: TierConfig,
) -> list[TierScore]:
    """Return TierScore per video, with `.tier` ∈ {S, A, B} via percentile cutoffs."""
    raw_signals: list[tuple[str, int, int, float, float]] = []
    for vid, items_per_task in by_video.items():
        ts, sp, alen, gen = compute_video_signals(items_per_task, tier_cfg)
        raw_signals.append((vid, ts, sp, alen, gen))

    length_med = median(s[3] for s in raw_signals) if raw_signals else 0.0
    scored: list[TierScore] = []
    for vid, ts, sp, alen, gen in raw_signals:
        s = score_video(ts, sp, alen, gen, length_med)
        scored.append(
            TierScore(
                video_id=vid,
                s_score=s,
                timestamp_hits=ts,
                spatial_hits=sp,
                avg_reasoning_len=alen,
                generic_ratio=gen,
            )
        )

    scores = [r.s_score for r in scored]
    s_cut = percentile(scores, tier_cfg.s_score_percentile)
    a_cut = percentile(scores, tier_cfg.a_score_percentile)

    for r in scored:
        if r.s_score >= s_cut:
            r.tier = "S"
        elif r.s_score >= a_cut:
            r.tier = "A"
        else:
            r.tier = "B"

    counts = {"S": 0, "A": 0, "B": 0}
    for r in scored:
        counts[r.tier] += 1
    logger.info(
        "Tier classification: S=%d A=%d B=%d (cut_s=%.3f cut_a=%.3f len_median=%.0f)",
        counts["S"], counts["A"], counts["B"], s_cut, a_cut, length_med,
    )
    return scored
