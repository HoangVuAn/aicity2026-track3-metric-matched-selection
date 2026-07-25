"""Combine task JSONs + tier + consistency into a single training JSONL.

Output schema (one line per item):
{
    "id": "<video_id>::<task>::<idx>",   # idx >0 only for bcq/bcq_openended (2/video)
    "video_id": "...",
    "task": "...",
    "question": "...",
    "answer": "...",
    "reasoning": "...",                  # flat reasoning string
    "tier": "S" | "A" | "B",
    "consistency_score": 0.0..1.0,
    "init_weight": float,                # tier_weight * consistency_downweight
    "split": "train" | "val"
}

Validation split is stratified per task, with all items of a held-out video
moved together to avoid leakage.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from .config import DataPrepConfig
from .consistency import ConsistencyReport
from .io import write_jsonl
from .tier_classifier import TierScore

logger = logging.getLogger(__name__)


def _stratified_val_videos(
    by_video: dict[str, dict[str, list[dict]]],
    val_ratio: float,
    seed: int,
) -> set[str]:
    rng = random.Random(seed)
    videos = sorted(by_video.keys())
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio))
    return set(videos[:n_val])


def build_records(
    by_video: dict[str, dict[str, list[dict]]],
    tier_scores: dict[str, TierScore],
    consistency_reports: dict[str, ConsistencyReport],
    cfg: DataPrepConfig,
) -> list[dict]:
    val_set = _stratified_val_videos(
        by_video, cfg.dataset_build.val_ratio, cfg.dataset_build.random_seed
    )
    tier_w = cfg.tier.weights
    records: list[dict] = []
    for vid, items_per_task in by_video.items():
        tier_label = tier_scores[vid].tier
        cons = consistency_reports[vid]
        init_w = tier_w[tier_label] * cons.downweight
        split = "val" if vid in val_set else "train"
        for task, items in items_per_task.items():
            for idx, it in enumerate(items):
                records.append({
                    "id": f"{vid}::{task}::{idx}",
                    "video_id": vid,
                    "task": task,
                    "question": it.get("question", ""),
                    "answer": it.get("answer", ""),
                    "reasoning": it.get("reasoning", ""),
                    "tier": tier_label,
                    "consistency_score": round(cons.mean_score, 4),
                    "init_weight": round(init_w, 4),
                    "split": split,
                })
    return records


def write_dataset(records: list[dict], out_dir: Path) -> tuple[Path, Path]:
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)
    return train_path, val_path


def summarize(records: list[dict]) -> dict[str, int | float]:
    tiers = {"S": 0, "A": 0, "B": 0}
    tasks: dict[str, int] = {}
    splits = {"train": 0, "val": 0}
    weight_sum = 0.0
    for r in records:
        tiers[r["tier"]] += 1
        tasks[r["task"]] = tasks.get(r["task"], 0) + 1
        splits[r["split"]] += 1
        weight_sum += r["init_weight"]
    return {
        "n_items": len(records),
        "by_tier": tiers,
        "by_task": tasks,
        "by_split": splits,
        "mean_weight": round(weight_sum / max(len(records), 1), 3),
    }
