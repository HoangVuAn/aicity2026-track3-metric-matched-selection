"""IO helpers for raw task JSONs and processed JSONL."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


def load_task_json(path: Path) -> tuple[str, list[dict]]:
    """Return (task_name, items) from a task JSON file."""
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return path.stem, payload
    task = payload.get("metadata", {}).get("task") or path.stem
    items = payload.get("items", [])
    return task, items


def load_all_tasks(task_paths: dict[str, Path]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for task, p in task_paths.items():
        if not p.exists():
            logger.warning("Missing task file: %s", p)
            continue
        name, items = load_task_json(p)
        if name != task:
            logger.warning("Task name mismatch in %s: meta=%s expected=%s", p, name, task)
        out[task] = items
        logger.info("Loaded %d items for task=%s", len(items), task)
    return out


def group_by_video(tasks: dict[str, list[dict]]) -> dict[str, dict[str, list[dict]]]:
    """{video_id: {task: [items]}}. Most tasks have 1 item/video; bcq variants have 2."""
    by_vid: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for task, items in tasks.items():
        for it in items:
            vid = it.get("video_id")
            if vid is None:
                continue
            by_vid[vid][task].append(it)
    return by_vid


def write_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    logger.info("Wrote %d records to %s", n, path)
    return n


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
