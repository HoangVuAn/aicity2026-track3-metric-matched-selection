"""Build the full training corpus from the official challenge data (end-to-end, deterministic).

Stage 1  10 task JSONs (dataset/train/train/*.json)
           -> tier classification (S/A/B) + cross-task consistency
           -> dataset/processed/train.jsonl + val.jsonl (+ tier/consistency/summary reports)
Stage 2  train.jsonl + optional Vad-R1 external SFT set
           -> training/data/merged_sft.jsonl   (the file every training script consumes;
              unified schema: video_path, task, question, answer, reasoning, tier, source)

Usage:
    python3 data_prep/build_corpus.py                # full pipeline
    python3 data_prep/build_corpus.py --skip-vad     # challenge data only (no external set)

Vad-R1 (external, publicly declared): download Vad-Reasoning-SFT from
the official Vad-R1 release (https://github.com/wbfwonderful/Vad-R1) and place it under
dataset/external/vad_r1/ (see README). If absent, stage 2 warns and proceeds
with the challenge records only.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data_prep"))

from prep.config import load_config
from prep.consistency import consistency_for_all
from prep.dataset_builder import build_records, summarize, write_dataset
from prep.io import group_by_video, load_all_tasks
from prep.tier_classifier import classify_tiers

VAD_SFT = REPO_ROOT / "dataset/external/vad_r1/Vad-Reasoning-SFT-train.jsonl"
MERGED_OUT = REPO_ROOT / "training/data/merged_sft.jsonl"
THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANS = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

log = logging.getLogger("build_corpus")


def stage1(config_path: Path | None) -> Path:
    cfg = load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading raw tasks from %s", cfg.raw_train_dir)
    tasks = load_all_tasks(cfg.task_json_paths)
    by_video = group_by_video(tasks)
    log.info("Loaded %d unique videos across %d tasks", len(by_video), len(tasks))

    tier_scores = classify_tiers(by_video, cfg.tier)
    consistency_reports = consistency_for_all(by_video, cfg.consistency)
    tier_by_vid = {t.video_id: t for t in tier_scores}
    cons_by_vid = {r.video_id: r for r in consistency_reports}

    records = build_records(by_video, tier_by_vid, cons_by_vid, cfg)
    train_path, val_path = write_dataset(records, cfg.output_dir)

    (cfg.output_dir / "tier_report.json").write_text(
        json.dumps([asdict(t) for t in tier_scores], indent=2)
    )
    (cfg.output_dir / "consistency_report.json").write_text(
        json.dumps(
            [
                {
                    "video_id": r.video_id,
                    "mean_score": round(r.mean_score, 4),
                    "downweight": r.downweight,
                    "pair_scores": {f"{a}|{b}": s for (a, b), s in r.pair_scores.items()},
                }
                for r in consistency_reports
            ],
            indent=2,
        )
    )
    (cfg.output_dir / "summary.json").write_text(json.dumps(summarize(records), indent=2))
    log.info("Stage 1 done: %s / %s", train_path, val_path)
    return train_path


def mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def challenge_records(train_jsonl: Path):
    for line in train_jsonl.open():
        r = json.loads(line)
        yield {
            "video_path": f"dataset/train/videos/{r['video_id']}",
            "task": r["task"],
            "question": r["question"],
            "answer": r["answer"],
            "reasoning": r.get("reasoning", ""),
            "tier": r.get("tier", "B"),
            "source": "challenge",
        }


def vad_records():
    import cv2  # only needed for the optional external set

    for line in VAD_SFT.open():
        r = json.loads(line)
        p = r["path"]
        if not (REPO_ROOT / p).exists() and not Path(p).exists():
            continue
        vp = p if Path(p).exists() else str(REPO_ROOT / p)
        tm, am = THINK.search(r.get("think", "")), ANS.search(r.get("answer", ""))
        reasoning = tm.group(1).strip() if tm else ""
        answer = am.group(1).strip() if am else str(r.get("answer", ""))
        cap = cv2.VideoCapture(vp)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if r.get("start") is not None and r.get("end") is not None and fps:
            yield {
                "video_path": vp,
                "task": "temporal_localization",
                "question": "When does the anomalous event occur in the video? "
                            "Provide the start and end time in mm:ss.",
                "answer": json.dumps({"start": mmss(r["start"] / fps), "end": mmss(r["end"] / fps)}),
                "reasoning": reasoning,
                "tier": "A",
                "source": "vad",
            }
        yield {
            "video_path": vp,
            "task": "anomaly_description",
            "question": "Describe the anomalous event in this video, including what happens "
                        "and why it is abnormal.",
            "answer": answer,
            "reasoning": reasoning,
            "tier": "A",
            "source": "vad",
        }


def stage2(train_jsonl: Path, skip_vad: bool) -> None:
    MERGED_OUT.parent.mkdir(parents=True, exist_ok=True)
    gens = [lambda: challenge_records(train_jsonl)]
    if skip_vad:
        log.info("Stage 2: --skip-vad set, merging challenge records only")
    elif not VAD_SFT.exists():
        log.warning(
            "Stage 2: %s not found — merging challenge records only. "
            "To include the external Vad-R1 set, follow the download step in the README.",
            VAD_SFT,
        )
    else:
        gens.append(vad_records)

    src, tsk = Counter(), Counter()
    with MERGED_OUT.open("w") as f:
        for gen in gens:
            for rec in gen():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                src[rec["source"]] += 1
                tsk[rec["task"]] += 1
    log.info("Stage 2 done: %s (%d records)", MERGED_OUT, sum(src.values()))
    log.info("by source: %s", dict(src))
    log.info("by task: %s", dict(tsk))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to data_prep.yaml")
    parser.add_argument("--skip-vad", action="store_true",
                        help="Skip the external Vad-R1 set even if present")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_jsonl = stage1(args.config)
    stage2(train_jsonl, args.skip_vad)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
