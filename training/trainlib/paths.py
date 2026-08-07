"""Canonical paths. Import-light (no torch/cv2) so any module can use REPO/SNAP.

SNAP (the local HF snapshot dir for nvidia/Cosmos3-Super) is overridable via the
COSMOS_SNAP env var for portability when the organizers re-run on another machine."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]            # training/trainlib/ -> repo root
TRAIN = REPO / "training"
DATA = TRAIN / "data"
LOGS = TRAIN / "logs"

# Cosmos3-Super reasoner snapshot (33.5B): your HF-cache snapshot dir of nvidia/Cosmos3-Super
# (e.g. $HF_HOME/hub/models--nvidia--Cosmos3-Super/snapshots/<hash>). Set via COSMOS_SNAP.
SNAP = os.getenv("COSMOS_SNAP", "")


def require_snap() -> str:
    """Checked accessor for Cosmos consumers; qwen training never needs it."""
    if not SNAP:
        raise RuntimeError(
            "Set COSMOS_SNAP to your local nvidia/Cosmos3-Super snapshot directory "
            "(download once via `huggingface-cli download nvidia/Cosmos3-Super`)."
        )
    return SNAP

# Generated datasets
MERGED_SFT = DATA / "merged_sft.jsonl"

# Challenge / official-test data (shared inputs, not Cosmos-specific).
# TEST_JSON / VIDEO_DIR overridable via env (defaults unchanged) so the same
# faithful gen/assemble scripts can be pointed at the validation split.
TEST_JSON = Path(os.getenv("TEST_JSON", str(REPO / "dataset/official_test/test/test.json")))
TEST_VIDEO_BASE = Path(os.getenv("VIDEO_DIR", str(REPO / "dataset/official_test/videos")))
TRAIN_VIDEO_BASE = Path(os.getenv("TRAIN_VIDEO_DIR", str(REPO / "dataset/train/videos")))
VAL_JSONL = REPO / "dataset/processed/val.jsonl"
VAL_SAMPLE_IDS = REPO / "dataset/processed/val_eval/sample_ids.json"


def rel_to_repo(p: str | Path) -> Path:
    """Resolve a repo-relative path (records store paths relative to the repo root)."""
    p = Path(p)
    return p if p.is_absolute() else REPO / p
