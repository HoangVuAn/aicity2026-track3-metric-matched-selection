from __future__ import annotations
import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]  # inference/ -> repo root
TEST_JSON = Path(os.getenv("TEST_JSON", str(REPO / "dataset/official_test/test/test.json")))
TEST_VIDEO_BASE = Path(os.getenv("VIDEO_DIR", str(REPO / "dataset/official_test/videos")))
TRAIN_VIDEO_BASE = Path(os.getenv("TRAIN_VIDEO_DIR", str(REPO / "dataset/train/videos")))
VAL_JSONL = REPO / "dataset/processed/val.jsonl"
