"""Config loader for data prep pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "data_prep" / "config" / "data_prep.yaml"


@dataclass(slots=True, frozen=True)
class TierHeuristic:
    timestamp_pattern: str
    spatial_keywords: tuple[str, ...]
    generic_terms: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class TierConfig:
    s_score_percentile: float
    a_score_percentile: float
    weights: dict[str, float]
    s_heuristic: TierHeuristic


@dataclass(slots=True, frozen=True)
class ConsistencyConfig:
    pairs: tuple[tuple[str, str], ...]
    conflict_threshold: float
    conflict_downweight: float


@dataclass(slots=True, frozen=True)
class DatasetBuildConfig:
    val_ratio: float
    random_seed: int


@dataclass(slots=True, frozen=True)
class DataPrepConfig:
    raw_train_dir: Path
    output_dir: Path
    cache_dir: Path
    tasks: tuple[str, ...]
    tier: TierConfig
    consistency: ConsistencyConfig
    dataset_build: DatasetBuildConfig

    @property
    def task_json_paths(self) -> dict[str, Path]:
        return {t: self.raw_train_dir / f"{t}.json" for t in self.tasks}


def load_config(path: Path | str | None = None) -> DataPrepConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text())

    paths = raw["paths"]
    base = REPO_ROOT
    tier_raw = raw["tier"]
    s_heur = tier_raw["s_heuristic"]
    cons = raw["consistency"]
    build = raw["dataset_build"]

    return DataPrepConfig(
        raw_train_dir=base / paths["raw_train_dir"],
        output_dir=base / paths["output_dir"],
        cache_dir=base / paths["cache_dir"],
        tasks=tuple(raw["tasks"]),
        tier=TierConfig(
            s_score_percentile=float(tier_raw["s_score_percentile"]),
            a_score_percentile=float(tier_raw["a_score_percentile"]),
            weights={k: float(v) for k, v in tier_raw["weights"].items()},
            s_heuristic=TierHeuristic(
                timestamp_pattern=s_heur["timestamp_pattern"],
                spatial_keywords=tuple(s_heur["spatial_keywords"]),
                generic_terms=tuple(s_heur["generic_terms"]),
            ),
        ),
        consistency=ConsistencyConfig(
            pairs=tuple((p[0], p[1]) for p in cons["pairs"]),
            conflict_threshold=float(cons["conflict_threshold"]),
            conflict_downweight=float(cons["conflict_downweight"]),
        ),
        dataset_build=DatasetBuildConfig(
            val_ratio=float(build["val_ratio"]),
            random_seed=int(build["random_seed"]),
        ),
    )
