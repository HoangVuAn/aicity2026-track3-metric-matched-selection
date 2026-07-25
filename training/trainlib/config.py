"""Minimal YAML config support for the training/eval scripts. A `--config path.yaml`
sets argparse defaults (CLI flags still override), so configs/ holds the canonical,
reproducible hyperparameters while keeping the scripts CLI-driven."""

from __future__ import annotations

import argparse
from pathlib import Path


def apply_config_defaults(parser: argparse.ArgumentParser, argv=None) -> argparse.ArgumentParser:
    """If --config FILE is present in argv, load it as YAML and set it as argparse defaults
    (so explicit CLI flags win). Returns the same parser for chaining before parse_args()."""
    import yaml

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        cfg = yaml.safe_load(Path(known.config).read_text()) or {}
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in cfg.items()})
    return parser
