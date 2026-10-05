"""Loads the pinned track config (configs/hpc_cpu.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .model.qwen38_cpu import ModelSpec

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "hpc_cpu.yaml"
REQUIRED_SECTIONS = ("track", "model", "runtime", "limits", "accuracy", "perf", "gates", "frontier", "tiers")


def load_config(path: str | Path | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG
    with open(p, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{p}: config must be a mapping")
    missing = [k for k in REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise ValueError(f"{p}: missing sections: {', '.join(missing)}")
    return cfg


def load_model_spec(path: str | Path | None = None) -> ModelSpec:
    return ModelSpec.from_mapping(load_config(path)["model"])
