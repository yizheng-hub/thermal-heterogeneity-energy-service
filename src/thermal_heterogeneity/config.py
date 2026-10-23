"""Configuration loading with paths resolved relative to the configuration file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(
    path: str | Path, data_root: str | Path | None = None
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("paper configuration must be a mapping")
    payload["_config_path"] = config_path
    payload["_config_dir"] = config_path.parent
    configured_root = data_root or os.environ.get("THERMAL_HETEROGENEITY_DATA_ROOT")
    if configured_root is not None:
        payload["_data_root"] = Path(configured_root).expanduser().resolve()
    return payload


def resolve_asset(config: dict[str, Any], asset_id: str) -> Path:
    assets = config.get("assets", {})
    if asset_id not in assets:
        raise KeyError(f"unknown asset_id: {asset_id}")
    value = assets[asset_id]
    if not isinstance(value, str):
        raise TypeError(f"asset path must be a string: {asset_id}")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"asset paths must be relative: {asset_id}")
    root = config.get("_data_root")
    if root is None:
        configured = config.get("data_root")
        if configured is None:
            raise ValueError(
                "Level-1 data root is required; pass --data-root or set "
                "THERMAL_HETEROGENEITY_DATA_ROOT"
            )
        configured_path = Path(configured)
        if configured_path.is_absolute():
            raise ValueError("configured data_root must be relative")
        root = (config["_config_dir"] / configured_path).resolve()
    return (Path(root) / path).resolve()


def output_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    value = config.get("output", "../outputs")
    path = Path(value)
    if path.is_absolute():
        raise ValueError("configured output path must be relative")
    return (config["_config_dir"] / path).resolve()
