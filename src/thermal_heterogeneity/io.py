"""Deterministic input/output helpers for scientific payloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target


def write_table(
    frame: pd.DataFrame,
    path: str | Path,
    *,
    sort_by: Iterable[str] | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.copy()
    if sort_by:
        keys = list(sort_by)
        ordered = ordered.sort_values(keys, kind="mergesort").reset_index(drop=True)
    ordered.to_csv(target, index=False, lineterminator="\n", float_format="%.12g")
    return target


def canonical_table_hash(frame: pd.DataFrame, sort_by: Iterable[str]) -> str:
    ordered = frame.sort_values(list(sort_by), kind="mergesort").reset_index(drop=True)
    payload = ordered.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
