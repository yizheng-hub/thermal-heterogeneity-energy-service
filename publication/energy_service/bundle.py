"""Verify relative input manifests and independently regenerated commands."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify_bundle(root: Path, manifest: Path):
    root = root.resolve()
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig", newline="")))
    names = [row["path"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate bundle paths")
    for row in rows:
        name = PurePosixPath(row["path"])
        if name.is_absolute() or ".." in name.parts or "\\" in row["path"]:
            raise ValueError("Bundle manifest paths must be relative and contained")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("Bundle path escapes the supplied input root")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*")
              if p.is_file() and p.resolve() != manifest.resolve()}
    if actual != set(names):
        raise ValueError("Bundle has missing or undeclared files")
    return {"status": "PASS", "verified_input_files": len(rows),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def verify_commands(calculated: Path, reference: Path):
    keys = ["context", "composition_id", "layout", "permission", "control", "variant"]
    generated = json.loads(calculated.read_text(encoding="utf-8"))
    frozen = json.loads(reference.read_text(encoding="utf-8"))
    index = {tuple(row[k] for k in keys): row["targets_c"] for row in frozen}
    if len(index) != 1296 or len(generated) != 1296:
        raise ValueError("Incomplete frozen or regenerated command matrix")
    seen, maximum, native = set(), 0., 0
    for row in generated:
        key = tuple(row[k] for k in keys)
        if key in seen or key not in index:
            raise ValueError("Duplicated or unexpected controller design")
        seen.add(key)
        if set(row["targets_c"]) != set(index[key]):
            raise ValueError("The controlled thermal zones changed")
        maximum = max(maximum, max(abs(value - index[key][zone])
                                  for zone, value in row["targets_c"].items()))
        native += row["permission"] == "native_zone"
    if maximum != 0 or native != 432:
        raise ValueError("Independent controller regeneration differs from frozen commands")
    return {"status": "PASS", "logical_commands": len(generated),
            "native_commands": native, "maximum_target_error_c": maximum,
            "reference_used_only_after_generation": True}
