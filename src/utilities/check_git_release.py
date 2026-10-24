#!/usr/bin/env python3
"""Verify that the actual Git index is exactly the public allow-list."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from check_release_layout import _is_ignored_local


ALLOWLIST = "src/provenance/PUBLIC_FILE_ALLOWLIST.txt"
MAX_FILE_BYTES = 5 * 1024 * 1024


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _allowlist(root: Path) -> set[str]:
    return {
        line.strip()
        for line in (root / ALLOWLIST).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def check_repository(root: Path, require_clean: bool) -> dict[str, object]:
    root = root.resolve()
    errors: list[str] = []
    if not (root / ".git").exists():
        return {"status": "FAIL", "check": "git_release", "errors": ["Git is not initialized"]}

    expected = _allowlist(root)
    for path in sorted(expected):
        if _is_ignored_local(Path(path)):
            errors.append(f"local-only file must not be allow-listed: {path}")
    tracked = set(_git(root, "ls-files").stdout.splitlines())
    missing = sorted(expected - tracked)
    extra = sorted(tracked - expected)
    if missing:
        errors.append(f"allow-listed files are not tracked: {missing}")
    if extra:
        errors.append(f"tracked files are not allow-listed: {extra}")

    stage_lines = _git(root, "ls-files", "--stage").stdout.splitlines()
    bad_modes = sorted(
        line.split(maxsplit=1)[1]
        for line in stage_lines
        if line.split(maxsplit=1)[0] not in {"100644", "100755"}
    )
    if bad_modes:
        errors.append(f"non-regular Git entries are forbidden: {bad_modes}")

    oversized = sorted(
        path for path in tracked if (root / path).is_file() and (root / path).stat().st_size > MAX_FILE_BYTES
    )
    if oversized:
        errors.append(f"tracked files exceed the 5 MiB ceiling: {oversized}")

    status_lines = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines()
    if require_clean and status_lines:
        errors.append(f"working tree is not clean: {status_lines}")

    head_exists = _git(root, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
    fsck_status = "SKIP_NO_HISTORY"
    if head_exists:
        fsck = _git(root, "fsck", "--full", "--strict", check=False)
        fsck_status = "PASS" if fsck.returncode == 0 else "FAIL"
        if fsck.returncode != 0:
            errors.append("git fsck --full --strict failed")

    count_objects = {}
    for line in _git(root, "count-objects", "-v").stdout.splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            count_objects[key] = value

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "git_release",
        "allowlisted_file_count": len(expected),
        "tracked_file_count": len(tracked),
        "missing": missing,
        "extra": extra,
        "head_exists": head_exists,
        "fsck": fsck_status,
        "count_objects": count_objects,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--require-clean", action="store_true")
    args = parser.parse_args()
    result = check_repository(args.root, args.require_clean)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
