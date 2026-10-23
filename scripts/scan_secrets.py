#!/usr/bin/env python3
"""Scan the exact Git index, staged diff and existing history with Gitleaks."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def _run(
    command: list[str],
    root: Path,
    *,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        cwd=root,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.decode("utf-8", errors="replace"))
        if result.stderr:
            print(result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"secret scan failed with exit code {result.returncode}")
    return result


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def scan(root: Path, gitleaks: Path) -> dict[str, object]:
    root = root.resolve()
    gitleaks = gitleaks.resolve()
    if not gitleaks.is_file():
        raise FileNotFoundError(f"Gitleaks executable not found: {gitleaks}")
    if not (root / ".git").exists():
        raise RuntimeError("secret scanning requires an initialized Git repository")

    tracked_bytes = _git(root, "ls-files", "-z").stdout
    tracked = [item.decode("utf-8") for item in tracked_bytes.split(b"\0") if item]
    base = [str(gitleaks), "--no-banner", "--no-color", "--redact=100"]
    scans: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix="tracked-index-") as temporary:
        export_root = Path(temporary)
        for relative in tracked:
            destination = export_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_git(root, "show", f":{relative}").stdout)
        _run([*base, "dir", str(export_root)], root)
        scans["exact_index"] = "PASS"

    staged_diff = _git(root, "diff", "--cached", "--binary", "--no-ext-diff", "--no-renames").stdout
    _run([*base, "stdin"], root, input_bytes=staged_diff)
    scans["staged_diff"] = "PASS"

    head_exists = _git(root, "rev-parse", "--verify", "HEAD", check=False).returncode == 0
    if head_exists:
        _run([*base, "git", "."], root)
        scans["history"] = "PASS"
    else:
        scans["history"] = "SKIP_EMPTY_HISTORY"

    version = _run([str(gitleaks), "version"], root).stdout.decode("utf-8").strip()
    return {
        "status": "PASS",
        "tool": "gitleaks",
        "version": version,
        "tracked_files_scanned": len(tracked),
        "coverage": {
            "provider_token_rules": "Gitleaks embedded default rules",
            "private_keys": "Gitleaks embedded default rules",
            "credential_assignments": "Gitleaks embedded default rules",
            "generic_high_entropy_secrets": "Gitleaks embedded default rules",
        },
        "scans": scans,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gitleaks", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(scan(args.root, args.gitleaks), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
