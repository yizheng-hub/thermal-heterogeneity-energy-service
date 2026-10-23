#!/usr/bin/env python3
"""Install the pinned official Gitleaks binary after checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


VERSION = "8.29.1"
RELEASE_BASE = f"https://github.com/gitleaks/gitleaks/releases/download/v{VERSION}"
ASSETS = {
    ("Linux", "x86_64"): (
        f"gitleaks_{VERSION}_linux_x64.tar.gz",
        "e4eb209d04e20339d77122a3bdf9cd41351255cfb27ebcb75e85325e04f88924",
        "gitleaks",
    ),
    ("Windows", "AMD64"): (
        f"gitleaks_{VERSION}_windows_x64.zip",
        "e4b7d556f0cddbe23d10d8fac2ab0f29f68f019091c6599ffbeaa8a4fb71ac78",
        "gitleaks.exe",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_checksum(text: str, asset_name: str) -> str:
    matches = [
        line.split()[0].lower()
        for line in text.splitlines()
        if len(line.split()) == 2 and line.split()[1] == asset_name
    ]
    if len(matches) != 1 or len(matches[0]) != 64:
        raise ValueError(f"official checksum entry is not unique for {asset_name}")
    return matches[0]


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "thermal-heterogeneity-release"})
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def install(destination: Path) -> dict[str, object]:
    key = (platform.system(), platform.machine())
    if key not in ASSETS:
        raise RuntimeError(f"unsupported Gitleaks platform: {key[0]} {key[1]}")
    asset_name, pinned_checksum, executable_name = ASSETS[key]
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    executable = destination / executable_name

    with tempfile.TemporaryDirectory(prefix="gitleaks-download-") as temporary:
        temporary_root = Path(temporary)
        checksums = temporary_root / f"gitleaks_{VERSION}_checksums.txt"
        archive = temporary_root / asset_name
        _download(f"{RELEASE_BASE}/{checksums.name}", checksums)
        _download(f"{RELEASE_BASE}/{asset_name}", archive)
        official_checksum = _official_checksum(checksums.read_text(encoding="utf-8"), asset_name)
        actual_checksum = _sha256(archive)
        if official_checksum != pinned_checksum:
            raise RuntimeError("official checksum does not match the repository-pinned checksum")
        if actual_checksum != official_checksum:
            raise RuntimeError("downloaded Gitleaks archive failed checksum verification")

        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive) as package:
                binary = package.read(executable_name)
        else:
            with tarfile.open(archive, "r:gz") as package:
                member = package.getmember(executable_name)
                extracted = package.extractfile(member)
                if extracted is None:
                    raise RuntimeError("Gitleaks executable is missing from the archive")
                binary = extracted.read()
        executable.write_bytes(binary)
        if os.name != "nt":
            executable.chmod(0o755)

    return {
        "status": "PASS",
        "tool": "gitleaks",
        "version": VERSION,
        "asset": asset_name,
        "official_checksum": pinned_checksum,
        "checksum_verified": True,
        "executable": str(executable),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.destination), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
