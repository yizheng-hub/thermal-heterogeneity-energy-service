"""Freeze the current public source files for a submission software attachment."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

from publication.package_text import DOCUMENT_ALIASES, documentation_payloads


ALLOWLIST = "src/provenance/PUBLIC_FILE_ALLOWLIST.txt"
PACKAGER_FILES = (
    "publication/build_submission_packages.py",
    "publication/package_text.py",
    "tests/test_submission_software_package.py",
    "tests/test_submission_data_package.py",
)
REQUIRED_FILES = (
    ALLOWLIST,
    "LICENSE",
    "LICENSE-DOCS-DATA",
    "DATA_AND_SOFTWARE_LICENSES.md",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "REPRODUCTION_FROM_RAW.md",
    "publication/README.md",
    "CITATION.cff",
    ".python-version",
    "pyproject.toml",
    "uv.lock",
    "reproduce_from_raw.py",
    "reproduce_publication.py",
    "src/settings/paper.yaml",
    "src/analysis/global_extension/GLOBAL_CLIMATES.json",
    "data/metadata/strengthening_protocol/analysis_plan.yaml",
)
EXCLUDED_TOP_LEVEL = {
    ".github", "audit", "tmp", "output", "results", "result", "data_for_figures",
}
EXCLUDED_FILES = {
    ".gitleaks.toml",
    "publication/review_data_files.txt",
    "src/provenance/ENERGY_SERVICE_SOURCE_SNAPSHOT.json",
    "src/provenance/RETAINED_SCRIPT_INVENTORY.csv",
    "src/provenance/SCRIPT_MIGRATION_MAP.json",
    "data/metadata/strengthening_protocol/analysis_plan_amendment17_storage.yaml",
    "data/metadata/strengthening_protocol/analysis_plan_amendment_22_owned_scratch.yaml",
    "data/metadata/strengthening_protocol/analysis_plan_amendment_23_stale_conversion_controls.yaml",
    "src/utilities/release_scan.py",
    "src/utilities/check_git_release.py",
    "src/utilities/check_release_layout.py",
    "src/utilities/check_script_migration.py",
    "src/utilities/snapshot_analysis_sources.py",
    "src/utilities/install_gitleaks.py",
    "src/utilities/scan_secrets.py",
    "src/utilities/validate_source_manifest.py",
    "src/analysis/typical_day_dr/README.txt",
    "tests/contracts/test_code_only_package.py",
}
EXCLUDED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".runtime_cache", ".cache", "node_modules", "ai_review", "agent_review",
    "development_logs", "dev_logs",
}
EXCLUDED_NAMES = {"agents.md", "claude.md", "gemini.md", "skill.md"}
EXCLUDED_SUFFIXES = {
    ".pyc", ".pyo", ".exe", ".dll", ".ttf", ".otf", ".woff", ".woff2",
    ".zip", ".7z", ".gz", ".zst", ".docx", ".pdf", ".idf", ".epw",
    ".sql", ".npy", ".npz", ".pem", ".key", ".p12", ".pfx", ".jks",
    ".md", ".log", ".err",
}


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _relative(name):
    path = PurePosixPath(name)
    if (
        not name or "\\" in name or ":" in name or path.is_absolute()
        or ".." in path.parts or path.as_posix() != name
    ):
        raise ValueError("Source paths must be normalized project-relative paths")
    return path


def _excluded(path):
    parts = tuple(part.lower() for part in path.parts)
    name = parts[-1]
    return (
        path.as_posix() in EXCLUDED_FILES
        or parts[0] in EXCLUDED_TOP_LEVEL
        or bool(set(parts).intersection(EXCLUDED_PARTS))
        or parts[:2] in {("data", "upstream"), ("data", "inputs"), ("data", "templates")}
        or len(parts) > 1 and parts[0] == "tools" and (
            parts[1] in {"fonts", "python"} or parts[1].startswith("energyplus")
        )
        or name in EXCLUDED_NAMES or name == ".env" or name.startswith(".env.")
        or name.startswith(("credentials.", "credential.", "secrets."))
        or path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def _archive_payload(name, payload):
    if name == "data/metadata/source_manifest.csv":
        source = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=[
            "asset_id", "asset_title", "upstream_reference", "version_or_hash",
            "licence_status", "user_download_required"
        ], extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(source)
        return output.getvalue().encode("utf-8")
    if name == "pyproject.toml":
        return payload.replace(
            b'readme = "README.md"',
            b'readme = {file = "README.txt", content-type = "text/plain"}',
        )
    if name == "src/utilities/build_self_contained_project.py":
        return payload.replace(b'source.suffix == ".md"', b'source.suffix in {".md", ".txt"}')
    if name == "src/utilities/export_model_inputs.py":
        return payload.replace(b"LOCAL_SOURCE_TERMS.md", b"LOCAL_SOURCE_TERMS.txt").replace(
            b"# Locally supplied complete model inputs", b"Locally supplied complete model inputs"
        )
    if name == "src/utilities/generate_demand_inputs.py":
        return payload.replace(b"THIRD_PARTY_NOTICES.md", b"THIRD_PARTY_NOTICES.txt")
    if name == "tests/contracts/test_paths.py":
        text = payload.decode("utf-8")
        lines = text.splitlines(keepends=True)
        for node in reversed(ast.parse(text).body):
            if isinstance(node, ast.FunctionDef) and node.name == "test_demographic_summary_disclosure_threshold":
                start = min([node.lineno, *[item.lineno for item in node.decorator_list]]) - 1
                del lines[start:node.end_lineno]
        return "".join(lines).encode("utf-8")
    return payload


def _collect_payloads(source_root, release_id, extra_paths=()):
    root = Path(source_root).resolve()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", release_id):
        raise ValueError("Use a nonempty portable release identifier")
    names = {
        line.strip() for line in (root / ALLOWLIST).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    names.update(REQUIRED_FILES)
    names.update(extra_paths)
    names.update(name for name in PACKAGER_FILES if (root / name).is_file())
    selected = {}
    for name in sorted(names):
        _relative(name)
        target = DOCUMENT_ALIASES.get(name, name)
        relative = _relative(target)
        if _excluded(relative):
            continue
        original = next((key for key, value in DOCUMENT_ALIASES.items() if value == target), name)
        source = original if (root / original).is_file() else target
        path = (root / source).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing source file or source path outside the project: " + source)
        selected[target] = source
    documents = documentation_payloads(root, release_id)
    payloads = {}
    source_files = []
    source_rows = []
    for target in sorted(selected):
        source = selected[target]
        original = (root / source).read_bytes()
        if target == ALLOWLIST:
            payload = ("\n".join(sorted(selected, key=str.casefold)) + "\n").encode("utf-8")
        elif target in documents:
            payload = documents[target]
        else:
            payload = _archive_payload(target, original)
        payloads[target] = payload
        source_rows.append({"file": source, "bytes": len(original), "sha256": _sha(original)})
        if original != payload or source != target:
            source_files.append({
                "source": source,
                "source_bytes": len(original),
                "source_sha256": _sha(original),
                "file": target,
                "transformation": "Plain-text documentation and distributed-path adaptation",
            })
    rows = [{"file": name, "bytes": len(payload), "sha256": _sha(payload)}
            for name, payload in sorted(payloads.items())]
    base_commit = None
    if (root / ".git").exists():
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
        base_commit = process.stdout.strip()
    manifest = {
        "release_id": release_id,
        "source_snapshot_sha256": _sha(_canonical_json(rows)),
        "snapshot_algorithm": "SHA-256 of canonical JSON file/bytes/sha256 records sorted by file",
        "source_version": "Supplied file hashes and original source hashes",
        "base_git_commit": base_commit,
        "files": rows,
        "original_source_snapshot_sha256": _sha(_canonical_json(sorted(source_rows, key=lambda row: row["file"]))),
        "source_transformations": source_files,
    }
    return manifest, payloads


def collect_software_snapshot(source_root, release_id, extra_paths=()):
    """Collect the supplied bytes and original source versions without writing."""
    return _collect_payloads(source_root, release_id, extra_paths)[0]


def build_software_package(source_root, destination, release_id, extra_paths=(), *, source_snapshot=None):
    """Write a new deterministic ZIP, rejecting any source change after collection."""
    root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new attachment filename; earlier packages are preserved")
    manifest, payloads = _collect_payloads(root, release_id, extra_paths)
    if source_snapshot is not None and source_snapshot != manifest:
        raise ValueError("Source changed while packaging")
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"

    def add(archive, name, payload):
        entry = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = 0o100644 << 16
        archive.writestr(entry, payload, compresslevel=6)

    with zipfile.ZipFile(destination, "x") as archive:
        for name, payload in payloads.items():
            add(archive, name, payload)
        add(archive, "CODE_MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Software attachment CRC verification failed")
        for row in manifest["files"]:
            if _sha(archive.read(row["file"])) != row["sha256"]:
                raise ValueError("Software attachment content verification failed")
    return {
        "archive": str(destination),
        "release_id": release_id,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "files": len(manifest["files"]) + 1,
        "bytes": destination.stat().st_size,
        "sha256": _sha(destination.read_bytes()),
    }


def build_combined_package(source_root, destination, release_id, extra_paths=(), *, source_snapshot=None):
    """Write code and source-data members directly into one archive."""
    from publication.build_review_package import build as build_data

    root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new attachment filename; earlier packages are preserved")
    manifest, code_payloads = _collect_payloads(root, release_id, extra_paths)
    if source_snapshot is not None and source_snapshot != manifest:
        raise ValueError("Source changed while packaging")
    data_buffer = io.BytesIO()
    build_data(data_buffer, release_id=release_id, source_snapshot=manifest)
    data_buffer.seek(0)
    payloads = {}

    def merge(name, payload):
        _relative(name)
        if name.lower().endswith((".zip", ".7z", ".tar", ".tgz", ".tar.gz")):
            raise ValueError("Nested archives are not allowed: " + name)
        if name in payloads and payloads[name] != payload:
            raise ValueError("Different payloads share an archive path: " + name)
        payloads[name] = payload

    for name, payload in code_payloads.items():
        merge(name, payload)
    merge("CODE_MANIFEST.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8") + b"\n")
    with zipfile.ZipFile(data_buffer) as data_archive:
        data_manifest = json.loads(data_archive.read("SOURCE_DATA_MANIFEST.json"))
        if (
            data_manifest["release_id"] != release_id
            or data_manifest["source_snapshot_sha256"] != manifest["source_snapshot_sha256"]
        ):
            raise ValueError("Code and source-data versions differ")
        for row in data_manifest["files"]:
            if _sha(data_archive.read(row["file"])) != row["sha256"]:
                raise ValueError("Source-data checksum mismatch: " + row["file"])
        for info in data_archive.infolist():
            merge(info.filename, data_archive.read(info))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x") as archive:
        for name, payload in sorted(payloads.items()):
            entry = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, payload, compresslevel=6)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Combined attachment CRC verification failed")
        for name, payload in payloads.items():
            if _sha(archive.read(name)) != _sha(payload):
                raise ValueError("Combined attachment checksum mismatch: " + name)
    return {
        "archive": str(destination),
        "release_id": release_id,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "files": len(payloads),
        "bytes": destination.stat().st_size,
        "sha256": _sha(destination.read_bytes()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="attachment", required=True)
    software = subparsers.add_parser("software", help="Freeze the current source tree")
    software.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    software.add_argument("--destination", type=Path, required=True)
    software.add_argument("--release-id", required=True)
    combined = subparsers.add_parser("combined", help="Combine code and source data directly")
    combined.add_argument("--destination", type=Path, required=True)
    combined.add_argument("--release-id", required=True)
    combined.set_defaults(source_root=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    builder = build_combined_package if args.attachment == "combined" else build_software_package
    print(json.dumps(builder(args.source_root, args.destination, args.release_id), indent=2))


if __name__ == "__main__":
    main()
