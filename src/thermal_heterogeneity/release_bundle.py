"""Deterministic Level-1 analysis-bundle construction, retrieval and verification."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import zstandard


BUNDLE_SCHEMA = "thermal-heterogeneity-analysis-bundle-v1"
LOCATOR_ENV = "THERMAL_HETEROGENEITY_ANALYSIS_BUNDLE_URL"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_logical_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise ValueError(f"invalid logical path: {value!r}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"invalid logical path: {value!r}")
    return logical


def read_analysis_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "release_id",
        "logical_path",
        "table_id",
        "row_count",
        "column_count",
        "primary_key",
        "sha256",
        "size_bytes",
    }
    if not rows:
        raise ValueError("analysis manifest is empty")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"analysis manifest missing columns: {sorted(missing)}")
    logical_paths = [str(_safe_logical_path(row["logical_path"])) for row in rows]
    if len(logical_paths) != len(set(logical_paths)):
        raise ValueError("analysis manifest contains duplicate logical paths")
    table_ids = [row["table_id"] for row in rows]
    if len(table_ids) != len(set(table_ids)):
        raise ValueError("analysis manifest contains duplicate table IDs")
    release_ids = {row["release_id"] for row in rows}
    if len(release_ids) != 1:
        raise ValueError("analysis manifest must contain one release ID")
    return sorted(rows, key=lambda row: row["logical_path"])


def _payload_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def _basic_manifest_check(root: Path, rows: list[dict[str, str]]) -> None:
    expected = {row["logical_path"] for row in rows}
    actual = _payload_files(root)
    if actual != expected:
        raise ValueError(
            "payload file set differs from manifest: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for row in rows:
        path = root / Path(*PurePosixPath(row["logical_path"]).parts)
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"size mismatch: {row['logical_path']}")
        if sha256_file(path) != row["sha256"].lower():
            raise ValueError(f"SHA-256 mismatch: {row['logical_path']}")


def build_bundle(
    input_root: Path,
    output_path: Path,
    manifest_path: Path,
    release_record_path: Path,
    *,
    source_date_epoch: int = 0,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_path = output_path.resolve()
    manifest_path = manifest_path.resolve()
    release_record_path = release_record_path.resolve()
    if not input_root.is_dir():
        raise ValueError(f"input root is not a directory: {input_root}")
    rows = read_analysis_manifest(manifest_path)
    _basic_manifest_check(input_root, rows)
    if output_path.suffixes[-2:] != [".tar", ".zst"]:
        raise ValueError("analysis bundle output must end in .tar.zst")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    release_record_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".analysis-bundle-", dir=output_path.parent
    ) as temporary:
        temporary_root = Path(temporary)
        tar_path = temporary_root / "payload.tar"
        compressed_path = temporary_root / "payload.tar.zst"
        with tarfile.open(tar_path, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for row in rows:
                source = input_root / Path(*PurePosixPath(row["logical_path"]).parts)
                info = tarfile.TarInfo(row["logical_path"])
                info.size = source.stat().st_size
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = source_date_epoch
                with source.open("rb") as handle:
                    archive.addfile(info, handle)
        compressor = zstandard.ZstdCompressor(
            level=19,
            threads=0,
            write_checksum=True,
            write_content_size=True,
            write_dict_id=False,
        )
        with tar_path.open("rb") as source, compressed_path.open("wb") as destination:
            compressor.copy_stream(source, destination)
        os.replace(compressed_path, output_path)

    files = [
        {
            "logical_path": row["logical_path"],
            "sha256": row["sha256"].lower(),
            "size_bytes": int(row["size_bytes"]),
        }
        for row in rows
    ]
    record: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "release_id": rows[0]["release_id"],
        "tier": "analysis",
        "archive_format": "tar+zstd",
        "bundle_filename": output_path.name,
        "bundle_sha256": sha256_file(output_path),
        "bundle_size_bytes": output_path.stat().st_size,
        "file_count": len(files),
        "payload_size_bytes": sum(item["size_bytes"] for item in files),
        "analysis_manifest": manifest_path.name,
        "analysis_manifest_sha256": sha256_file(manifest_path),
        "source_date_epoch": source_date_epoch,
        "locator": f"env:{LOCATOR_ENV}",
        "files": files,
    }
    release_record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return record


def _count_csv(path: Path, key_columns: list[str]) -> tuple[int, int, list[str]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    seen: set[tuple[str, ...]] = set()
    duplicate = False
    null_key = False
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        missing_keys = [key for key in key_columns if key not in fields]
        count = 0
        for row in reader:
            count += 1
            if key_columns and not missing_keys:
                key = tuple(row[column] for column in key_columns)
                if any(value == "" for value in key):
                    null_key = True
                if key in seen:
                    duplicate = True
                seen.add(key)
    errors = []
    if missing_keys:
        errors.append(f"missing primary-key columns {missing_keys}")
    if null_key:
        errors.append("null primary-key value")
    if duplicate:
        errors.append("duplicate primary key")
    return count, len(fields), errors


def verify_payload(root: Path, manifest_path: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    errors: list[str] = []
    try:
        rows = read_analysis_manifest(manifest_path)
    except (OSError, ValueError) as exc:
        return {
            "status": "FAIL",
            "check": "analysis_bundle_payload",
            "errors": [str(exc)],
        }
    if not root.is_dir():
        return {
            "status": "FAIL",
            "check": "analysis_bundle_payload",
            "errors": [f"payload root is not a directory: {root}"],
        }
    expected = {row["logical_path"] for row in rows}
    actual = _payload_files(root)
    if actual != expected:
        errors.append(
            "payload file set differs from manifest: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    hash_passed = 0
    structure_passed = 0
    for row in rows:
        path = root / Path(*PurePosixPath(row["logical_path"]).parts)
        if not path.is_file():
            continue
        if path.stat().st_size != int(row["size_bytes"]):
            errors.append(f"size mismatch: {row['logical_path']}")
        elif sha256_file(path) != row["sha256"].lower():
            errors.append(f"SHA-256 mismatch: {row['logical_path']}")
        else:
            hash_passed += 1
        if path.suffix == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                structure_passed += 1
            except (OSError, json.JSONDecodeError) as exc:
                errors.append(f"invalid JSON {row['logical_path']}: {exc}")
            continue
        keys = [key for key in row["primary_key"].split(";") if key]
        try:
            row_count, column_count, table_errors = _count_csv(path, keys)
        except (OSError, csv.Error, UnicodeError) as exc:
            errors.append(f"cannot read table {row['logical_path']}: {exc}")
            continue
        if row_count != int(row["row_count"]):
            errors.append(f"row-count mismatch: {row['logical_path']}")
        if column_count != int(row["column_count"]):
            errors.append(f"column-count mismatch: {row['logical_path']}")
        errors.extend(f"{row['logical_path']}: {message}" for message in table_errors)
        if (
            row_count == int(row["row_count"])
            and column_count == int(row["column_count"])
            and not table_errors
        ):
            structure_passed += 1
    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "analysis_bundle_payload",
        "release_id": rows[0]["release_id"],
        "expected_file_count": len(rows),
        "actual_file_count": len(actual),
        "hashes_passed": hash_passed,
        "structures_passed": structure_passed,
        "errors": errors,
    }


def _validate_source_decisions(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("source manifest is empty")
    if any(row.get("decision_status") != "LOCKED_R1" for row in rows):
        raise ValueError("source manifest contains an unlocked decision")
    if any("UNKNOWN" in str(value).upper() for row in rows for value in row.values()):
        raise ValueError("source manifest contains UNKNOWN state")


def _copy_or_download(locator: str, destination: Path) -> None:
    direct_path = Path(locator)
    if direct_path.is_absolute():
        if not direct_path.is_file():
            raise ValueError("analysis bundle locator does not identify a file")
        shutil.copyfile(direct_path, destination)
        return
    parsed = urllib.parse.urlparse(locator)
    if parsed.scheme in {"http", "https"}:
        with urllib.request.urlopen(locator, timeout=120) as response, destination.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        return
    if parsed.scheme == "file":
        source = Path(urllib.request.url2pathname(parsed.path))
        if parsed.netloc:
            source = Path(f"//{parsed.netloc}{urllib.request.url2pathname(parsed.path)}")
    elif parsed.scheme == "":
        source = Path(locator)
    else:
        raise ValueError(f"unsupported bundle locator scheme: {parsed.scheme}")
    if not source.is_file():
        raise ValueError("analysis bundle locator does not identify a file")
    shutil.copyfile(source, destination)


def _extract_bundle(archive_path: Path, destination: Path, expected: set[str]) -> None:
    tar_path = archive_path.with_suffix("")
    decompressor = zstandard.ZstdDecompressor()
    with archive_path.open("rb") as source, tar_path.open("wb") as target:
        decompressor.copy_stream(source, target)
    try:
        with tarfile.open(tar_path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ValueError("bundle contains duplicate member names")
            if set(names) != expected:
                raise ValueError(
                    "bundle members differ from release record: "
                    f"missing={sorted(expected - set(names))}, extra={sorted(set(names) - expected)}"
                )
            for member in members:
                logical = _safe_logical_path(member.name)
                if not member.isfile() or member.issym() or member.islnk():
                    raise ValueError(f"non-regular bundle member: {member.name}")
                target = destination.joinpath(*logical.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read bundle member: {member.name}")
                with extracted, target.open("wb") as output:
                    shutil.copyfileobj(extracted, output, length=1024 * 1024)
    finally:
        tar_path.unlink(missing_ok=True)


def fetch_bundle(
    *,
    tier: str,
    source_manifest_path: Path,
    bundle_record_path: Path,
    analysis_manifest_path: Path,
    destination: Path,
    locator_override: str | None = None,
) -> dict[str, Any]:
    if tier != "analysis":
        raise ValueError("only the locked analysis tier is available")
    _validate_source_decisions(source_manifest_path.resolve())
    record = json.loads(bundle_record_path.resolve().read_text(encoding="utf-8"))
    if record.get("schema") != BUNDLE_SCHEMA or record.get("tier") != tier:
        raise ValueError("bundle release record does not match the requested tier")
    if sha256_file(analysis_manifest_path.resolve()) != record["analysis_manifest_sha256"]:
        raise ValueError("analysis manifest hash differs from the bundle release record")
    manifest_rows = read_analysis_manifest(analysis_manifest_path.resolve())
    expected_files = {row["logical_path"] for row in manifest_rows}
    record_files = {item["logical_path"] for item in record.get("files", [])}
    if expected_files != record_files or len(expected_files) != int(record.get("file_count", -1)):
        raise ValueError("bundle release record and analysis manifest differ")

    locator = locator_override
    if not locator:
        locked_locator = str(record.get("locator", ""))
        if locked_locator.startswith("env:"):
            locator = os.environ.get(locked_locator.removeprefix("env:"), "")
        else:
            locator = locked_locator
    if not locator or locator == "ZENODO_DOI_PENDING":
        raise ValueError(
            f"analysis bundle location unavailable; set {LOCATOR_ENV} or pass --url"
        )
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("destination must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = destination.parent / f".{destination.name}.fetch-{uuid.uuid4().hex}"
    work.mkdir()
    try:
        archive_path = work / record["bundle_filename"]
        _copy_or_download(locator, archive_path)
        if archive_path.stat().st_size != int(record["bundle_size_bytes"]):
            raise ValueError("downloaded bundle size mismatch")
        actual_bundle_sha = sha256_file(archive_path)
        if actual_bundle_sha != record["bundle_sha256"]:
            raise ValueError("downloaded bundle SHA-256 mismatch")
        payload = work / "payload"
        payload.mkdir()
        _extract_bundle(archive_path, payload, expected_files)
        verification = verify_payload(payload, analysis_manifest_path)
        if verification["status"] != "PASS":
            raise ValueError(f"extracted payload verification failed: {verification['errors']}")
        archive_path.unlink()
        os.replace(payload, destination)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {
        "status": "PASS",
        "tier": tier,
        "release_id": record["release_id"],
        "bundle_sha256": record["bundle_sha256"],
        "file_count": len(expected_files),
        "destination": str(destination),
    }
