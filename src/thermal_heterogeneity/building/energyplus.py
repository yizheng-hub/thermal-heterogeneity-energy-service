"""Fail-closed adapter for optional Level 2 EnergyPlus trajectory reruns.

Level 2 accepts only the exact IDF/EPW assets named by the frozen manifest.  The
assets are intentionally not distributed in this repository.  No prototype,
weather file, or nearby run is substituted when the local rights-cleared input
package is incomplete.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping


EXPECTED_ENERGYPLUS_VERSION = "22.1.0-ed759b17ee"
LEVEL2_INPUT_PACKAGE_UNAVAILABLE = "LEVEL2_INPUT_PACKAGE_UNAVAILABLE"
LEVEL2_INPUT_PACKAGE_INVALID = "LEVEL2_INPUT_PACKAGE_INVALID"
LEVEL2_RUNTIME_UNAVAILABLE = "LEVEL2_RUNTIME_UNAVAILABLE"
LEVEL2_RUNTIME_VERSION_MISMATCH = "LEVEL2_RUNTIME_VERSION_MISMATCH"
LEVEL2_RUN_NOT_FOUND = "LEVEL2_RUN_NOT_FOUND"
LEVEL2_OUTPUT_NOT_EMPTY = "LEVEL2_OUTPUT_NOT_EMPTY"
LEVEL2_ENERGYPLUS_FAILED = "LEVEL2_ENERGYPLUS_FAILED"
LEVEL2_ENGINEERING_GATE_FAILED = "LEVEL2_ENGINEERING_GATE_FAILED"
LEVEL2_READY = "LEVEL2_READY"
LEVEL2_RUN_COMPLETE = "LEVEL2_RUN_COMPLETE"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_package_path(package_root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe package-relative path: {relative}")
    resolved = (package_root / candidate).resolve()
    if resolved != package_root and package_root not in resolved.parents:
        raise ValueError(f"package path escapes input root: {relative}")
    return resolved


def load_run_record(manifest_path: str | Path, run_id: str) -> dict[str, str] | None:
    """Return one exact frozen run record, or ``None`` when it is absent."""

    with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
        matches = [row for row in csv.DictReader(handle) if row.get("run_id") == run_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate run_id in manifest: {run_id}")
    return matches[0] if matches else None


def validate_level2_input_package(
    run: Mapping[str, str], input_package: str | Path
) -> dict[str, Any]:
    """Validate exact local assets and their user-supplied rights declaration.

    A missing directory, asset, declaration, or authorization returns
    ``LEVEL2_INPUT_PACKAGE_UNAVAILABLE``.  A present file with the wrong hash or
    a malformed rights manifest returns ``LEVEL2_INPUT_PACKAGE_INVALID``.
    """

    root = Path(input_package).expanduser().resolve()
    if not root.is_dir():
        return {
            "status": LEVEL2_INPUT_PACKAGE_UNAVAILABLE,
            "reason": "input package directory is missing",
            "missing": [str(root)],
        }

    try:
        rights_path = _safe_package_path(root, run["rights_manifest_path"])
        idf_path = _safe_package_path(root, run["idf_package_path"])
        epw_path = _safe_package_path(root, run["epw_package_path"])
    except (KeyError, ValueError) as exc:
        return {"status": LEVEL2_INPUT_PACKAGE_INVALID, "reason": str(exc)}

    missing = [
        path.relative_to(root).as_posix()
        for path in (rights_path, idf_path, epw_path)
        if not path.is_file()
    ]
    if missing:
        return {
            "status": LEVEL2_INPUT_PACKAGE_UNAVAILABLE,
            "reason": "one or more required Level 2 files are missing",
            "missing": missing,
        }

    try:
        rights = json.loads(rights_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": LEVEL2_INPUT_PACKAGE_INVALID, "reason": f"invalid rights manifest: {exc}"}
    if not isinstance(rights, dict) or not isinstance(rights.get("assets"), dict):
        return {"status": LEVEL2_INPUT_PACKAGE_INVALID, "reason": "rights manifest must contain an assets mapping"}
    if rights.get("package_id") != run.get("rights_package_id"):
        return {"status": LEVEL2_INPUT_PACKAGE_INVALID, "reason": "rights package_id does not match the run manifest"}
    if rights.get("local_use_authorized") is not True:
        return {
            "status": LEVEL2_INPUT_PACKAGE_UNAVAILABLE,
            "reason": "local-use authorization is absent from the rights package",
        }

    for kind, path in (("idf", idf_path), ("epw", epw_path)):
        asset_id = run[f"{kind}_asset_id"]
        expected_hash = run[f"{kind}_sha256"]
        declaration = rights["assets"].get(asset_id)
        if not isinstance(declaration, dict) or not declaration.get("rights_basis"):
            return {
                "status": LEVEL2_INPUT_PACKAGE_UNAVAILABLE,
                "reason": f"rights declaration is missing for {asset_id}",
            }
        if declaration.get("sha256") != expected_hash:
            return {
                "status": LEVEL2_INPUT_PACKAGE_INVALID,
                "reason": f"rights declaration hash does not match for {asset_id}",
            }
        observed_hash = _sha256(path)
        if not _SHA256.fullmatch(expected_hash) or observed_hash != expected_hash:
            return {
                "status": LEVEL2_INPUT_PACKAGE_INVALID,
                "reason": f"asset hash does not match for {asset_id}",
                "expected_sha256": expected_hash,
                "observed_sha256": observed_hash,
            }

    return {
        "status": LEVEL2_READY,
        "run_id": run["run_id"],
        "idf_path": str(idf_path),
        "epw_path": str(epw_path),
        "rights_path": str(rights_path),
    }


def prepare_level2_run(
    manifest_path: str | Path,
    run_id: str,
    input_package: str | Path,
    output_root: str | Path,
    *,
    energyplus_executable: str | Path = "energyplus",
) -> dict[str, Any]:
    """Resolve one run without executing it, preserving fail-closed statuses."""

    run = load_run_record(manifest_path, run_id)
    if run is None:
        return {"status": LEVEL2_RUN_NOT_FOUND, "run_id": run_id}
    checked = validate_level2_input_package(run, input_package)
    if checked["status"] != LEVEL2_READY:
        return checked | {"run_id": run_id}
    output_dir = Path(output_root).expanduser().resolve() / run_id
    argv = [
        str(energyplus_executable),
        "-w",
        checked["epw_path"],
        "-d",
        str(output_dir),
        checked["idf_path"],
    ]
    return checked | {
        "output_dir": str(output_dir),
        "argv": argv,
        "energyplus_version": run["energyplus_version"],
        "engineering_gate_id": run["engineering_gate_id"],
    }


def evaluate_engineering_gate(
    output_dir: str | Path, *, exit_code: int = 0
) -> dict[str, Any]:
    """Evaluate the locked Level 2 completion and severe/fatal-error gate."""

    root = Path(output_dir)
    err_path = root / "eplusout.err"
    end_path = root / "eplusout.end"
    if not err_path.is_file() or not end_path.is_file():
        return {
            "status": LEVEL2_ENGINEERING_GATE_FAILED,
            "gate_id": "LEVEL2_TRAJECTORY_GATE_V1",
            "reason": "required EnergyPlus completion files are missing",
        }
    err_text = err_path.read_text(encoding="utf-8", errors="replace")
    end_text = end_path.read_text(encoding="utf-8", errors="replace")
    severe_count = len(re.findall(r"\*\*\s*Severe\s*\*\*", err_text, flags=re.IGNORECASE))
    fatal_count = len(re.findall(r"\*\*\s*Fatal\s*\*\*", err_text, flags=re.IGNORECASE))
    version_ok = EXPECTED_ENERGYPLUS_VERSION in err_text
    completed = "EnergyPlus Completed Successfully" in end_text
    passed = exit_code == 0 and version_ok and completed and severe_count == 0 and fatal_count == 0
    return {
        "status": LEVEL2_RUN_COMPLETE if passed else LEVEL2_ENGINEERING_GATE_FAILED,
        "gate_id": "LEVEL2_TRAJECTORY_GATE_V1",
        "exit_code": exit_code,
        "version_ok": version_ok,
        "completed_successfully": completed,
        "severe_count": severe_count,
        "fatal_count": fatal_count,
    }


def _runtime_version(executable: str | Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": LEVEL2_RUNTIME_UNAVAILABLE, "reason": str(exc)}
    output = f"{completed.stdout}\n{completed.stderr}"
    if completed.returncode != 0:
        return {
            "status": LEVEL2_RUNTIME_UNAVAILABLE,
            "reason": "EnergyPlus --version returned a nonzero exit code",
        }
    if EXPECTED_ENERGYPLUS_VERSION not in output:
        return {
            "status": LEVEL2_RUNTIME_VERSION_MISMATCH,
            "expected": EXPECTED_ENERGYPLUS_VERSION,
            "observed": output.strip(),
        }
    return {"status": LEVEL2_READY, "observed": output.strip()}


def run_level2_trajectory(
    manifest_path: str | Path,
    run_id: str,
    input_package: str | Path,
    output_root: str | Path,
    *,
    energyplus_executable: str | Path = "energyplus",
) -> dict[str, Any]:
    """Run one exact frozen trajectory; never search for replacement assets."""

    prepared = prepare_level2_run(
        manifest_path,
        run_id,
        input_package,
        output_root,
        energyplus_executable=energyplus_executable,
    )
    if prepared["status"] != LEVEL2_READY:
        return prepared
    output_dir = Path(prepared["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        return {"status": LEVEL2_OUTPUT_NOT_EMPTY, "run_id": run_id, "output_dir": str(output_dir)}
    runtime = _runtime_version(energyplus_executable)
    if runtime["status"] != LEVEL2_READY:
        return runtime | {"run_id": run_id}
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        prepared["argv"],
        check=False,
        capture_output=True,
        text=True,
    )
    (output_dir / "console.log").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    gate = evaluate_engineering_gate(output_dir, exit_code=completed.returncode)
    if completed.returncode != 0:
        gate["status"] = LEVEL2_ENERGYPLUS_FAILED
    return gate | {"run_id": run_id, "output_dir": str(output_dir)}
