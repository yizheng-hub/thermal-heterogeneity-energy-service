"""Assemble a Windows project with local raw inputs and both local runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utilities.audit_reproduction_inputs import raw_inputs  # noqa: E402


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publication_inputs(root):
    """Authoring inputs are optional but must be complete when requested."""
    paths = [root / name for name in [
        "publication/Main_manuscript.md", "publication/Supplementary_information.md",
        "data/templates/Manuscript.docx", "data/templates/Original_Main_manuscript.docx",
        "Result/FIGURE_DATA_MANIFEST.json",
    ]]
    manifest = paths[-1]
    if not manifest.is_file():
        raise FileNotFoundError("Publication assembly requires the Result manifest")
    for row in json.loads(manifest.read_text())["files"]:
        path = (root / row["result"]).resolve()
        if not path.is_relative_to(root.resolve() / "Result") or not path.is_file():
            raise ValueError("Publication input is absent or outside Result")
        if sha(path) != row["sha256"]:
            raise ValueError("Publication input checksum changed")
        paths.append(path)
    return paths


def build(destination, python_home, site_packages, include_publication=False):
    if destination.exists():
        raise FileExistsError("Choose a new destination; an existing project is not overwritten")
    allowed = [
        line.strip()
        for line in (ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    omitted_methods = sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts
        and path.relative_to(ROOT).as_posix() not in allowed
    )
    if omitted_methods:
        raise ValueError("Scientific source files are missing from the package allowlist: " + ", ".join(omitted_methods))
    sources = [ROOT / name for name in allowed] + raw_inputs(ROOT, include_validation=True)
    sources += list((ROOT / "data/metadata").glob("*.json")) + list(
        (ROOT / "data/metadata").glob("*.csv")
    )
    if include_publication:
        sources += publication_inputs(ROOT)
    for path in sources:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise FileNotFoundError(
                f"Required project input is absent or outside the source project: {path}"
            )
    if not (python_home / "python.exe").is_file() or not (python_home / "python311.dll").is_file():
        raise ValueError("The locked Windows CPython3.11 runtime is required")
    destination.mkdir(parents=True)
    for path in sorted(set(sources)):
        target = destination / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for source in (ROOT / "data/upstream").rglob("*"):
        if source.is_file() and (source.suffix == ".md" or source.name.startswith("LICENSE")):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copytree(ROOT / "tools/EnergyPlus-22.1.0", destination / "tools/EnergyPlus-22.1.0")
    shutil.copytree(ROOT / "tools/fonts", destination / "tools/fonts")
    if include_publication and (ROOT / "tools/document_libraries").is_dir():
        shutil.copytree(ROOT / "tools/document_libraries", destination / "tools/document_libraries")
    local_python = destination / "tools/python"
    shutil.copytree(
        python_home, local_python, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    local_site = local_python / "Lib/site-packages"
    local_site.mkdir(parents=True, exist_ok=True)
    for source in site_packages.iterdir():
        if source.name.startswith(("__editable__", "thermal_heterogeneity-")):
            continue
        target = local_site / source.name
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            shutil.copy2(source, target)
    verify(destination)


def verify(destination):
    local_python = destination / "tools/python"
    # Windows _pth paths are relative to the runtime. They isolate imports from
    # registry, user-site and environment paths, including after a folder move.
    (local_python / "python311._pth").write_text(
        "python311.zip\n.\nDLLs\nLib\nLib/site-packages\n../../src\nimport site\n"
    )
    # Isolated _pth mode omits the normal script-directory import entry.
    # Restore it only for scripts physically inside this assembled project.
    (local_python / "Lib/sitecustomize.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "project = Path(__file__).resolve().parents[3]\nsys.path.insert(0, str(project))\n"
        "if sys.argv and sys.argv[0] and not sys.argv[0].startswith('-'):\n"
        "    script = Path(sys.argv[0]).resolve()\n"
        "    if script.is_file() and script.is_relative_to(project):\n"
        "        sys.path.insert(0, str(script.parent))\n"
    )
    bootstrap = """$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    $env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.runtime_cache/matplotlib'
    $env:MPLBACKEND = 'Agg'
    & (Join-Path $PSScriptRoot 'tools/python/python.exe') (Join-Path $PSScriptRoot 'reproduce_from_raw.py') @args
    $reproductionExit = $LASTEXITCODE
} finally { Pop-Location }
exit $reproductionExit
"""
    (destination / "run.ps1").write_text(bootstrap, encoding="utf-8")
    probe = "import sys,json,numpy,pandas,scipy,sklearn,matplotlib,zstandard,pathlib,thermal_heterogeneity;print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'isolated':sys.flags.isolated,'modules':{m.__name__:m.__file__ for m in [numpy,pandas,scipy,sklearn,matplotlib,zstandard,pathlib,thermal_heterogeneity]}}))"
    process = subprocess.run(
        [str(local_python / "python.exe"), "-c", probe],
        cwd=destination,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "MPLCONFIGDIR": str(destination / ".runtime_cache/matplotlib"),
            "MPLBACKEND": "Agg",
        },
    )
    if process.returncode:
        raise RuntimeError("Contained Python import check failed:\n" + process.stderr)
    runtime = json.loads(process.stdout)
    for path in [runtime["executable"], runtime["prefix"], *runtime["modules"].values()]:
        if not Path(path).resolve().is_relative_to(destination):
            raise ValueError(
                "The assembled Python runtime imported a dependency outside the project"
            )
    if runtime["isolated"] != 1:
        raise ValueError("The local Python import paths are not isolated")
    files = {}
    for path in sorted(destination.rglob("*")):
        if (
            path.is_file()
            and not {"__pycache__", "output", "audit", ".runtime_cache", ".pytest_cache"}.intersection(
                path.relative_to(destination).parts
            )
            and path.name != "SELF_CONTAINED_PROJECT_MANIFEST.json"
        ):
            files[path.relative_to(destination).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha(path),
            }
    report = {
        "status": "inputs_and_runtimes_assembled",
        "files": files,
        "runtime_probe": {"isolated": 1, "all_module_paths_inside_project": True},
        "simulation_completion": "Recorded separately; assembly is not a completed end-to-end simulation check.",
        "entry_point": "run.ps1 --output-root output/from_raw",
        "external_scientific_file_dependencies": [],
    }
    (destination / "SELF_CONTAINED_PROJECT_MANIFEST.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({"destination": str(destination), "files": len(files), "isolated_python": True}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--python-home", type=Path)
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--verify-existing", action="store_true")
    parser.add_argument("--include-publication", action="store_true",
                        help="Include retained author text, templates and verified Result data")
    args = parser.parse_args()
    if args.verify_existing:
        verify(args.destination.resolve())
    elif args.python_home and args.site_packages:
        build(args.destination.resolve(), args.python_home.resolve(), args.site_packages.resolve(),
              include_publication=args.include_publication)
    else:
        parser.error("A new build requires --python-home and --site-packages")
