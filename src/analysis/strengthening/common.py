from pathlib import Path
import hashlib
import json
import os
import sys

CODE = Path(__file__).resolve().parents[3]
RAW_RUN = Path(os.environ.get("HEM_RAW_RUN", str(CODE / "results/new_run"))).resolve()
ROOT = Path(os.environ.get("HEM_STRENGTHENING_OUTPUT", str(RAW_RUN / "strengthening"))).resolve()
PROJECT = CODE.parent
SNAPSHOT = CODE
GLOBAL = CODE / "src/analysis/global_extension"
for folder in [
    CODE,
    CODE / "src",
    GLOBAL,
    CODE / "src/analysis/energy_service",
    CODE / "src/analysis/targeted_extension",
]:
    sys.path.insert(0, str(folder))
MATRIX = RAW_RUN / "matrix"
RESULTS = RAW_RUN / "results"
DEMAND_REBUILD = RAW_RUN / "demand"
RUNTIME = Path(
    os.environ.get("HEM_ENERGYPLUS", str(CODE / "tools/EnergyPlus-22.1.0/energyplus.exe"))
).resolve()
if not all(path.is_relative_to(CODE) for path in [RAW_RUN, ROOT, RUNTIME]):
    raise ValueError("Demand inputs, generated results and EnergyPlus must remain inside this project")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def record_status(task, status, evidence, blockers=None):
    write_json(
        ROOT / "audit/status" / (task + ".json"),
        {"task": task, "status": status, "evidence": evidence, "blockers": blockers or []},
    )
