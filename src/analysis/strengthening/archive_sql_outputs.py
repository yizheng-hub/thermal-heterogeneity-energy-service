"""Losslessly compact completed new SQL outputs between simulation phases."""

from common import ROOT, write_json
from hvac_models import SIM
from single_event_simulation import lossless_sql_archive
import json
from pathlib import Path


def record_paths(value):
    if isinstance(value, dict):
        if value.get("completed") and value.get("sql"):
            yield Path(value["sql"])
        for child in value.values():
            yield from record_paths(child)
    elif isinstance(value, list):
        for child in value:
            yield from record_paths(child)


def main():
    paths = set()
    annual = SIM / "candidate_runs"
    if annual.exists():
        for folder in annual.iterdir():
            qa = folder / "STRENGTHENING_QA.json"
            if qa.is_file() and json.loads(qa.read_text()).get("completed"):
                paths.add(folder / "eplusout.sql")
                paths.add(SIM / "candidate_retry100" / folder.name / "eplusout.sql")
    for name in [
        "hvac_confirmation",
        "vav_outdoor_reset",
        "hvac_retuned",
        "placement_hvac_interaction",
    ]:
        for file in (ROOT / "results" / name).glob("*.json"):
            if any(token in file.name for token in ["OPERATIONS", "PHYSICAL", "CACHE", "LOGICAL"]):
                paths.update(record_paths(json.loads(file.read_text())))
    archived = []
    for path in sorted(paths):
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(SIM.resolve()):
            raise ValueError("SQL archival is restricted to this run's new simulation directory")
        result = lossless_sql_archive(path)
        write_json(path.parent / "RAW_SQL_LOSSLESS_ARCHIVE.json", result)
        archived.append({"path": str(path.relative_to(SIM)), **result})
    write_json(
        ROOT / "audit/LATEST_SQL_ARCHIVAL.json",
        {"status": "completed", "archived_files": len(archived), "files": archived},
    )
    print("Losslessly archived", len(archived), "completed new SQL files", flush=True)


if __name__ == "__main__":
    main()
