"""Losslessly archive deferred SQL only after compact and metadata verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/strengthening"))
from common import digest, write_json  # noqa: E402
from single_event_simulation import lossless_sql_archive  # noqa: E402


def archive_ready(raw):
    archived = []
    for record in (raw / "matrix").rglob("RUN_QA.json"):
        sql = record.with_name("eplusout.sql")
        if not sql.is_file() or time.time() - record.stat().st_mtime < 60:
            continue
        qa = json.loads(record.read_text())
        if not qa.get("completed") or not qa.get("sql_metadata_sha256"):
            continue
        metadata, compact = Path(qa["sql_metadata_path"]), Path(qa["compact_output"])
        if not all(p.resolve().is_relative_to(raw) for p in [sql, metadata, compact]):
            raise ValueError("Deferred SQL and its verified products must stay inside this raw run")
        if digest(metadata) != qa["sql_metadata_sha256"] or digest(compact) != qa["hourly_sha256"]:
            raise ValueError("Do not archive SQL when a required retained product has changed")
        result = lossless_sql_archive(sql)
        write_json(record.with_name("RAW_SQL_LOSSLESS_ARCHIVE.json"), result)
        archived.append(str(sql.relative_to(raw)))
    return archived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    raw = args.raw_run_root.resolve()
    if not raw.is_relative_to(ROOT) or raw == ROOT:
        raise ValueError("Archive only generated outputs inside this project")
    while True:
        archived = archive_ready(raw)
        if archived:
            write_json(raw / "audit/LATEST_DEFERRED_SQL_ARCHIVE.json", {"files": archived})
            print("Verified lossless SQL archives:", len(archived), flush=True)
        if not args.watch or (raw / "audit/STOP_SQL_ARCHIVER").exists():
            break
        plan = json.loads((raw / "EXECUTION_PLAN.json").read_text())["stages"]
        status = json.loads((raw / "STAGE_STATUS.json").read_text())
        if all(status.get(stage, {}).get("status") == "completed" for stage in plan):
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
