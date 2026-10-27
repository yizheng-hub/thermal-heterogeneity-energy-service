"""Plan and execute exact reruns for missing facility-energy SQLite outputs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .enrich_facility_energy import _run_folder, sha256
    from .global_matrix import run_energyplus
except ImportError:
    from enrich_facility_energy import _run_folder, sha256  # type: ignore[no-redef]
    from global_matrix import run_energyplus  # type: ignore[no-redef]


def _sql_index(roots: tuple[Path, ...]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("eplusout.sql"):
            run_id = path.parent.name
            result.setdefault(run_id, []).append(path)
    return result


def build_recovery_plan(
    manifest_path: Path,
    *,
    model_manifest_path: Path | None = None,
    sql_search_roots: tuple[Path, ...] = (),
) -> pd.DataFrame:
    """Locate exact SQL artifacts and identify the runs that require rerunning."""
    manifest_path = manifest_path.resolve()
    frame = pd.read_csv(manifest_path)
    required = {"run_id", "climate", "hourly_path", "input_sha256"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Run manifest lacks {sorted(missing)}")
    models = {}
    if model_manifest_path is not None:
        models = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    alternate = _sql_index(sql_search_roots)
    rows: list[dict] = []
    for _, run in frame.iterrows():
        run_id = str(run.run_id)
        primary = _run_folder(run, manifest_path) / "eplusout.sql"
        if primary.is_file():
            status = "available_primary"
            sql_path: Path | None = primary
        else:
            sql_path = None
            for candidate in alternate.get(run_id, []):
                source_qa = candidate.parent / "RUN_QA.json"
                if not source_qa.is_file():
                    continue
                source = json.loads(source_qa.read_text(encoding="utf-8"))
                if (
                    source.get("completed") is True
                    and str(source.get("input_idf_sha256", ""))
                    == str(run.input_sha256)
                ):
                    sql_path = candidate
                    break
            status = (
                "available_search_root" if sql_path is not None else "rerun_required"
            )

        input_path = (
            Path(str(run.input_idf_path))
            if "input_idf_path" in frame and pd.notna(run.get("input_idf_path"))
            else manifest_path.parent / "inputs" / f"{run_id}.idf"
        )
        if not input_path.is_file() or sha256(input_path) != str(run.input_sha256):
            raise ValueError(f"Input IDF is missing or changed for {run_id}")
        if "weather_path" in frame and pd.notna(run.get("weather_path")):
            weather_path = Path(str(run.weather_path))
        elif str(run.climate) in models:
            weather_path = Path(models[str(run.climate)]["weather_path"])
        else:
            raise ValueError(f"Weather path is unavailable for {run_id}")
        if not weather_path.is_file():
            raise FileNotFoundError(weather_path)

        qa_path = _run_folder(run, manifest_path) / "RUN_QA.json"
        if not qa_path.is_file():
            raise FileNotFoundError(qa_path)
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if str(qa.get("input_idf_sha256", "")) != str(run.input_sha256):
            raise ValueError(f"Run QA input hash differs for {run_id}")
        if sql_path is not None:
            source_qa = sql_path.parent / "RUN_QA.json"
            if not source_qa.is_file():
                raise FileNotFoundError(source_qa)
            source = json.loads(source_qa.read_text(encoding="utf-8"))
            if source.get("completed") is not True:
                raise ValueError(f"Recovered SQL source did not complete for {run_id}")
            if str(source.get("input_idf_sha256", "")) != str(run.input_sha256):
                raise ValueError(f"Recovered SQL input hash differs for {run_id}")
        rows.append({
            "run_id": run_id,
            "climate": str(run.climate),
            "recovery_status": status,
            "sql_path": "" if sql_path is None else str(sql_path.resolve()),
            "sql_sha256": "" if sql_path is None else sha256(sql_path),
            "input_idf_path": str(input_path.resolve()),
            "input_idf_sha256": sha256(input_path),
            "weather_path": str(weather_path.resolve()),
            "weather_sha256": sha256(weather_path),
            "runtime_sha256": str(qa.get("runtime_sha256", "")),
            "rejected_search_candidates": (
                len(alternate.get(run_id, []))
                if status == "rerun_required" else 0
            ),
            "legacy_hourly_path": str(Path(str(run.hourly_path)).resolve()),
            "legacy_hourly_sha256": str(run.get("hourly_sha256", "")),
        })
    return pd.DataFrame(rows).sort_values(
        ["recovery_status", "climate", "run_id"], kind="stable"
    ).reset_index(drop=True)


def _validate_repair(
    legacy_path: Path, repaired_path: Path, run_id: str
) -> dict[str, float]:
    with np.load(legacy_path, allow_pickle=False) as saved:
        legacy = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(repaired_path, allow_pickle=False) as saved:
        repaired = {name: np.asarray(saved[name]) for name in saved.files}
    exact = ("zone_ids", "occupied", "month", "day", "hour", "dst")
    for name in exact:
        if name in legacy and not np.array_equal(legacy[name], repaired[name]):
            raise ValueError(f"Repair changed {name} for {run_id}")
    compared = (
        "temperatures", "people", "heating_electric_kwh",
        "cooling_electric_kwh", "fans_electric_kwh", "gas_hhv_kwh",
    )
    maxima: dict[str, float] = {}
    for name in compared:
        if name not in legacy:
            continue
        difference = float(np.max(np.abs(legacy[name] - repaired[name])))
        maxima[f"max_abs_{name}_difference"] = difference
        if not np.allclose(legacy[name], repaired[name], rtol=1e-12, atol=1e-9):
            raise ValueError(f"Repair changed {name} for {run_id}: {difference}")
    return maxima


def execute_recovery_plan(
    plan: pd.DataFrame,
    runtime: Path,
    output_root: Path,
    *,
    workers: int = 4,
) -> pd.DataFrame:
    """Rerun only missing SQL cases and verify them against legacy compacts."""
    selected = plan[plan.recovery_status.eq("rerun_required")].copy()
    runtime = runtime.resolve()
    runtime_hash = sha256(runtime)
    if not selected.runtime_sha256.astype(str).eq(runtime_hash).all():
        raise ValueError("Repair runtime differs from the original EnergyPlus runtime")

    def execute(row: dict) -> dict:
        run_id = str(row["run_id"])
        legacy_path = Path(row["legacy_hourly_path"])
        with np.load(legacy_path, allow_pickle=False) as saved:
            zones = np.asarray(saved["zone_ids"]).astype(str).tolist()
        folder = output_root / "runs" / run_id
        compact = output_root / "hourly" / f"{run_id}.npz"
        qa_path = folder / "RUN_QA.json"
        if qa_path.is_file() and compact.is_file() and (folder / "eplusout.sql").is_file():
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
            if (
                qa.get("completed")
                and qa.get("hourly_sha256") == sha256(compact)
                and qa.get("input_idf_sha256") == row["input_idf_sha256"]
                and qa.get("weather_sha256") == row["weather_sha256"]
                and qa.get("runtime_sha256") == runtime_hash
            ):
                return row | qa | _validate_repair(legacy_path, compact, run_id)
        qa = run_energyplus(
            Path(row["input_idf_path"]), Path(row["weather_path"]), runtime,
            folder, compact, zones, cleanup_raw=False,
        )
        if not qa.get("completed"):
            raise RuntimeError(f"Facility SQL repair failed for {run_id}")
        return row | qa | _validate_repair(legacy_path, compact, run_id)

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(execute, row) for row in selected.to_dict("records")]
        for future in as_completed(pending):
            rows.append(future.result())
    result = pd.DataFrame(rows).sort_values(["climate", "run_id"], kind="stable")
    output_root.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_root / "FACILITY_SQL_RERUN_MANIFEST.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--manifest", type=Path, required=True)
    plan_parser.add_argument("--model-manifest", type=Path)
    plan_parser.add_argument("--sql-search-root", action="append", type=Path, default=[])
    plan_parser.add_argument("--output", type=Path, required=True)
    execute_parser = sub.add_parser("execute")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--runtime", type=Path, required=True)
    execute_parser.add_argument("--output-root", type=Path, required=True)
    execute_parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "plan":
        plan = build_recovery_plan(
            args.manifest,
            model_manifest_path=args.model_manifest,
            sql_search_roots=tuple(args.sql_search_root),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(args.output, index=False)
        print(plan.recovery_status.value_counts().to_string())
    else:
        result = execute_recovery_plan(
            pd.read_csv(args.plan), args.runtime, args.output_root,
            workers=args.workers,
        )
        print(json.dumps({"status": "PASS", "reruns": len(result)}, indent=2))


if __name__ == "__main__":
    main()
