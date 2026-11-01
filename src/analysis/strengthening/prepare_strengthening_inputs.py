"""Link the fresh raw-input outputs to the strengthening analyses."""

from common import *
import shutil
import pandas as pd
import numpy as np
import gzip
from src.analysis.validation.personal_range_validation import load_ashrae


def write_gzip_csv_if_changed(frame, path):
    payload = frame.to_csv(index=False).encode("utf-8")
    if path.is_file() and gzip.decompress(path.read_bytes()) == payload:
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as archive:
            archive.write(payload)
    temporary.replace(path)


def main():
    for name in ["audit/status", "config", "logs", "results"]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    project = {
        "code": str(CODE),
        "raw_run": str(RAW_RUN),
        "output": str(ROOT),
        "simulation_root": str(ROOT / "simulations"),
        "confirmation_workers": int(__import__("os").environ.get("HEM_WORKERS", "6")),
    }
    if (ROOT / "config/PROJECT.json").is_file():
        old = json.loads((ROOT / "config/PROJECT.json").read_text())
        if old.get("raw_run") != str(RAW_RUN):
            raise ValueError("Use a separate output directory for different raw runs")
    write_json(ROOT / "config/PROJECT.json", project)
    (ROOT / "simulations").mkdir(exist_ok=True)
    for source in (CODE / "data/metadata/strengthening_protocol").glob("*.yaml"):
        shutil.copy2(source, ROOT / source.name)
    for source_name, dest_name in [
        ("global_final", "annual"),
        ("spatial_final", "spatial"),
        ("multi_event_dr_final", "response"),
    ]:
        source = RESULTS / source_name
        target = ROOT / "results/base" / dest_name
        target.mkdir(parents=True, exist_ok=True)
        if not source.is_dir():
            continue
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path, target / path.name)
    response = ROOT / "results/base/response/EVENT_RESPONSE_SCORES.csv"
    if response.is_file():
        scores = pd.read_csv(response)
        scores = scores[np.isclose(scores.event_service_limit_factor, 1.0)]
        if len(scores) != 6480:
            raise ValueError("Fresh strict response design is incomplete")
        scores.to_csv(response, index=False)
    for task, path in [
        ("original_annual_reproduction", ROOT / "results/base/annual/GLOBAL_RUN_METRICS.csv"),
        ("original_response_reproduction", response),
    ]:
        record_status(
            task,
            "completed" if path.is_file() else "not_completed",
            {
                "scope": "Fresh raw-input base analysis; historical final tables are not generator inputs.",
                "available": path.is_file(),
            },
        )
    links = []
    for archive in ["ASHRAE", "CTCD"]:
        frame = pd.read_csv(
            DEMAND_REBUILD / (archive + "_ASSIGNMENTS_LINKED.csv.gz"), low_memory=False
        )
        frame["archive"] = archive
        frame["person_key"] = (
            frame.source_group.astype(str) + "|" + frame.unit_group.astype(str)
            if archive == "ASHRAE"
            else "CTCD|" + frame.profile_id.astype(str)
        )
        frame["source_key"] = (
            frame.source_group.astype(str)
            if archive == "ASHRAE"
            else "CTCD|" + frame.contributor.astype(str)
        )
        links.append(frame)
    write_gzip_csv_if_changed(pd.concat(links, ignore_index=True),
        ROOT / "audit/RAW_REGENERATED_PROFILE_IDENTITY_LINKS.csv.gz")
    support = ROOT / "results/data_support"
    support.mkdir(exist_ok=True)
    ash = CODE / "data/upstream/ashrae_db_ii"
    known = load_ashrae(ash / "db_measurements_v2.1.0.csv.gz", ash / "db_metadata.csv")
    write_gzip_csv_if_changed(known, support / "IDENTIFIED_AS_HRAE_RECORDS.csv.gz")
    climates = json.loads((GLOBAL / "GLOBAL_CLIMATES.json").read_text())["climates"]
    weather = []
    for climate in climates:
        frame = pd.read_csv(
            CODE / "data/upstream/global_climate_matrix/weather" / climate["weather"],
            skiprows=8, header=None,
        )
        temperature = pd.to_numeric(frame[6]).to_numpy()
        weather.append(
            {
                "climate": climate["climate"],
                "heating": np.maximum(18 - temperature, 0).sum(),
                "cooling": np.maximum(temperature - 24, 0).sum(),
            }
        )
    weather = pd.DataFrame(weather)
    weather["combined"] = weather.heating + weather.cooling
    cold = weather.sort_values(["heating", "climate"], ascending=[False, True]).iloc[0].climate
    hot = (
        weather[weather.climate.ne(cold)]
        .sort_values(["cooling", "climate"], ascending=[False, True])
        .iloc[0]
        .climate
    )
    mild = (
        weather[~weather.climate.isin([cold, hot])]
        .sort_values(["combined", "climate"])
        .iloc[0]
        .climate
    )
    write_json(
        ROOT / "config/PILOT_SETTINGS.json",
        {
            "climates": [cold, hot, mild],
            "compositions": [2026102001, 2026102501],
            "selection_uses": "Weather only, before outcome inspection",
        },
    )
    write_json(
        ROOT / "audit/INITIALIZATION.json",
        {
            "status": "completed",
            "raw_run": str(RAW_RUN),
            "raw_assignments_sha256": digest(DEMAND_REBUILD / "PROFILE_REBUILD_AUDIT.json"),
            "historical_archive_inputs": [],
        },
    )


if __name__ == "__main__":
    main()
