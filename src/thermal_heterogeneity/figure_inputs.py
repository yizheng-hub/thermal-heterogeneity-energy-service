"""Export the exact retained figure/table inputs into the portable results/figure_data folder."""

from pathlib import Path
import argparse
import hashlib
import json
import shutil
import gzip
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floor_geometry(source):
    floors = {}
    for block in re.sub(r"!.*", "", source.read_text()).split(";"):
        obj = [value.strip() for value in block.split(",")]
        if len(obj) > 12 and obj[0].lower() == "buildingsurface:detailed" and obj[2].lower() == "floor":
            coordinates = list(map(float, obj[12:]))
            floors[obj[4]] = [coordinates[i:i + 3] for i in range(0, len(coordinates), 3)]
    if not floors:
        raise ValueError("No floor geometry in the illustrated model")
    return {"floors": floors}


def minimize_human_inputs(name, frame):
    """Keep only the anonymous quantities used by publication plots."""
    if name in {"ASHRAE_RECORDS.csv.gz", "CTCD_RECORDS.csv.gz"}:
        return frame[["reference_g050_c"]].copy(), "unlinked demand-center values only"
    if name == "IDENTIFIED_AS_HRAE_RECORDS.csv.gz":
        support = (
            frame.groupby("person")
            .agg(
                records=("temperature", "size"),
                lo=("temperature", "min"),
                hi=("temperature", "max"),
            )
            .reset_index(drop=True)
        )
        return support, "unlinked per-person count and temperature extent; no identity or times"
    if name == "DEMAND_COMPOSITION_PROFILES.csv":
        return frame[
            ["context", "composition_id", "centre_c", "range_width_c"]
        ].copy(), "unlinked simulated demand values"
    if name == "CONSTRAINED_PLACEMENT_PROFILES.csv":
        keep = ["context", "composition_id", "layout", "zone_id", "profile_id", "centre_c"]
        result = frame[keep].copy()
        keys = sorted(result.profile_id.unique())
        result["profile_id"] = result.profile_id.map(
            {key: f"profile{n:06d}" for n, key in enumerate(keys)}
        )
        return result, "anonymous simulation profile indices and plotted placement fields"
    return frame, None


def export(publication):
    destination = ROOT / "results/figure_data"
    destination.mkdir(parents=True, exist_ok=True)
    expected = {}
    # Publication notebooks already read the minimized, verified results/figure_data files.
    # Keep their original provenance when refreshing the publication data bundle.
    retained = {}
    index = destination / "index.json"
    indexed = {row["file"]: row["sha256"] for row in verify()["files"]} if index.is_file() else {}
    receipts = []
    for folder, pattern, count in [
        ("main", "Figure_*.json", 6),
        ("extended", "Figure_E*.json", 4),
        ("supplementary", "Figure_S*.json", 7),
    ]:
        section = sorted((publication / folder).glob(pattern))
        if len(section) != count:
            raise ValueError(f"Expected {count} complete {folder} figures")
        receipts.extend(section)
    if index.is_file():
        for receipt in [*receipts, publication / "tables/TABLES.json"]:
            for path, digest in json.loads(receipt.read_text(encoding="utf-8"))["input_sha256"].items():
                if indexed.get(path.replace("\\", "/")) != digest:
                    raise ValueError(f"Source differs from {receipt.name}: {path}")
        return
    for receipt in receipts:
        for name, digest in json.loads(receipt.read_text(encoding="utf-8"))["input_sha256"].items():
            if name.startswith("results/figure_data/"):
                if not retained:
                    retained = {row["result"]: row for row in verify()["files"]}
                if name not in retained or retained[name]["sha256"] != digest:
                    raise ValueError(f"Unregistered results/figure_data input: {name}")
                record = retained[name]
                name, digest = record["source"], record["source_sha256"]
            elif "results/figure_data/" + name in retained:
                record = retained["results/figure_data/" + name]
                if digest == record["sha256"]:
                    digest = record["source_sha256"]
            if name in expected and expected[name] != digest:
                raise ValueError(f"Conflicting source versions: {name}")
            expected[name] = digest
    tables = json.loads((publication / "tables/TABLES.json").read_text(encoding="utf-8"))
    for name, digest in tables["input_sha256"].items():
        if name.startswith("results/figure_data/"):
            if name not in retained or retained[name]["sha256"] != digest:
                raise ValueError(f"Unregistered results/figure_data table input: {name}")
            record = retained[name]
            name, digest = record["source"], record["source_sha256"]
        elif "results/figure_data/" + name in retained and digest == retained["results/figure_data/" + name]["sha256"]:
            digest = retained["results/figure_data/" + name]["source_sha256"]
        if name in expected and expected[name] != digest:
            raise ValueError(f"Figure/table source mismatch: {name}")
        expected[name] = digest
    # Completion receipts are consulted before selecting a diagnostic table.
    for name in list(expected):
        path = ROOT / name
        if path.parent.name.startswith("run_") and (path.parent / "STATUS.json").exists():
            status = path.parent / "STATUS.json"
            expected[status.relative_to(ROOT).as_posix()] = sha(status)
    records = []
    retained_by_source = {row["source"]: row for row in retained.values()}
    for name, digest in sorted(expected.items()):
        existing = retained_by_source.get(name)
        if existing is not None and existing["source_sha256"] == digest:
            records.append(existing)
            continue
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or sha(source) != digest:
            raise ValueError(f"Source changed or missing: {name}")
        target = destination / name
        if source.name.endswith(".csv.gz"):
            target = target.with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        transformation = "byte-identical"
        if source.name.endswith(".csv.gz"):
            target.write_bytes(gzip.decompress(source.read_bytes()))
            transformation = "gzip container removed without changing CSV text"
        elif source.suffix.lower() != ".idf":
            shutil.copy2(source, target)
        if source.name in {"NORMAL_OPERATION_REPLAY_COMPLETED.csv", "ACTION_FULL_COMPLETED.csv"}:
            frame = pd.read_csv(source)
            keep = frame.hourly_path.map(
                lambda value: Path(value).is_relative_to(ROOT)
                and Path(value).relative_to(ROOT).as_posix() in expected
            )
            frame = frame[keep].copy()
            if frame.empty:
                raise ValueError("Missing illustrated response trajectories")
            frame["hourly_path"] = frame.hourly_path.map(
                lambda value: "results/figure_data/" + Path(value).relative_to(ROOT).as_posix()
            )
            frame.to_csv(target, index=False)
            transformation = "illustrated trajectory rows; project-relative hourly paths"
        elif source.name == "MODEL_MANIFEST.json":
            model = json.loads(source.read_text())
            path = Path(model["Miami"]["prepared_idf"])
            relative = path.relative_to(ROOT).as_posix()
            if relative not in expected:
                raise ValueError("Illustrated physical geometry was not recorded")
            model = {
                "Miami": {
                    "floor_geometry": "results/figure_data/"
                    + str(Path(relative).with_suffix(".geometry.json")).replace("\\", "/")
                }
            }
            target.write_text(json.dumps(model, indent=2) + "\n")
            transformation = "illustrated floor-geometry link only; no executable model"
        if source.suffix.lower() == ".idf":
            # Export factual floor vertices for the diagram, not the third-party model.
            target = target.with_suffix(".geometry.json")
            target.write_text(json.dumps(floor_geometry(source), indent=2))
            transformation = "factual floor vertices only; executable third-party model excluded"
        if source.name.endswith((".csv", ".csv.gz")):
            frame = pd.read_csv(target, low_memory=False)
            frame, minimized = minimize_human_inputs(source.name, frame)
            if minimized:
                transformation = minimized
            # Only the three illustrated trajectory links are executable inputs here.
            unused_paths = [
                column
                for column in frame.columns
                if (column.endswith("_path") or "_path_" in column)
                and not (
                    column == "hourly_path"
                    and source.name
                    in {"NORMAL_OPERATION_REPLAY_COMPLETED.csv", "ACTION_FULL_COMPLETED.csv"}
                )
            ]
            if unused_paths:
                frame = frame.drop(columns=unused_paths)
                transformation += "; unused local provenance paths removed"
            if minimized or unused_paths:
                frame.to_csv(target, index=False)
        records.append(
            {
                "source": name,
                "source_sha256": digest,
                "result": target.relative_to(ROOT).as_posix(),
                "sha256": sha(target),
                "bytes": target.stat().st_size,
                "transformation": transformation,
            }
        )
    if retained:
        retained.update({row["result"]: row for row in records})
        records = sorted(retained.values(), key=lambda row: row["result"])
    manifest = {
        "status": "exported_from_verified_figure_inputs",
        "files": records,
        "scientific_values_changed": False,
        "purpose": "Figure and table reproduction; full scientific runs remain under output.",
    }
    (destination / "FIGURE_DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"files": len(records), "bytes": sum(r["bytes"] for r in records)}))


def verify():
    for folder in [ROOT / "results/figure_data", ROOT / "Source_Data", ROOT / "data/figure_data"]:
        index = folder / "index.json"
        if index.is_file():
            records = json.loads(index.read_text(encoding="utf-8"))
            for row in records:
                path = (ROOT / row["file"]).resolve()
                within_inputs = path.is_relative_to(folder.resolve())
                if folder == ROOT / "results/figure_data":
                    within_inputs |= path.is_relative_to((ROOT / "results/simulation_run").resolve())
                if not within_inputs or not path.is_file() or sha(path) != row["sha256"]:
                    raise ValueError("Source data changed or missing: " + row["file"])
            return {"files": records}
    manifest = json.loads((ROOT / "results/figure_data/FIGURE_DATA_MANIFEST.json").read_text())
    for row in manifest["files"]:
        path = (ROOT / row["result"]).resolve()
        if (
            not path.is_relative_to(ROOT / "results/figure_data")
            or not path.is_file()
            or sha(path) != row["sha256"]
        ):
            raise ValueError(f"results/figure_data input changed or missing: {row['result']}")
    return manifest


def resolve(request):
    """Locate a numerical input in a figure-data index."""
    def key(value):
        value = str(value).replace("\\", "/").split("results/figure_data/")[-1]
        value = value.replace("output/from_raw/", "baseline/").replace("output/strengthening/", "analysis/")
        return re.sub(r"run_[^/]+/", "", value)
    records = verify()["files"]
    exact = [row for row in records if key(row.get("key", row.get("result", ""))) == key(request)]
    matches = exact or [row for row in records if Path(row.get("key", row.get("result", ""))).name == Path(request).name]
    if len(matches) != 1:
        raise FileNotFoundError("Expected exactly one dataset for " + str(request))
    return ROOT / matches[0].get("file", matches[0].get("result"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, default=ROOT / "output/publication")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps({"verified_files": len(verify()["files"])}))
    else:
        export(args.publication_root)
