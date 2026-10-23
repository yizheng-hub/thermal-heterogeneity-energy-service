"""Maintainer export of minimized controller inputs from archived probe runs.

This exporter reads source files but never imports historical project modules.
Released calculations use only the resulting relative-path bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def column(frame, zone, fragment):
    matches = [c for c in frame if c.startswith(zone.upper() + ":") and fragment in c]
    if len(matches) != 1:
        raise ValueError(f"Expected one {zone} {fragment} probe column")
    return frame[matches[0]].to_numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    stage, project, archive, out = (args.stage_root.resolve(), args.project_root.resolve(),
                                     args.archive_root.resolve(), args.output_root.resolve())
    controller = out / "controller"
    controller.mkdir(parents=True, exist_ok=False)
    contexts = json.loads((stage / "engineering/MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(stage / "engineering/ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    climate = archive / "results/three_control_main/climate_service_uncertainty_20261020_v1"
    extension = archive / "results/three_control_main/frozen_control_extension_20261020_v1"
    known_path = climate / "ALL_EXACT_NATIVE_ENGINEERING_QA.json"
    known = {row["folder"]: row for row in json.loads(known_path.read_text(encoding="utf-8"))}
    half_path = archive / ("results/g1c_full_archive_virtual_ltpref/"
                           "full_21418_19q_griffiths_anchored_v4/"
                           "full_archive_anchored_executable_intervals.csv.gz")
    halfwidths = pd.read_csv(half_path, usecols=["record_id", "ltpref_symmetric_half_width_c"])
    halfwidths = halfwidths.set_index("record_id").ltpref_symmetric_half_width_c
    portable, provenance = {}, []

    def remap(value):
        path = Path(value)
        return archive / path.relative_to(project) if path.is_relative_to(project) else path

    for city, cfg in contexts.items():
        original = pd.read_csv(cfg["native_assignment_path"], low_memory=False)
        zones = original.zone.unique().tolist()
        profile = original.rename(columns={"seed": "composition_id", "allocation": "layout",
                                             "zone": "zone_id", "reference_center_g050_c": "centre_c"})
        centres = original.reference_center_g050_c.to_numpy()
        half = halfwidths.loc[original.record_id].to_numpy()
        low, high = centres - half, centres + half
        valid = (low <= 30) & (high >= 16)
        profile["controller_lower_c"] = np.where(valid, np.maximum(low, 16), low)
        profile["controller_upper_c"] = np.where(valid, np.minimum(high, 30), high)
        profile["profile_id"] = original.virtual_profile_id.astype(str).map(
            lambda v: "sp_" + hashlib.sha256(v.encode()).hexdigest()[:24])
        fields = ["composition_id", "layout", "profile_id", "zone_id", "centre_c",
                  "controller_lower_c", "controller_upper_c"]
        np.savez_compressed(controller / f"{city}_profiles.npz", **{
            field: profile[field].to_numpy(dtype=str if profile[field].dtype == object else None)
            for field in fields})
        if city == "Miami":
            sweep = archive / "results/three_control_main/office_v6_equal_service_20261020_v1/SWEEP_INPUTS_BEFORE_RUNS.json"
            probes = [{"uniform_target_c": d["target_c"], "folder": d["folder"]}
                      for d in json.loads(sweep.read_text(encoding="utf-8"))]
            probes += json.loads((extension / "fine_uniform/DESIGNS_BEFORE_RUNS.json").read_text(encoding="utf-8"))
        else:
            probes = json.loads((climate / city / "uniform_probes/DESIGNS_BEFORE_RUNS.json").read_text(encoding="utf-8"))
        probes = sorted({float(d["uniform_target_c"]): d for d in probes}.values(),
                        key=lambda d: d["uniform_target_c"])
        probes = [d for d in probes if known[d["folder"]]["engineering_pass"]]
        anchor = manifest[manifest.context.eq(city) & manifest.variant.eq(3)].iloc[0]
        anchor_path = Path(anchor.path_hourly)
        if not anchor_path.is_absolute():
            anchor_path = stage / "engineering" / anchor_path
        with np.load(anchor_path, allow_pickle=False) as saved:
            occupied = saved["occupied"].astype(bool)
        temperatures, occupancies, energies = [], [], []
        for probe in probes:
            csv_path = remap(probe["folder"]) / "eplusout.csv"
            frame = pd.read_csv(csv_path, usecols=lambda c: "Operative Temperature" in c or "Occupant Count" in c)
            temperature = np.stack([column(frame, z, "Operative Temperature")[occupied] for z in zones], axis=1)
            people = np.stack([column(frame, z, "Occupant Count")[occupied] for z in zones], axis=1)
            temperatures.append(temperature)
            occupancies.append(people)
            energies.append(known[probe["folder"]]["controlled_hvac_electric_kwh"]
                            + known[probe["folder"]]["space_heating_gas_kwh_fuel"])
            provenance.append({"context": city, "uniform_target_c": probe["uniform_target_c"],
                               "csv_sha256": sha256(csv_path), "engineering_qualified": True})
        if any(not np.array_equal(occupancies[0], p) for p in occupancies[1:]):
            raise ValueError(f"Native probe occupancy differs in {city}")
        np.savez_compressed(controller / f"{city}_probes.npz",
                            uniform_target_c=np.array([d["uniform_target_c"] for d in probes]),
                            operative_c=np.array(temperatures), native_people=occupancies[0],
                            purchased_site_kwh=np.array(energies), zone_ids=np.array(zones),
                            occupied_source_indices=np.flatnonzero(occupied))
        portable[city] = {"zones": zones, "floor_groups": cfg["floor_groups"],
                          "floor_binding_equals_building": cfg["floor_binding_equals_building"],
                          "half_deadband_c": cfg["half_deadband_c"],
                          "floor_area_m2": cfg["floor_area_m2"],
                          "probes": f"controller/{city}_probes.npz",
                          "profiles": f"controller/{city}_profiles.npz"}
        print(json.dumps({"context_exported": city, "qualified_probes": len(probes),
                          "occupied_hours": int(occupied.sum()), "zones": len(zones)}), flush=True)
    (controller / "CONTEXTS.json").write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")
    (controller / "PROBE_PROVENANCE.json").write_text(json.dumps({
        "source_engineering_qa_sha256": sha256(known_path),
        "source_halfwidth_table_sha256": sha256(half_path), "probes": provenance,
        "raw_personal_identifiers_exported": False,
        "controller_bounds": "Exact source halfwidth recomputation and16-30 clipping, before CSV rounding",
        "temperature": "degrees Celsius; operative, native occupied probe hours",
        "energy_proxy": "Qualified probe heating/cooling/fans electricity plus gas HHV kWh",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
