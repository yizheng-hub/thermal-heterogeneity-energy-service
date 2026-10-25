"""Compare archived official model objects with frozen four-context inputs.

This is a local, read-only input audit. It neither downloads source assets nor
changes a model. Original EnergyPlus examples and DOE ZIP assets are supplied
separately under their original terms. Differences are calculated, not inferred
from the model's successful execution or from a stored PASS flag.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np


PHYSICS = {
    "building", "globalgeometryrules", "zone", "space", "buildingsurface:detailed",
    "fenestrationsurface:detailed", "internalmass", "shading:building:detailed",
    "shading:zone:detailed", "heatbalancealgorithm", "surfacetemperaturealgorithm",
    "surfaceconvectionalgorithm:inside", "surfaceconvectionalgorithm:outside",
}


def parse_idf(text):
    objects = collections.defaultdict(list)
    clean = re.sub(r"!.*", "", text)
    for block in clean.split(";"):
        if block.strip():
            fields = [value.strip() for value in block.split(",")]
            objects[fields[0].lower()].append(fields[1:])
    return dict(objects)


def category(kind):
    if kind in PHYSICS or kind.startswith(("material", "construction", "windowmaterial")):
        return "geometry_envelope_mass"
    if kind.startswith(("zoneinfiltration", "zonemixing", "zonecrossmixing", "airflownetwork")):
        return "infiltration_interzone_airflow"
    if kind.startswith(("site:", "sizingperiod:")):
        return "location_design_climate"
    if kind.startswith("energymanagementsystem:"):
        return "EMS_execution_requires_interpretation"
    if kind.startswith("output:"):
        return "output_instrumentation"
    if kind in {"schedule:compact", "thermostatsetpoint:dualsetpoint"}:
        return "schedule_or_thermostat_control"
    if kind in {"runperiod", "simulationcontrol"}:
        return "simulation_period_configuration"
    return "other_model_object"


def object_differences(source, frozen):
    rows = []
    for kind in sorted(set(source) | set(frozen)):
        old, new = source.get(kind, []), frozen.get(kind, [])
        old_count, new_count = collections.Counter(map(tuple, old)), collections.Counter(map(tuple, new))
        removed, added = list((old_count - new_count).elements()), list((new_count - old_count).elements())
        old_names, new_names = collections.Counter(row[0] for row in removed), collections.Counter(row[0] for row in added)
        paired = set()
        for row in removed:
            matches = [candidate for candidate in added if candidate[0] == row[0]]
            if old_names[row[0]] == new_names[row[0]] == 1:
                other = matches[0]
                changed = [i for i in range(max(len(row), len(other)))
                           if (row[i] if i < len(row) else None) != (other[i] if i < len(other) else None)]
                rows.append({"object_type": kind, "object_name": row[0], "difference": "changed",
                             "category": category(kind), "field_indices_zero_based": changed,
                             "source_fields": list(row), "frozen_fields": list(other)})
                paired.add(row[0])
            else:
                rows.append({"object_type": kind, "object_name": row[0], "difference": "removed",
                             "category": category(kind), "field_indices_zero_based": [],
                             "source_fields": list(row), "frozen_fields": []})
        for row in added:
            if row[0] not in paired:
                rows.append({"object_type": kind, "object_name": row[0], "difference": "added",
                             "category": category(kind), "field_indices_zero_based": [],
                             "source_fields": [], "frozen_fields": list(row)})
    return rows


def polygon_area(fields, start):
    points = np.asarray(list(map(float, fields[start:])), dtype=float).reshape(-1, 3)
    return float(np.linalg.norm(np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)) / 2)


def physics_summary(objects, conditioned):
    surfaces = objects.get("buildingsurface:detailed", [])
    windows = objects.get("fenestrationsurface:detailed", [])
    area = {row[0]: polygon_area(row, 11) for row in surfaces}
    lookup = {row[0]: row for row in surfaces}
    pairs, unresolved, zone_links = [], [], []
    zone_names = {row[0] for row in objects.get("zone", [])}
    for row in surfaces:
        if row[5].lower() == "surface":
            if row[6] not in lookup:
                unresolved.append(row[0])
            else:
                other = lookup[row[6]]
                pairs.append({"surface": row[0], "neighbor_surface": row[6],
                              "reciprocal": other[6] == row[0],
                              "absolute_area_difference_m2": abs(area[row[0]] - area[other[0]])})
        elif row[5].lower() == "zone":
            zone_links.append({"surface": row[0], "zone": row[3], "neighbor_zone": row[6],
                               "neighbor_exists": row[6] in zone_names,
                               "area_m2": area[row[0]], "construction": row[2]})
    materials = {}
    for row in objects.get("material", []):
        thickness, conductivity, density, heat = map(float, row[2:6])
        materials[row[0]] = {"type": "Material", "thickness_m": thickness,
                              "conductivity_W_m_K": conductivity, "density_kg_m3": density,
                              "specific_heat_J_kg_K": heat,
                              "areal_heat_capacity_J_m2_K": thickness * density * heat,
                              "resistance_m2_K_W": thickness / conductivity,
                              "positive_physical_fields": min(thickness, conductivity, density, heat) > 0}
    for row in objects.get("material:nomass", []):
        materials[row[0]] = {"type": "Material:NoMass", "resistance_m2_K_W": float(row[2]),
                              "areal_heat_capacity_J_m2_K": 0.0}
    for row in objects.get("material:airgap", []):
        materials[row[0]] = {"type": "Material:AirGap", "resistance_m2_K_W": float(row[1]),
                              "areal_heat_capacity_J_m2_K": 0.0}
    constructions = {row[0]: row[1:] for row in objects.get("construction", [])}
    used = sorted({row[2] for row in surfaces} | {row[1] for row in objects.get("internalmass", [])})
    layers = []
    for name in used:
        names = constructions.get(name, [])
        known = bool(names) and all(layer in materials for layer in names)
        layers.append({"construction": name, "layers_outside_to_inside": names,
                       "solid_material_areal_heat_capacity_J_m2_K": sum(materials[layer]["areal_heat_capacity_J_m2_K"] for layer in names) if known else None,
                       "layer_resistance_excluding_surface_films_m2_K_W": sum(materials[layer]["resistance_m2_K_W"] for layer in names) if known else None,
                       "definition_type": "Construction" if names else "nonlayered_foundation_or_other"})
    gross_wall = sum(area[row[0]] for row in surfaces if row[1].lower() == "wall" and row[5].lower() == "outdoors")
    window_area = sum(polygon_area(row, 9) * float(row[7] or 1) for row in windows if row[1].lower() == "window")
    floor = {zone: sum(area[row[0]] for row in surfaces if row[3] == zone and row[1].lower() == "floor") for zone in conditioned}
    return {
        "zone_count_including_unconditioned": len(objects.get("zone", [])),
        "controlled_thermal_zone_count": len(conditioned), "conditioned_floor_area_m2_from_polygons": sum(floor.values()),
        "conditioned_zone_floor_areas_m2": floor,
        "north_axis_deg": float(objects["building"][0][1]),
        "opaque_surface_count": len(surfaces), "fenestration_count_including_doors": len(windows),
        "outside_boundary_condition_counts": dict(collections.Counter(row[5] for row in surfaces)),
        "gross_exterior_wall_area_m2": gross_wall, "window_area_m2": window_area,
        "window_to_gross_wall_ratio": window_area / gross_wall,
        "explicit_interzone_surface_pairs": pairs, "unresolved_surface_references": unresolved,
        "native_zone_boundary_links": zone_links,
        "unresolved_zone_boundary_references": [row for row in zone_links if not row["neighbor_exists"]],
        "explicit_zone_mixing_objects": len(objects.get("zonemixing", [])) + len(objects.get("zonecrossmixing", [])),
        "airflow_interpretation": "Interzone conduction/radiation use native adjacent boundaries; zero ZoneMixing objects does not remove interzone heat transfer. No added doorway-airflow model is inferred.",
        "materials": materials, "used_construction_layers": layers,
        "internal_mass_objects": objects.get("internalmass", []),
        "heat_balance_algorithm": objects.get("heatbalancealgorithm", []),
        "heat_capacity_interpretation": "Computed layer areal heat capacities are material inventory properties, not a fitted lumped building capacitance or a validation of transient temperatures.",
        "simple_glazing_SI_fields": objects.get("windowmaterial:simpleglazingsystem", []),
        "infiltration_design_fields": objects.get("zoneinfiltration:designflowrate", []),
        "infiltration_units": "Flow/ExteriorWallArea design coefficient is m3/s per m2 exterior wall; A/B/C/D multiply schedule and wind/temperature terms, not air changes per hour.",
        "infiltration_schedule_objects": [row for row in objects.get("schedule:compact", [])
                                            if row[0] in {entry[2] for entry in objects.get("zoneinfiltration:designflowrate", [])}],
        "ground_temperature_and_foundation": {key: value for key, value in objects.items()
                                               if key.startswith(("site:groundtemperature", "construction:ffactor", "construction:cfactor"))},
    }


def weather_summary(path, eio):
    with path.open(encoding="utf-8", errors="replace") as stream:
        lines = stream.readlines()
    rows = list(csv.reader(lines[8:]))
    header = next(csv.reader([lines[0]]))
    actual = [line.strip() for line in eio.read_text(encoding="utf-8", errors="replace").splitlines()
              if line.startswith("Site:Location,")]
    return {"epw_location_fields": header, "weather_records": len(rows),
            "source_years": sorted({int(row[0]) for row in rows}),
            "hour_endpoints": sorted({int(row[3]) for row in rows}),
            "drybulb_min_max_C": [min(float(row[6]) for row in rows), max(float(row[6]) for row in rows)],
            "actual_runtime_location_EIO": actual}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def standby_evidence(source, frozen, baseline_csv):
    """Trace the shipped small-example standby without inventing new settings."""
    source_schedules = {row[0]: row for row in source["schedule:compact"]}
    frozen_schedules = {row[0]: row for row in frozen["schedule:compact"]}
    heat = source_schedules["HTGSETP_SCH_NO_OPTIMUM_w_SB"]
    cool = source_schedules["CLGSETP_SCH_NO_OPTIMUM_w_SB"]
    def numeric(row):
        values = []
        for token in row[2:]:
            try:
                values.append(float(token))
            except ValueError:
                pass
        return sorted(set(values))
    standby_heat = [value for value in numeric(heat) if 15.56 < value < 21.11]
    standby_cool = [value for value in numeric(cool) if 23.89 < value < 29.44]
    if len(standby_heat) != 1 or len(standby_cool) != 1:
        raise ValueError("The archived small-example standby source changed")
    rows, people_nonzero = 0, 0
    with baseline_csv.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        keys = {name: next(col for col in reader.fieldnames if "PERIMETER_ZN_2:" in col
                          and name in col) for name in ["Occupant Count", "Heating Setpoint", "Cooling Setpoint"]}
        for row in reader:
            if (abs(float(row[keys["Heating Setpoint"]]) - standby_heat[0]) < 1e-6
                    and abs(float(row[keys["Cooling Setpoint"]]) - standby_cool[0]) < 1e-6):
                rows += 1
                people_nonzero += int(float(row[keys["Occupant Count"]]) > 1e-9)
    return {"source_standby_heating_c": standby_heat[0], "source_standby_cooling_c": standby_cool[0],
            "source_standby_schedules_unchanged": all(source_schedules[name] == frozen_schedules[name]
                for name in ["HTGSETP_SCH_NO_OPTIMUM_w_SB", "CLGSETP_SCH_NO_OPTIMUM_w_SB", "BLDG_OCC_SCH_w_SB"]),
            "baseline_hours_with_source_standby_thresholds": rows,
            "baseline_standby_hours_with_positive_native_occupancy": people_nonzero,
            "interpretation": "Shipped official EnergyPlus small-example zero-occupancy standby is retained; it is not an occupied target-plus/minus-gap command. Effects on later conditions remain part of the physical trajectories."}


def run(model_root, output_root):
    original = model_root / "provenance/original_sources"
    archive = original / "ASHRAE901_OfficeMedium_STD2019.zip"
    small_source = original / "ASHRAE901_OfficeSmall_STD2019_Denver.idf"
    source_sha = {archive.name: digest(archive), small_source.name: digest(small_source)}
    settings = json.loads((model_root / "ENGINEERING_CONTEXTS.json").read_text(encoding="utf-8"))
    locked = json.loads((model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    contexts, differences = {}, []
    with zipfile.ZipFile(archive) as stored:
        for context, setting in settings.items():
            frozen_path, epw = model_root / context / "baseline.idf", model_root / context / "weather.epw"
            frozen = parse_idf(frozen_path.read_text(encoding="utf-8", errors="replace"))
            if context in {"Miami", "NewDelhi"}:
                member = next(name for name in stored.namelist() if name.endswith(f"_{context}.idf"))
                raw = stored.read(member)
                source_locator = archive.name + "::" + member
                origin = "DOE/PNNL ASHRAE 90.1-2019 location-specific medium office"
            else:
                raw, source_locator = small_source.read_bytes(), small_source.name
                origin = "Official EnergyPlus v22.1 Denver small-office example transferred to the named weather"
            source = parse_idf(raw.decode("utf-8", errors="replace"))
            changes = object_differences(source, frozen)
            for row in changes:
                differences.append({"context": context} | row)
            changed_types = sorted({row["object_type"] for row in changes})
            contexts[context] = {
                "source_origin": origin, "original_idf_locator": source_locator,
                "original_idf_sha256": hashlib.sha256(raw).hexdigest(),
                "frozen_idf_sha256": digest(frozen_path), "epw_sha256": digest(epw),
                "matches_frozen_idf_contract": digest(frozen_path) == locked[context]["baseline_idf_sha256"],
                "matches_frozen_epw_contract": digest(epw) == locked[context]["weather_sha256"],
                "changed_object_types": changed_types,
                "changed_object_counts_by_category": dict(collections.Counter(row["category"] for row in changes)),
                "no_geometry_envelope_mass_changes": not any(row["category"] == "geometry_envelope_mass" for row in changes),
                "no_infiltration_object_changes": not any(row["category"] == "infiltration_interzone_airflow" for row in changes),
                "physics": physics_summary(frozen, setting["zones"]),
                "source_version": source.get("version"), "frozen_version": frozen.get("version"),
                "source_runperiods": source.get("runperiod"), "frozen_runperiods": frozen.get("runperiod"),
                "timestep_per_hour": frozen.get("timestep"), "simulation_control": frozen.get("simulationcontrol"),
                "source_site_location": source.get("site:location"), "frozen_site_location": frozen.get("site:location"),
                "source_design_days": source.get("sizingperiod:designday"),
                "frozen_design_days": frozen.get("sizingperiod:designday"),
                "weather": weather_summary(epw, model_root / "baseline_evidence" / context / "eplusout.eio"),
            }
            if context not in {"Miami", "NewDelhi"}:
                contexts[context]["standby"] = standby_evidence(source, frozen,
                    model_root / "baseline_evidence" / context / "eplusout.csv")
    output_root.mkdir(parents=True, exist_ok=True)
    fields = ["context", "object_type", "object_name", "difference", "category",
              "field_indices_zero_based", "source_fields", "frozen_fields"]
    with (output_root / "MODEL_OBJECT_DIFFERENCES.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in differences:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
                             for key, value in row.items()})
    report = {"audit": "local_official_source_to_frozen_model_foundation", "new_simulations": 0,
              "parameters_changed": False, "source_archive_sha256": source_sha, "contexts": contexts,
              "overall_scope": "Representative public prototypes; no assertion of site calibration. NewDelhi is the official location-specific DOE archive member, not a Miami-only weather replacement. Marine contexts retain Denver envelope and design days with local EPW annual weather.",
              "difference_interpretation": "Source identity verifies provenance, not every engineering choice. EMS, schedule and output differences remain explicit and require operational interpretation.",
              "sources": ["https://www.energycodes.gov/prototype-building-models",
                          "https://raw.githubusercontent.com/NatLabRockies/EnergyPlus/v22.1.0/testfiles/ASHRAE901_OfficeSmall_STD2019_Denver.idf",
                          "https://bigladdersoftware.com/epx/docs/22-1/input-output-reference/group-location-climate-weather-file-access.html",
                          "https://bigladdersoftware.com/epx/docs/22-1/input-output-reference/group-airflow.html"],
              "difference_table_sha256": digest(output_root / "MODEL_OBJECT_DIFFERENCES.csv")}
    (output_root / "MODEL_PHYSICS_FOUNDATION_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=root / "data/upstream/frozen_buildings")
    parser.add_argument("--output-root", type=Path, default=root / "output/model_foundation_audit")
    args = parser.parse_args()
    report = run(args.model_root.resolve(), args.output_root.resolve())
    print(json.dumps({context: {key: data[key] for key in ["changed_object_types", "no_geometry_envelope_mass_changes", "no_infiltration_object_changes"]}
                      for context, data in report["contexts"].items()}, indent=2))


if __name__ == "__main__":
    main()
