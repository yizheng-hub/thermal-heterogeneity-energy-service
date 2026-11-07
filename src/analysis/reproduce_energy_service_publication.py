"""Rebuild scientific comparisons, 31 figures and traceable source tables.

External processed inputs are verified before analysis. No manuscript, Word
template, raw survey or EnergyPlus runtime is required by this entry point.
No new physical simulation or building-parameter calibration occurs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_supporting_sources(previous, paper, tables, data, repo):
    """Export retained panel inputs and map current figures to their exact sources."""
    panel_root = previous / "panel_data"
    panel_target = tables / "Supporting_panel_inputs"
    panel_target.mkdir(exist_ok=True)
    for path in sorted(panel_root.glob("*.csv")):
        shutil.copyfile(path, panel_target / path.name)
    supporting = previous / "paper/supplementary_figures"
    diagnostics = tables / "Supporting_diagnostics"
    diagnostics.mkdir(exist_ok=True)
    for name in ("baseline_coordinate_summary.csv", "spatial_interpolation_city_holdout.csv",
                 "spatial_interpolation_support_masks.csv"):
        shutil.copyfile(supporting / name, diagnostics / name)

    with (previous / "paper/panel_mapping.csv").open(encoding="utf-8-sig", newline="") as stream:
        panel_mapping = list(csv.DictReader(stream))
    supporting_manifest = json.loads(
        (supporting / "SUPPLEMENTARY_FIGURE_MANIFEST.json").read_text(encoding="utf-8"))
    source_lists = {int(item["figure"][1:]): item["sources"] for item in supporting_manifest}
    # Historical panel IDs identify data inputs, not the current displayed figure number.
    for old_number, old_main in ((16, 2), (17, 3), (18, 6), (19, 5), (20, 4)):
        source_lists[old_number] = [item["old_frozen_panel"] + ".csv"
                                   for item in panel_mapping if item["new_figure"] == f"Fig{old_main}"]
    source_lists[21] = ["Figure_sources/Supplementary_Figure_S21_source.csv"]
    source_lists.update(seasonal_direct=["Seasonal/PERIOD_TOTALS.csv", "Seasonal/SELECTION_STATUS.csv"],
                        seasonal_common=["Seasonal/PERIOD_TOTALS.csv", "Seasonal/SELECTION_STATUS.csv"],
                        typical_day=["Typical_day_DR/THREE_DATE_COMPARISON.csv", "Typical_day_DR/EVENT_ALL_CANDIDATES.csv",
                                     "Typical_day_DR/EVENT_SELECTIONS.csv", "Typical_day_DR/EVENT_HOURLY.csv.gz"])

    def locate(source_name):
        if source_name.startswith("engineering/"):
            return data / "energy_service" / source_name
        if source_name.startswith("science_audit/"):
            return repo / "src/analysis/support" / source_name
        if source_name.startswith("spatial_method_audit/"):
            return repo / "src/analysis/support/spatial_audit" / source_name.split("/", 1)[1]
        if source_name == "natural_earth_110m_land":
            return repo / "src/thermal_heterogeneity/assets/ne_110m_land.geojson"
        if source_name == "MAIN_FIG6G_Q5_REGRET.csv":
            return previous / "decision_value" / source_name
        if source_name.startswith(("Seasonal/", "Typical_day_DR/")):
            return tables / source_name
        if source_name.startswith("Figure_sources/"):
            return tables / source_name
        if source_name.startswith("Fig") and "/" not in source_name:
            path = panel_target / source_name
            if path.is_file():
                return path
            # Frozen response traces use descriptive suffixes on their panel filenames.
            matches = list(panel_target.glob(source_name.removesuffix(".csv") + "_*.csv"))
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or missing supporting figure input: {source_name}")
            return matches[0]
        return data / "legacy_analysis" / source_name

    rows = []

    def add(figure, panel, key):
        path = locate(key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing current figure input: {key}")
        for base, kind in ((tables, "Source Data"), (data, "Computational data package"),
                           (repo, "Source repository"), (previous, "Generated supporting output")):
            if path.is_relative_to(base):
                locator = path.relative_to(base).as_posix()
                break
        else:
            raise ValueError(f"Unregistered current figure input: {key}")
        rows.append({"current_figure": figure, "panel": panel, "source_key": key,
                     "source_kind": kind, "relative_path": locator, "sha256": sha256(path),
                     "identifier_note": "Source panel IDs retain historical names; displayed labels are current"})

    for item in panel_mapping:
        if item["new_figure"] == "Fig1":
            add("Fig. 1", item["new_panel"], item["old_frozen_panel"] + ".csv")
    for number in range(2, 7):
        for path in sorted((tables / "Figure_sources").glob(f"Figure_{number:02d}*.csv")):
            add(f"Fig. {number}", "source table", "Figure_sources/" + path.name)
    order = json.loads((paper / "SUPPLEMENTARY_FIGURE_ORDER.json").read_text(encoding="utf-8"))
    for item in order:
        for key in source_lists[item["source_figure_number"]]:
            add(f"Supplementary Fig. S{item['display_number']}", "figure inputs", key)
    with (tables / "FIGURE_SOURCE_MAP.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def export_source_data(previous, paper, science, data, repo, out):
    """Refresh source tables and their current figure routes after analysis or redraw."""
    flat = paper / "figures"
    tables = out / "Supplementary_Data"
    shutil.copytree(previous / "Supplementary_Data", tables, dirs_exist_ok=True)
    strict = out / "strict_service"
    subprocess.run([sys.executable, str(repo / "src/analysis/experiments/candidate_support/strict_service.py"),
                    "--input-root", str(repo / "src/analysis/support/candidate_support"),
                    "--output-root", str(strict)], check=True, stdout=subprocess.DEVNULL)
    strict_target = tables / "Strict_service"
    strict_target.mkdir(exist_ok=True)
    for path in sorted(strict.glob("*.csv")):
        shutil.copyfile(path, strict_target / path.name)
    source_data = tables / "Main_comparison"
    for group in ("annual", "events", "costs", "portfolio", "nested_permissions", "common_layout", "dense_uniform_bound"):
        target = source_data / group
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted((science / group).glob("*.csv*")):
            shutil.copyfile(path, target / path.name)
    control_sources = previous / "paper/supplementary_figures"
    target_sources = tables / "Control_inputs"
    target_sources.mkdir(exist_ok=True)
    for path in sorted(control_sources.glob("actual_*_source.csv")):
        shutil.copyfile(path, target_sources / path.name)
    figure_sources = tables / "Figure_sources"
    figure_sources.mkdir(exist_ok=True)
    for path in sorted(flat.glob("*.csv")):
        shutil.copyfile(path, figure_sources / path.name)
    for path in sorted((paper / "figures/figure_01").glob("*source*.csv")):
        shutil.copyfile(path, figure_sources / path.name)
    for directory, source_root in (("Seasonal", out / "seasonal"), ("Typical_day_DR", out / "typical_day_dr")):
        target = tables / directory
        target.mkdir(exist_ok=True)
        for path in sorted(source_root.glob("*.csv*")):
            shutil.copyfile(path, target / path.name)
    for path in sorted((out / "typical_day_dr/figures").glob("*.csv")):
        shutil.copyfile(path, figure_sources / path.name)
    reading_order = json.loads((paper / 'SUPPLEMENTARY_READING_ORDER.json').read_text(encoding='utf-8'))
    table_aliases = reading_order['table_aliases_old_to_new']
    retained_tables = {old: (tables / f'TableS{old}.csv').read_bytes() for old in table_aliases}
    for old, new in table_aliases.items():
        (tables / f'TableS{new}.csv').write_bytes(retained_tables[old])
    export_supporting_sources(previous, paper, tables, data, repo)
    index = []
    for path in sorted(tables.rglob("*.csv*")):
        if path.name == "SOURCE_DATA_INDEX.csv":
            continue
        with (path.open(encoding="utf-8-sig", newline="") if path.suffix != ".gz" else
              gzip.open(path, "rt", encoding="utf-8-sig", newline="")) as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            row_count = sum(1 for _ in reader)
        index.append({"path": path.relative_to(tables).as_posix(), "rows": row_count,
                      "columns": len(fields), "sha256": sha256(path)})
    with (tables / "SOURCE_DATA_INDEX.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "rows", "columns", "sha256"])
        writer.writeheader()
        writer.writerows(index)
    return len(index)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--typical-day-data-root", type=Path, help="Verified extracted 44-file normalized typical-day supplement")
    parser.add_argument("--plan", action="store_true", help="Show the scientific workflow without reading data or writing outputs.")
    args = parser.parse_args()
    publication = Path(__file__).resolve().parent
    repo = REPOSITORY_ROOT
    import thermal_heterogeneity
    loaded_package = Path(thermal_heterogeneity.__file__).resolve()
    if not loaded_package.is_relative_to((repo / "src").resolve()):
        raise RuntimeError("Install this checkout in the locked environment before reproduction")
    source = publication / "energy_service_paper"
    if args.plan:
        scripts = [repo / "src/utilities/unpack_paper_data.py", repo / "src/utilities/unpack_typical_day_dr_data.py",
                   publication / "reproduce_publication.py",
                   publication / "reproduce_energy_service.py", source / "scripts/render_control_service_figures.py",
                   publication / "experiments/decision_value/summarize_information_hierarchy.py",
                   source / "scripts/render_response_implementation_figures.py", source / "scripts/render_response_traces.py",
                   source / "scripts/assemble_publication_assets.py",
                   publication / "seasonal/derive_seasonal_comparison.py", publication / "typical_day_dr/replay.py",
                   publication / "typical_day_dr/render.py", publication / "seasonal/extend_main_figure_three.py",
                   source / "scripts/finalize_temporal_assets.py"]
        print(json.dumps({"steps": [str(path.relative_to(repo).as_posix()) for path in scripts],
                          "data_files": 877, "typical_day_data_files": 44, "typical_day_manifest_files": 1, "figures": 30, "manuscripts_required": False,
                          "physical_simulations_run": 0}, indent=2))
        return
    if args.data_root is None or args.output_root is None or args.typical_day_data_root is None:
        parser.error("--data-root, --typical-day-data-root and --output-root are required unless --plan is used")
    data, out = args.data_root.resolve(), args.output_root.resolve()
    if out.exists():
        raise FileExistsError("Choose a new publication output directory")
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    steps = []

    def run(name, script, *arguments, python=None):
        result = subprocess.run([str(python or sys.executable), str(script), *map(str, arguments)],
                                text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, cwd=repo)
        (logs / f"{name}.log").write_text(result.stdout, encoding="utf-8")
        steps.append({"step": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")
        print(name, flush=True)

    run("00_verify_combined_inputs", repo / "src/utilities/unpack_paper_data.py", "--root", data,
        "--report", out / "PAPER_INPUT_VERIFICATION.json")
    run("00b_verify_typical_day_inputs", repo / "src/utilities/unpack_typical_day_dr_data.py",
        "--root", args.typical_day_data_root.resolve())
    previous = out / "supporting_evidence"
    run("01_supporting_human_and_library_analyses", publication / "reproduce_publication.py",
        "--data-root", data / "legacy_analysis", "--source-building-data", data / "source_building",
        "--output-root", previous, "--energy-service-input-root", data / "energy_service")
    science = out / "energy_service"
    run("02_three_controls_and_spatial_comparisons", publication / "reproduce_energy_service.py",
        "--input-root", data / "energy_service", "--output-root", science)
    hierarchy = out / "information_hierarchy"
    run(
        "02b_information_hierarchy",
        publication / "experiments/decision_value/summarize_information_hierarchy.py",
        "--phase-map", previous / "analysis/tables/result6_phase_map.csv",
        "--control-comparisons", science / "annual/CONTROL_COMPARISONS.csv",
        "--output-root", hierarchy,
    )
    paper = out / "paper"
    flat = paper / "figures"
    flat.mkdir(parents=True)
    run("03_service_energy_spatial_figures", source / "scripts/render_control_service_figures.py",
        "--annual-root", science / "annual",
        "--information-summary", hierarchy / "INFORMATION_HIERARCHY_SUMMARY.csv",
        "--output", flat)
    run("04_response_and_management_figures", source / "scripts/render_response_implementation_figures.py",
        "--results-root", science, "--output", flat)
    run("05_own_baseline_temperature_traces", source / "scripts/render_response_traces.py",
        "--results-root", science, "--engineering-root", data / "energy_service/engineering",
        "--service-module", publication / "energy_service/thermal_service.py", "--output", flat)
    run("06_assemble_main_and_supporting_graphics", source / "scripts/assemble_publication_assets.py",
        "--previous-root", previous / "paper", "--stage-root", paper)
    seasonal, typical = out / "seasonal", out / "typical_day_dr"
    run("07_monthly_and_seasonal_analysis", publication / "seasonal/derive_seasonal_comparison.py",
        "--engineering-root", data / "energy_service/engineering", "--annual-root", science / "annual",
        "--factors", data / "energy_service/factors/regional_factors.csv",
        "--analysis-module", publication / "energy_service/analyse_annual.py", "--output", seasonal)
    run("08_typical_day_response_analysis", publication / "typical_day_dr/replay.py",
        "--paper-data-root", data, "--annual-root", science / "annual",
        "--supplement-root", args.typical_day_data_root.resolve(), "--output-root", typical)
    run("09_typical_day_response_figure", publication / "typical_day_dr/render.py",
        "--replay-root", typical, "--output-root", typical / "figures")
    run("10_seasonal_main_and_supporting_figures", publication / "seasonal/extend_main_figure_three.py",
        "--main-builder", source / "scripts/render_control_service_figures.py", "--annual-root", science / "annual",
        "--seasonal-root", seasonal, "--output", seasonal / "figures")
    run("11_temporal_asset_reading_order", source / "scripts/finalize_temporal_assets.py",
        "--paper-root", paper, "--seasonal-figure-root", seasonal / "figures",
        "--typical-figure-root", typical / "figures")
    source_data_count = export_source_data(previous, paper, science, data, repo, out)
    files = [{"path": path.relative_to(out).as_posix(), "sha256": sha256(path)}
             for path in sorted(out.rglob("*")) if path.is_file()]
    report = {"status": "PASS", "scope": "Scientific comparisons, figures and source data",
              "external_paper_files": 877, "external_typical_day_files": 44,
              "retained_new_typical_day_trajectories": 32,
              "typical_day_analysis_replay_only": True,
              "main_figures": 6, "supplementary_figures": 24, "supplementary_numbered_tables": 19,
              "source_data_csvs": source_data_count,
              "physical_simulations_run": 0, "building_parameters_retuned": False,
              "package_loaded_from_current_repository": True,
              "manuscript_or_template_inputs_required": False, "word_documents_built": 0,
              "steps": steps, "outputs": files}
    (out / "SCIENTIFIC_REPLAY_RUN.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
