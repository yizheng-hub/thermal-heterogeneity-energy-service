"""Redraw all 30 figures from an existing scientific replay without rerunning analyses."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--typical-day-data-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data, out = args.data_root.resolve(), args.replay_root.resolve()
    run_report = out / "SCIENTIFIC_REPLAY_RUN.json"
    if not run_report.is_file() or json.loads(run_report.read_text(encoding="utf-8"))["status"] != "PASS":
        parser.error("--replay-root must contain a successful ctl/reproduce.py run")
    previous, paper = out / "supporting_evidence", out / "paper"
    publication = root / "publication"
    current = publication / "energy_service_paper/scripts"
    steps = []

    def run(name, script, *arguments):
        result = subprocess.run([sys.executable, str(script), *map(str, arguments)], cwd=root,
                                text=True, encoding="utf-8", stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (out / "logs" / f"redraw_{name}.log").write_text(result.stdout, encoding="utf-8")
        steps.append({"step": name, "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see its redraw log")
        print(name, flush=True)

    run("00_verify_inputs", root / "scripts/unpack_paper_data.py", "--root", data)
    run("00b_verify_typical_day", root / "scripts/unpack_typical_day_dr_data.py", "--root", args.typical_day_data_root.resolve())
    run("01_supporting_main", publication / "scripts/render_revised_main_figures.py",
        "--release-root", root, "--panel-data", previous / "panel_data", "--output-root", previous / "paper",
        "--decision-value", previous / "decision_value/MAIN_FIG6G_Q5_REGRET.csv")
    run("02_supporting_figures", publication / "scripts/render_supplementary.py",
        "--data-root", data / "legacy_analysis", "--panel-data", previous / "panel_data",
        "--output", previous / "paper/supplementary_figures", "--energy-service-input-root", data / "energy_service",
        "--science-audit", publication / "support/science_audit",
        "--spatial-audit", publication / "support/spatial_audit",
        "--land", root / "src/thermal_heterogeneity/assets/ne_110m_land.geojson")
    run("03_annual", current / "render_control_service_figures.py",
        "--annual-root", out / "energy_service/annual", "--output", paper / "figures")
    run("04_response_costs", current / "render_response_implementation_figures.py",
        "--results-root", out / "energy_service", "--output", paper / "figures")
    run("05_traces", current / "render_response_traces.py",
        "--results-root", out / "energy_service", "--engineering-root", data / "energy_service/engineering",
        "--service-module", publication / "energy_service/thermal_service.py", "--output", paper / "figures")
    run("06_assemble", current / "assemble_publication_assets.py",
        "--previous-root", previous / "paper", "--stage-root", paper)
    run("07_typical_day_figure", publication / "typical_day_dr/render.py",
        "--replay-root", out / "typical_day_dr", "--output-root", out / "typical_day_dr/figures")
    run("08_seasonal_figures", publication / "seasonal/extend_main_figure_three.py",
        "--main-builder", current / "render_control_service_figures.py", "--annual-root", out / "energy_service/annual",
        "--seasonal-root", out / "seasonal", "--output", out / "seasonal/figures")
    run("09_finalize_temporal", current / "finalize_temporal_assets.py", "--paper-root", paper,
        "--seasonal-figure-root", out / "seasonal/figures", "--typical-figure-root", out / "typical_day_dr/figures")
    spec = importlib.util.spec_from_file_location("scientific_replay", publication / "reproduce_energy_service_publication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = module.export_source_data(previous, paper, out / "energy_service", data, root, out)
    report = {"status": "PASS", "figures": 30, "source_data_csvs": count,
              "analyses_rerun": False, "physical_simulations_run": 0, "steps": steps,
              "scientific_replay_record_sha256": hashlib.sha256(run_report.read_bytes()).hexdigest(),
              "source_files": [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                               for name in (root / "protocol/PUBLIC_FILE_ALLOWLIST.txt").read_text(encoding="utf-8").splitlines()
                               if name and not name.startswith("#")],
              "outputs": [{"path": path.relative_to(out).as_posix(),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for path in sorted(out.rglob("*"))
                          if path.is_file() and path.name != "FIGURE_REDRAW_RUN.json"]}
    (out / "FIGURE_REDRAW_RUN.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"outputs", "source_files"}}, indent=2))


if __name__ == "__main__":
    main()
