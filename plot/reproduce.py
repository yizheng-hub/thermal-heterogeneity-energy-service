"""Redraw all 27 figures from an existing scientific replay without rerunning analyses."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
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
    spec = importlib.util.spec_from_file_location("scientific_replay", publication / "reproduce_energy_service_publication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    count = module.export_source_data(previous, paper, out / "energy_service", data, root, out)
    report = {"status": "PASS", "figures": 27, "source_data_csvs": count,
              "analyses_rerun": False, "physical_simulations_run": 0, "steps": steps}
    (out / "FIGURE_REDRAW_RUN.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
