"""Reproduce scientific analyses, 31 figures and source tables from external inputs."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    arguments = list(sys.argv[1:])
    if "--plan" not in arguments:
        for flag, folder in (("--data-root", "data/inputs"),
                             ("--typical-day-data-root", "data/typical_days"),
                             ("--output-root", "output/replay")):
            if flag not in arguments:
                arguments.extend((flag, str(root / folder)))
    command = [
        sys.executable, str(root / "src/analysis/reproduce_energy_service_publication.py"),
        *arguments,
    ]
    if "--plan" in arguments:
        result = subprocess.run(command, cwd=root, env=environment, text=True,
                                encoding="utf-8", capture_output=True, check=True)
        plan = json.loads(result.stdout)
        plan["figures"] = 31
        plan["steps"].append("src/analysis/targeted_extension/run_extension.py")
        plan["steps"].append("src/analysis/targeted_extension/replay_ctcd_external.py")
        plan["steps"].append("src/analysis/targeted_extension/replay_parallel_main.py")
        plan["steps"].append("src/analysis/targeted_extension/reviewer_sensitivity.py")
        plan["steps"].append(
            "src/analysis/experiments/decision_value/summarize_information_hierarchy.py")
        plan["bounded_extension"] = "Four retained-trajectory checks; frozen physics unchanged"
        plan["external_office_composition"] = "CTCD: regenerate 144 commands and score retained annual trajectories"
        ctcd = json.loads((root / "data/metadata/ctcd_bundle_source.json").read_text(encoding="utf-8"))
        plan["ctcd_data_files"] = ctcd["files_excluding_manifest"]
        plan["ctcd_manifest_files"] = 1
        parallel = json.loads((root / "data/metadata/ctcd_parallel_bundle_source.json").read_text(encoding="utf-8"))
        plan["parallel_ctcd_data_files"] = parallel["files_excluding_manifest"]
        plan["parallel_ctcd_manifest_files"] = 1
        plan["main_demand_compositions"] = 6
        plan["common_main_layouts"] = 2
        plan["joint_portfolio_offices"] = 144
        print(json.dumps(plan, indent=2))
        raise SystemExit(0)
    status = subprocess.call(command, cwd=root, env=environment)
    if status:
        raise SystemExit(status)
    replay = Path(arguments[arguments.index("--output-root") + 1]).resolve()
    subprocess.run([sys.executable, str(root / "src/analysis/targeted_extension/run_extension.py"),
                    "--repository-root", str(root), "--replay-root", str(replay),
                    "--output-root", str(replay / "targeted_extension")],
                   cwd=root, env=environment, check=True)
    subprocess.run([sys.executable,
                    str(root / "src/analysis/targeted_extension/replay_ctcd_external.py"),
                    "--input-root", str(root / "data/inputs/ctcd_external"),
                    "--main-input-root", str(root / "data/inputs/energy_service"),
                    "--output-root", str(replay / "ctcd_external")],
                   cwd=root, env=environment, check=True)
    subprocess.run([sys.executable,
                    str(root / "src/analysis/targeted_extension/replay_parallel_main.py"),
                    "--repository-root", str(root), "--primary-replay-root", str(replay),
                    "--output-root", str(replay / "ctcd_parallel")],
                   cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, str(root / "src/utilities/prepare_plot_data.py"),
                    "--replay", str(replay)], cwd=root, env=environment, check=True)
    subprocess.run([sys.executable,
                    str(root / "src/analysis/targeted_extension/reviewer_sensitivity.py"),
                    "--repository-root", str(root),
                    "--output-root", str(replay / "reviewer_sensitivity")],
                   cwd=root, env=environment, check=True)
    subprocess.run([
        sys.executable,
        str(root / "src/analysis/experiments/decision_value/"
            "summarize_information_hierarchy.py"),
        "--phase-map",
        str(replay / "supporting_evidence/analysis/tables/result6_phase_map.csv"),
        "--control-comparisons",
        str(root / "results/Joint_main/annual/CONTROL_COMPARISONS.csv"),
        "--output-root", str(replay / "information_hierarchy"),
    ], cwd=root, env=environment, check=True)
    subprocess.run([sys.executable, str(root / "plot/reproduce.py")],
                   cwd=root, env=environment, check=True)
    report_path = replay / "SCIENTIFIC_REPLAY_RUN.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["supplementary_figures"] = 25
    report["bounded_extension_manifest"] = "targeted_extension/BOUNDED_EXTENSION_MANIFEST.json"
    report["external_ctcd_replay"] = "ctcd_external/CTCD_REPLAY_QA.json"
    report["parallel_main_replay"] = "ctcd_parallel/PARALLEL_MAIN_REPLAY_QA.json"
    report["reviewer_sensitivity"] = "reviewer_sensitivity/REVIEWER_SENSITIVITY_QA.json"
    report["information_hierarchy"] = (
        "information_hierarchy/INFORMATION_HIERARCHY_SUMMARY.json")
    report["main_demand_compositions"] = 6
    report["common_main_layouts"] = 2
    report["joint_portfolio_offices"] = 144
    report["additional_steps"] = [
        {"step": name, "returncode": 0} for name in
        ("bounded_comparison_checks", "ctcd_external_replay", "parallel_main_replay",
         "prepare_plot_data", "reviewer_sensitivity", "information_hierarchy",
         "31_notebook_figures")]
    report["final_figures_from_editable_notebook"] = True
    with (root / "results/SOURCE_DATA_INDEX.csv").open(encoding="utf-8-sig", newline="") as stream:
        report["final_plot_source_data_csvs"] = sum(1 for _ in csv.DictReader(stream))
    report["base_source_data_csvs"] = report.get("base_source_data_csvs", report["source_data_csvs"])
    report["source_data_csvs"] = report["final_plot_source_data_csvs"]
    report["final_plot_input_manifest_sha256"] = hashlib.sha256(
        (root / "results/PLOT_INPUTS.json").read_bytes()).hexdigest()
    report["editable_figure_outputs_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (root / "plot/main", root / "plot/supplementary")
        for path in sorted(directory.glob("Figure_*"))
        if path.suffix in (".png", ".pdf", ".svg")}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
