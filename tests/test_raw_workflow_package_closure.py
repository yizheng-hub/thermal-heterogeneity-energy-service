"""Ensure a source-only package includes every declared raw-workflow command."""

from pathlib import Path
import runpy
import json
import subprocess
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_all_scientific_python_sources_are_in_the_package_allowlist():
    allowed = set((ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines())
    missing = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in p.parts and p.relative_to(ROOT).as_posix() not in allowed]
    assert not missing, missing


def test_raw_secondary_and_strengthening_commands_are_packaged():
    allowed = set((ROOT / "src/provenance/PUBLIC_FILE_ALLOWLIST.txt").read_text().splitlines())
    secondary = runpy.run_path(str(ROOT / "reproduce_secondary_analyses.py"))
    commands = secondary["commands"](ROOT / "output/not_run")
    methods = runpy.run_path(str(ROOT / "reproduce_strengthening.py"))
    files = [Path(command[1]) for command in commands.values()]
    files += [ROOT / "src/analysis/strengthening" / command[0]
              for phase in methods["PHASES"].values() for command in phase]
    for path in files:
        assert path.is_file(), path
        assert path.relative_to(ROOT).as_posix() in allowed, path


def test_secondary_end_use_command_reproduces_scored_energy_differences(tmp_path):
    annual = tmp_path / "results/global_final"
    annual.mkdir(parents=True)
    metrics = pd.DataFrame({"run_id": ["reference", "more", "less", "most"],
        "heating_electric_kwh": [10., 20., 5., 30.],
        "cooling_electric_kwh": [20., 25., 20., 30.],
        "fans_electric_kwh": [5.] * 4, "gas_hhv_kwh": [65., 70., 60., 65.],
        "facility_purchased_kwh": [100., 120., 90., 130.]})
    metrics.to_csv(annual / "GLOBAL_RUN_METRICS.csv", index=False)
    direct = pd.DataFrame({"alternative_run_id": ["more", "less", "most"],
        "reference_run_id": ["reference"] * 3,
        "alternative_control": ["personal_center", "personal_range", "personal_center"],
        "climate": ["Context"] * 3, "composition": ["Sample"] * 3,
        "delta_facility_purchased_kwh": [20., -10., 30.]})
    direct.to_csv(annual / "GLOBAL_DIRECT_COMPARISONS.csv", index=False)
    ledger = tmp_path / "matrix/annual_runs/ANNUAL_CONTROL_RUN_MANIFEST.csv"
    ledger.parent.mkdir(parents=True)
    metrics[["run_id"]].to_csv(ledger, index=False)
    command = runpy.run_path(str(ROOT / "reproduce_secondary_analyses.py"))["commands"](tmp_path)["hvac_end_uses"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    output = tmp_path / "results/hvac_end_use_mechanism"
    actual = pd.read_csv(output / "HVAC_END_USE_DIRECT_COMPARISONS.csv")
    assert actual.delta_whole_building_purchased_kwh.tolist() == [20., -10., 30.]
    report = json.loads((output / "HVAC_END_USE_SUMMARY.json").read_text())
    assert report["comparisons"] == 3 and report["whole_building_energy_increases"] == 2
def test_reinitialization_preserves_identical_identity_link_bytes(tmp_path):
    import gzip
    import pandas as pd
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/analysis/strengthening"))
    from prepare_strengthening_inputs import write_gzip_csv_if_changed

    frame = pd.DataFrame({"profile_id": ["a", "b"], "person_key": ["person1", "person2"]})
    target = tmp_path / "identity.csv.gz"
    original = gzip.compress(frame.to_csv(index=False).encode("utf-8"), mtime=123)
    target.write_bytes(original)
    write_gzip_csv_if_changed(frame, target)
    assert target.read_bytes() == original
    frame.loc[0, "person_key"] = "corrected_person"
    write_gzip_csv_if_changed(frame, target)
    changed = target.read_bytes()
    assert gzip.decompress(changed) == frame.to_csv(index=False).encode("utf-8")
    write_gzip_csv_if_changed(frame, target)
    assert target.read_bytes() == changed


