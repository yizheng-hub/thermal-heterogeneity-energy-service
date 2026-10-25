"""The final replay must include both cohorts and fail on missing joint evidence."""
import csv
import gzip
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/analysis/targeted_extension"))
from replay_parallel_main import workflow  # noqa: E402


def test_shared_replay_connects_frozen_inputs_and_joint_decisions(tmp_path):
    primary, bundle, output = (tmp_path / name for name in ("primary", "bundle", "output"))
    steps = workflow(ROOT, primary, bundle, output)
    names = [name for name, _, _ in steps]
    assert names[:7] == ["annual", "common_layout", "nested_permissions", "spatial", "service", "seasonal", "events"]
    assert names[-4:] == ["joint_portfolio", "joint_events", "joint_main", "independent_joint_audit"]
    assert "joint_spatial_diagnostics" in names
    event_args = next(args for name, _, args in steps if name == "events")
    assert event_args[event_args.index("--engineering-root") + 1] == bundle / "events/engineering"
    assert event_args[event_args.index("--expected-baselines") + 1] == "216"
    assert event_args[event_args.index("--expected-profile-groups") + 1] == "24"
    for cohort in ("ashrae", "ctcd"):
        assert cohort + "_portfolio" in names
        assert cohort + "_costs" in names
    assert all("energyplus" not in str(script).lower() for _, script, _ in steps)


def plot_collector():
    spec = spec_from_file_location("joint_plot_collector", ROOT / "src/utilities/prepare_plot_data.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_source_copy_indexes_plain_and_compressed_tables(tmp_path):
    source, target = tmp_path / "source", tmp_path / "results"
    target.mkdir()
    (target / "SOURCE_DATA_INDEX.csv").write_text("path,rows,columns,sha256\nold.csv,1,1,locked\n")
    for name in ("joint_main", "joint_spatial", "joint_spatial_diagnostics", "joint_events", "joint_portfolio"):
        (source / name).mkdir(parents=True)
        (source / name / "values.csv").write_text("value\n1\n2\n")
    (source / "joint_main/JOINT_EVIDENCE_QA.json").write_text('{"status": "PASS"}')
    with gzip.open(source / "joint_events/hourly.csv.gz", "wt") as stream:
        stream.write("hour,value\n1,3\n")
    module = plot_collector()
    module.prepare_joint_sources(source, target)
    with (target / "SOURCE_DATA_INDEX.csv").open(newline="") as stream:
        index = {row["path"]: row for row in csv.DictReader(stream)}
    assert index["old.csv"]["sha256"] == "locked"
    assert index["Joint_main/joint_events/hourly.csv.gz"]["rows"] == "1"
    assert index["Joint_main/joint_events/hourly.csv.gz"]["columns"] == "2"
    for name, row in index.items():
        if name.startswith("Joint_main/"):
            assert row["sha256"] == hashlib.sha256((target / name).read_bytes()).hexdigest()
    previous = (target / "SOURCE_DATA_INDEX.csv").read_bytes()
    module.prepare_joint_sources(source, target)
    assert (target / "SOURCE_DATA_INDEX.csv").read_bytes() == previous


def test_joint_sources_do_not_silently_omit_a_required_component(tmp_path):
    source = tmp_path / "source"
    (source / "joint_main").mkdir(parents=True)
    (source / "joint_main/JOINT_EVIDENCE_QA.json").write_text('{"status": "PASS"}')
    with pytest.raises(FileNotFoundError, match="component"):
        plot_collector().prepare_joint_sources(source, tmp_path / "results")


def test_current_main_panel_map_replaces_old_inputs_and_preserves_si(tmp_path):
    module = plot_collector()
    fields = ["current_figure", "panel", "source_key", "source_kind", "relative_path", "sha256", "identifier_note"]
    old = [dict(zip(fields, row)) for row in (
        ("Fig. 2", "a", "old_main", "Source Data", "CTCD_external/old.csv", "old", "old"),
        ("Fig. 1", "g", "old_compositions", "Source Data", "legacy_analysis/old.csv", "old", "old"),
        ("Supplementary Fig. S16", "b", "retained_source", "Source Data", "CTCD_external/old.csv", "locked", "retained"))]
    mapping = tmp_path / "FIGURE_SOURCE_MAP.csv"
    with mapping.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(old)
    for _, _, relative in module.JOINT_PANEL_SOURCES:
        path = tmp_path / "Joint_main" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value\n1\n")
    information = tmp_path / module.INFORMATION_PANEL_SOURCE[2]
    information.parent.mkdir(parents=True, exist_ok=True)
    information.write_text("metric,count,denominator\nzone_mean_safe_cells,275,606\n")
    module.refresh_joint_source_map(tmp_path)
    with mapping.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0] == old[2]
    assert not any(row["source_key"] in {"old_main", "old_compositions"} for row in rows)
    panels = {(row["current_figure"], row["panel"]) for row in rows}
    assert ("Fig. 3", "h") in panels and ("Fig. 1", "g") in panels
    assert ("Fig. 2", "f") in panels
    for row in rows[1:]:
        assert row["sha256"] == hashlib.sha256((tmp_path / row["relative_path"]).read_bytes()).hexdigest()


def test_information_hierarchy_is_copied_and_indexed(tmp_path):
    module = plot_collector()
    replay, destination = tmp_path / "replay", tmp_path / "results"
    source = replay / "information_hierarchy"
    source.mkdir(parents=True)
    (source / "INFORMATION_HIERARCHY_SUMMARY.csv").write_text(
        "metric,count,denominator\nzone_mean_safe_cells,275,606\n"
    )
    (source / "INFORMATION_HIERARCHY_SUMMARY.json").write_text('{"status":"PASS"}\n')
    destination.mkdir()
    (destination / "SOURCE_DATA_INDEX.csv").write_text(
        "path,rows,columns,sha256\nold.csv,1,1,locked\n"
    )
    module.prepare_information_hierarchy(replay, destination)
    copied = destination / "decision_value/INFORMATION_HIERARCHY_SUMMARY.csv"
    assert copied.read_bytes() == (source / copied.name).read_bytes()
    with (destination / "SOURCE_DATA_INDEX.csv").open(newline="") as stream:
        index = {row["path"]: row for row in csv.DictReader(stream)}
    assert index["old.csv"]["sha256"] == "locked"
    assert index[copied.relative_to(destination).as_posix()]["rows"] == "1"
