import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.target_domain_sensitivity import (
    _payload_hash,
    _trim_probe_archive,
    analyse_target_domain,
    annotate_candidates,
    matched_service_core,
    read_commands,
)


OFFSETS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    commands = []
    manifest = []
    metrics = []
    bases = {
        "uniform_average": 24.0,
        "personal_center": 20.0,
        "personal_range": 25.5,
    }
    for control, base in bases.items():
        for index, offset in enumerate(OFFSETS):
            target = float(np.clip(base + offset, 20.0, 27.0))
            targets = {"Zone 1": target, "Zone 2": target}
            run_id = f"{control}-{index}"
            commands.append(
                {
                    "context": "Test climate",
                    "composition_id": 1,
                    "layout": "random_mixing",
                    "permission": "native_zone",
                    "control": control,
                    "offset_c": offset,
                    "targets_c": targets,
                }
            )
            manifest.append(
                {
                    "run_id": run_id,
                    "climate": "Test climate",
                    "composition_id": 1,
                    "control": control,
                    "offset_c": offset,
                    "targets_sha256": _payload_hash(targets),
                }
            )
            control_shift = {
                "uniform_average": 0.0,
                "personal_center": -0.1,
                "personal_range": -0.2,
            }[control]
            metrics.append(
                {
                    "run_id": run_id,
                    "climate": "Test climate",
                    "composition": "Multinational A",
                    "layout": "random_mixing",
                    "permission": "native_zone",
                    "control": control,
                    "offset_c": offset,
                    "engineering_pass": True,
                    "physical_plan_id": run_id,
                    "mean_reference_mismatch_c": 1.0 + abs(offset) + control_shift,
                    "fixed_tail_reference_mismatch_c": 1.5 + abs(offset) + control_shift,
                    "zone_mean_mean_reference_mismatch_c": 1.0 + abs(offset),
                    "zone_mean_fixed_tail_reference_mismatch_c": 1.5 + abs(offset),
                    "modeled_interval_mean_excess_c": 0.8 + abs(offset) + control_shift,
                    "facility_purchased_kwh": 1000.0 + 20.0 * index
                    + 10.0 * list(bases).index(control),
                }
            )
    command_path = tmp_path / "commands.json"
    command_path.write_text(json.dumps(commands), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    metrics_path = tmp_path / "metrics.csv"
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    anchors_path = tmp_path / "anchors.csv"
    pd.DataFrame(
        [
            {
                "climate": "Test climate",
                "composition": "Multinational A",
                "layout": "random_mixing",
                "anchor_mean_mismatch_c": 3.0,
                "anchor_tail_mismatch_c": 3.0,
            }
        ]
    ).to_csv(anchors_path, index=False)
    return metrics_path, anchors_path, manifest_path, command_path


def test_candidate_annotation_tracks_boundary_and_central_domain(tmp_path: Path):
    metrics_path, _, manifest_path, command_path = _inputs(tmp_path)
    annotated = annotate_candidates(
        pd.read_csv(metrics_path),
        pd.read_csv(manifest_path),
        read_commands(command_path),
    )
    centre = annotated[
        annotated.control.eq("personal_center") & np.isclose(annotated.offset_c, 0.0)
    ].iloc[0]
    assert centre.base_optimizer_touches_declared_boundary
    assert not centre.candidate_within_central_domain
    shifted = annotated[
        annotated.control.eq("personal_center") & np.isclose(annotated.offset_c, 1.0)
    ].iloc[0]
    assert shifted.candidate_within_central_domain


def test_candidate_annotation_accepts_one_sided_domain_restrictions(tmp_path: Path):
    metrics_path, _, manifest_path, command_path = _inputs(tmp_path)
    inputs = (
        pd.read_csv(metrics_path),
        pd.read_csv(manifest_path),
        read_commands(command_path),
    )
    low_side = annotate_candidates(*inputs, central_domain_c=(21.0, 27.0))
    high_side = annotate_candidates(*inputs, central_domain_c=(20.0, 26.0))
    assert low_side.candidate_within_central_domain.any()
    assert high_side.candidate_within_central_domain.any()


def test_matched_service_accepts_unequal_central_candidate_depth(tmp_path: Path):
    metrics_path, _, manifest_path, command_path = _inputs(tmp_path)
    annotated = annotate_candidates(
        pd.read_csv(metrics_path),
        pd.read_csv(manifest_path),
        read_commands(command_path),
    )
    central = annotated[annotated.candidate_within_central_domain]
    result = matched_service_core(central, domain_scope="central_21_26_c")
    assert len(result) == 2
    assert result.selection_status.eq("selected").all()
    centre = result[result.alternative_control.eq("personal_center")].iloc[0]
    assert centre.uniform_candidate_count == 5
    assert centre.alternative_candidate_count == 1


def test_matched_service_reports_identity_with_no_retained_candidates(tmp_path: Path):
    metrics_path, _, manifest_path, command_path = _inputs(tmp_path)
    annotated = annotate_candidates(
        pd.read_csv(metrics_path),
        pd.read_csv(manifest_path),
        read_commands(command_path),
    )
    empty = annotated.iloc[0:0].copy()
    expected = annotated[list(("climate", "composition", "layout", "permission"))]
    result = matched_service_core(
        empty,
        domain_scope="central_21_26_c",
        expected_identities=expected.drop_duplicates(),
    )
    assert len(result) == 2
    assert result.selection_status.eq("candidate_unavailable").all()


def test_full_analysis_writes_auditable_outputs(tmp_path: Path):
    metrics_path, anchors_path, manifest_path, command_path = _inputs(tmp_path)
    output = tmp_path / "output"
    audit = analyse_target_domain(
        metrics_path,
        anchors_path,
        manifest_path,
        command_path,
        output,
    )
    assert audit["status"] == "COMPLETE"
    assert audit["counts"]["annual_candidates"] == 15
    assert audit["counts"]["base_optimizers_touching_declared_boundary"] == 1
    assert (output / "TARGET_DOMAIN_MATCHED_SERVICE.csv").is_file()
    assert (output / "TARGET_DOMAIN_COMMON_SERVICE_SELECTIONS.csv").is_file()
    assert (output / "TARGET_DOMAIN_INFORMATION_DIAGNOSTIC.csv").is_file()


def test_full_analysis_labels_one_sided_sensitivity_domain(tmp_path: Path):
    metrics_path, anchors_path, manifest_path, command_path = _inputs(tmp_path)
    audit = analyse_target_domain(
        metrics_path,
        anchors_path,
        manifest_path,
        command_path,
        tmp_path / "one-sided",
        central_domain_c=(20.0, 26.0),
    )
    assert "central_20_26_c" in audit["matched_service"]
    assert "central_21_26_c" not in audit["matched_service"]


def test_probe_trim_uses_only_existing_central_probe_rows(tmp_path: Path):
    source = tmp_path / "source.npz"
    target = np.arange(20.0, 28.0)
    np.savez_compressed(
        source,
        uniform_target_c=target,
        operative_c=np.arange(8 * 4 * 2, dtype=float).reshape(8, 4, 2),
        native_people=np.ones((4, 2)),
    )
    destination = tmp_path / "central.npz"
    _trim_probe_archive(source, destination, 21.0, 26.0)
    with np.load(destination, allow_pickle=False) as saved:
        assert saved["uniform_target_c"].tolist() == [21, 22, 23, 24, 25, 26]
        assert saved["operative_c"].shape == (6, 4, 2)
        assert saved["native_people"].shape == (4, 2)
