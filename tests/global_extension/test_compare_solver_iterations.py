from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

from compare_solver_iterations import compare_probe_manifests  # noqa: E402


def _compact(path: Path, temperature: float, electricity: float) -> None:
    hours = 24
    np.savez_compressed(
        path,
        zone_ids=np.array(["zone"]),
        temperatures=np.full((hours, 1), temperature),
        people=np.ones((hours, 1)),
        occupied=np.ones(hours, dtype=bool),
        hvac_electric_kwh=np.full(hours, electricity),
        gas_hhv_kwh=np.zeros(hours),
    )


def test_solver_comparison_reports_stable_probe(tmp_path: Path):
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    _compact(first_path, 25.0, 10.0)
    _compact(second_path, 25.005, 10.005)
    first = pd.DataFrame([{
        "climate": "Beijing", "target_c": 25.0,
        "maximum_hvac_iterations": 20, "hourly_path": str(first_path),
        "treatment_warning_profile": '{"hvac_iterations": 1}',
    }])
    second = pd.DataFrame([{
        "climate": "Beijing", "target_c": 25.0,
        "maximum_hvac_iterations": 40, "hourly_path": str(second_path),
        "treatment_warning_profile": '{"hvac_iterations": 0}',
    }])
    result = compare_probe_manifests(first, second)
    assert len(result) == 1
    assert result.trajectory_stable_within_declared_tolerance.iloc[0]
    assert result.first_maximum_hvac_iterations.iloc[0] == 20
    assert result.second_maximum_hvac_iterations.iloc[0] == 40


def test_solver_comparison_rejects_missing_probe(tmp_path: Path):
    path = tmp_path / "probe.npz"
    _compact(path, 25.0, 10.0)
    first = pd.DataFrame([{
        "climate": "Beijing", "target_c": 25.0, "hourly_path": str(path),
    }])
    second = pd.DataFrame([{
        "climate": "Beijing", "target_c": 24.5, "hourly_path": str(path),
    }])
    try:
        compare_probe_manifests(first, second)
    except ValueError as error:
        assert "same probes" in str(error)
    else:
        raise AssertionError("Mismatched solver manifests must fail")
