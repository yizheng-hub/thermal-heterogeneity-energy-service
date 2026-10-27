from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "global_extension"
sys.path.insert(0, str(MODULE))

import global_spatial_extension as spatial  # noqa: E402
from global_spatial_extension import (  # noqa: E402
    DEMAND_ORDERED_LAYOUTS,
    DEMAND_ORDERED_ZONE_ORDERS,
    METRICS,
    NEW_LOGICAL_RUNS,
    NEW_PHYSICAL_RUNS,
    PERSONAL_CONTROLS,
    _fixed_tail_positions,
    _pair_effects,
    _physical_run_id,
    demand_ordered_profiles,
    select_extension_designs,
)
from run_global_matrix import ZONES  # noqa: E402
from global_matrix import sha256  # noqa: E402


def _random_profiles() -> pd.DataFrame:
    counts = [18] * 12 + [17] * 3
    rows = []
    position = 0
    for zone, count in zip(ZONES, counts, strict=True):
        for _ in range(count):
            rows.append(
                {
                    "context": "Test climate",
                    "composition_id": 1,
                    "layout": "random_mixing",
                    # Reverse-looking names ensure stable ties are not silently
                    # reordered by identity.
                    "profile_id": f"person_{267 - position:03d}",
                    "zone_id": zone,
                    "centre_c": float(position % 7 + 20),
                    "controller_lower_c": float(position % 7 + 19),
                    "controller_upper_c": float(position % 7 + 21),
                }
            )
            position += 1
    return pd.DataFrame(rows)


def test_demand_ordering_preserves_identities_and_native_zone_slots():
    random_profiles = _random_profiles()
    counts = random_profiles.zone_id.value_counts().to_dict()
    stable = random_profiles.sort_values("centre_c", kind="mergesort")
    for layout, zone_order in DEMAND_ORDERED_ZONE_ORDERS.items():
        ordered = demand_ordered_profiles(random_profiles, layout=layout)

        assert len(ordered) == 267
        assert set(ordered.profile_id) == set(random_profiles.profile_id)
        assert ordered.layout.eq(layout).all()
        assert ordered.zone_id.value_counts().to_dict() == counts

        assigned = np.concatenate(
            [np.repeat(zone, counts[zone]) for zone in zone_order]
        )
        expected = dict(zip(stable.profile_id, assigned, strict=True))
        observed = dict(zip(ordered.profile_id, ordered.zone_id, strict=True))
        assert observed == expected

    assert DEMAND_ORDERED_ZONE_ORDERS["demand_ordered"] == tuple(ZONES)
    assert DEMAND_ORDERED_ZONE_ORDERS["demand_ordered_reversed"] == tuple(
        reversed(ZONES)
    )
    assert DEMAND_ORDERED_ZONE_ORDERS["demand_ordered_shift5"] == tuple(
        (*ZONES[5:], *ZONES[:5])
    )


def test_design_selection_is_exactly_nine_by_six_by_six():
    designs = []
    for climate_index in range(9):
        for composition in range(6):
            for layout in ("random_mixing", *DEMAND_ORDERED_LAYOUTS):
                for permission in ("building", "floor", "native_zone"):
                    for control in ("uniform_average", *PERSONAL_CONTROLS):
                        for offset in (0.0, 0.5):
                            designs.append(
                                {
                                    "context": f"climate_{climate_index}",
                                    "composition_id": composition,
                                    "layout": layout,
                                    "permission": permission,
                                    "control": control,
                                    "offset_c": offset,
                                }
                            )

    selected = select_extension_designs(designs)

    assert NEW_LOGICAL_RUNS == 540
    assert NEW_PHYSICAL_RUNS == 516
    assert len(selected) == NEW_LOGICAL_RUNS
    assert {row["control"] for row in selected} == set(PERSONAL_CONTROLS)
    assert {(row["layout"], row["permission"]) for row in selected} == {
        ("random_mixing", "building"),
        ("random_mixing", "floor"),
        *((layout, "native_zone") for layout in DEMAND_ORDERED_LAYOUTS),
    }


def test_random_anchor_tail_identities_are_unchanged_by_layout_mapping():
    random_profiles = _random_profiles().sort_values(
        "profile_id", kind="mergesort"
    ).reset_index(drop=True)
    fixed_ids = tuple(random_profiles.profile_id.iloc[[0, 19, 88, 143, 266]])

    for layout in DEMAND_ORDERED_LAYOUTS:
        ordered = demand_ordered_profiles(
            _random_profiles(), layout=layout
        ).sort_values("profile_id", kind="mergesort").reset_index(drop=True)
        positions = _fixed_tail_positions(ordered, fixed_ids)
        assert tuple(ordered.profile_id.iloc[positions]) == fixed_ids


def test_physical_run_identifier_depends_only_on_physical_inputs():
    common = {
        "climate": "Beijing",
        "model_sha256": "a" * 64,
        "weather_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
    }
    first = _physical_run_id(targets={zone: 24.0 for zone in ZONES}, **common)
    reordered = _physical_run_id(targets={zone: 24.0 for zone in reversed(ZONES)}, **common)
    changed = _physical_run_id(
        targets={zone: 24.5 if index == 0 else 24.0 for index, zone in enumerate(ZONES)},
        **common,
    )

    assert first == reordered
    assert changed != first


def test_permission_effects_pair_exact_whole_building_metrics():
    rows = []
    for permission, value in (("building", 100.0), ("floor", 90.0), ("native_zone", 80.0)):
        row = {
            "climate": "Beijing",
            "composition": 1,
            "control": "personal_center",
            "layout": "random_mixing",
            "permission": permission,
            "run_id": permission,
            "engineering_pass": True,
        }
        row.update({metric: value for metric in METRICS})
        rows.append(row)

    result = _pair_effects(
        pd.DataFrame(rows),
        axis="permission",
        reference="building",
        alternatives=("floor", "native_zone"),
    )

    assert len(result) == 2
    assert result.status.eq("matched").all()
    assert result.set_index("alternative").loc["floor", "delta_facility_purchased_kwh"] == -10.0
    assert (
        result.set_index("alternative").loc["native_zone", "delta_operational_co2_country_kg"]
        == -20.0
    )
    assert result.set_index("alternative").loc["floor", "delta_facility_purchased_pct"] == -10.0


def test_execute_manifest_is_compatible_with_solver_stability_audit(
    tmp_path: Path,
    monkeypatch,
):
    runtime = tmp_path / "energyplus.exe"
    model = tmp_path / "input.idf"
    weather = tmp_path / "weather.epw"
    runtime.write_bytes(b"runtime")
    model.write_text("idf", encoding="utf-8")
    weather.write_text("weather", encoding="utf-8")
    execution = pd.DataFrame(
        [
            {
                "run_id": "spatial_test",
                "climate": "climate_0",
                "composition_id": 1,
                "control": "personal_center",
                "offset_c": 0.0,
                "layout": "random_mixing",
                "permission": "floor",
                "input_idf_path": str(model),
                "input_sha256": sha256(model),
                "weather_path": str(weather),
                "weather_sha256": sha256(weather),
                "runtime_sha256": sha256(runtime),
            }
        ]
    )
    references = pd.DataFrame(
        [
            {
                "run_id": f"reference_{index}",
                "climate": f"climate_{index}",
                "target_c": 25.0,
            }
            for index in range(9)
        ]
    )
    stability_path = tmp_path / "stability.csv"
    stability_path.write_text("placeholder", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_execute(jobs, executable, workers):
        seen["workers"] = workers
        seen["executable"] = executable
        return [
            jobs[0]
            | {
                "run_qa": {
                    "completed": True,
                    "reported_severe_count": 0,
                    "fatal_occurrences": 0,
                    "unexplained_recurring_groups": [],
                    "hourly_sha256": "d" * 64,
                }
            }
        ]

    monkeypatch.setattr(spatial, "_execute_jobs", fake_execute)
    monkeypatch.setattr(spatial, "warning_profile", lambda path: {})
    monkeypatch.setattr(
        spatial,
        "relative_engineering_qa",
        lambda qa, treatment, baseline: {
            "engineering_pass_relative_to_prototype_baseline": False,
            "new_warning_categories": ["hvac_iterations"],
            "warning_count_increases": {"hvac_iterations": {"baseline": 0, "treatment": 1}},
            "critical_warning_count_increases": {
                "hvac_iterations": {"baseline": 0, "treatment": 1}
            },
            "baseline_warning_profile": {},
            "treatment_warning_profile": {"hvac_iterations": 1},
        },
    )
    monkeypatch.setattr(
        spatial,
        "read_annual_solver_stability_evidence",
        lambda path: ({"spatial_test": {"evidence": True}}, "e" * 64),
    )

    def fake_apply(relative, qa, evidence, **kwargs):
        seen["evidence"] = evidence
        seen.update(kwargs)
        return True, {
            "raw_relative_engineering_pass": False,
            "annual_solver_stability_exception_applied": True,
        }

    monkeypatch.setattr(spatial, "apply_annual_solver_stability_exception", fake_apply)
    output = tmp_path / "output"
    result = spatial.execute_plan(
        execution,
        references,
        tmp_path / "probe",
        runtime,
        output,
        workers=2,
        solver_stability_manifest=stability_path,
    )

    assert bool(result.loc[0, "relative_engineering_pass"])
    assert bool(result.loc[0, "final_engineering_qualified"])
    assert result.loc[0, "composition_id"] == 1
    assert result.loc[0, "control"] == "personal_center"
    assert seen["evidence"] == {"evidence": True}
    assert seen["current_maximum_hvac_iterations"] == 40
    assert seen["active_input_sha256"] == sha256(model)
    assert seen["active_hourly_sha256"] == "d" * 64
    assert (output / "GLOBAL_SPATIAL_RUN_MANIFEST_COMPLETED.csv").is_file()
