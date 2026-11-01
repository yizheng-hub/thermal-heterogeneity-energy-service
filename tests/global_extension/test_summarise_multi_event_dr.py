from pathlib import Path

import pandas as pd
import pytest

from analysis.global_extension.summarise_multi_event_dr import build_summary


CONTROLS = ("uniform_average", "personal_center", "personal_range")


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    contexts = [
        {
            "climate": "Warm", "composition_id": 1, "event_id": "e1",
            "action_id": "uniform_0.5", "local_season": "summer",
            "load_level": "typical",
        },
        {
            "climate": "Cold", "composition_id": 2, "event_id": "e2",
            "action_id": "uniform_1.0", "local_season": "winter",
            "load_level": "high",
        },
    ]
    qualifications = {
        ("e1", "uniform_average"): False,
        ("e1", "personal_center"): True,
        ("e1", "personal_range"): True,
        ("e2", "uniform_average"): True,
        ("e2", "personal_center"): False,
        ("e2", "personal_range"): True,
    }
    power = {"uniform_average": 10.0, "personal_center": 8.0, "personal_range": 9.0}
    runs: list[dict] = []
    events: list[dict] = []
    scores: list[dict] = []
    recovery: list[dict] = []
    for context in contexts:
        for control in CONTROLS:
            run_id = f"{context['event_id']}-{control}"
            runs.append({
                "run_id": run_id, "completed": True,
                "final_engineering_qualified": True,
            })
            event = {"run_id": run_id, "control": control, **context}
            events.append(event)
            qualified = qualifications[(context["event_id"], control)]
            for factor in (1.0, 1.1):
                scores.append({
                    **event, "event_service_limit_factor": factor,
                    "service_qualified": qualified,
                    "qualified_positive_event_response": qualified,
                    "qualified_positive_full_window_response": qualified,
                    "event_mean_power_reduction_kw": power[control],
                    "precondition_event_recovery_net_reduction_kwh": (
                        power[control] / 2
                    ),
                })
            for hour in range(3):
                recovery.append({
                    "run_id": run_id, "event_id": context["event_id"],
                    "control": control, "action_id": context["action_id"],
                    "audit_hour_index": hour,
                    "overlaps_later_event_window": False,
                    "operative_temperature_max_abs_difference_c": 0.01 * (hour + 1),
                    "facility_electricity_kw_difference": -0.1 * (hour + 1),
                    "gas_input_kw_hhv_difference": 0.0,
                })
    paths = tuple(tmp_path / name for name in (
        "scores.csv", "recovery.csv", "events.csv", "runs.csv"
    ))
    for frame, path in zip(
        (pd.DataFrame(scores), pd.DataFrame(recovery), pd.DataFrame(events),
         pd.DataFrame(runs)),
        paths,
        strict=True,
    ):
        frame.to_csv(path, index=False)
    return paths


def test_build_summary_validates_pairs_and_writes_hash_audited_outputs(
    tmp_path: Path,
) -> None:
    score, recovery, event, run = _inputs(tmp_path)
    report = build_summary(
        score, recovery, event, run, tmp_path / "summary",
        expected_runs=6, expected_events=6, expected_factors=2,
    )

    assert report["status"] == "PASS"
    assert report["counts"]["score_rows"] == 12
    assert report["paired_definition"]["pairs_per_alternative"] == 2
    pairs = pd.read_csv(
        tmp_path / "summary" / "EVENT_RESPONSE_PAIRED_CONTROL_SUMMARY.csv"
    )
    center = pairs[
        pairs.alternative_control.eq("personal_center")
        & pairs.group_dimension.eq("overall")
    ].iloc[0]
    assert center.pair_count == 2
    assert center.alternative_only_event_qualified == 1
    assert center.uniform_only_event_qualified == 1
    assert center.alternative_minus_uniform_power_median_kw == -2.0
    assert center.alternative_qualified_power_mean_kw == pytest.approx(4.0)
    assert center.uniform_qualified_power_mean_kw == pytest.approx(5.0)
    assert center.alternative_minus_uniform_qualified_power_mean_kw == pytest.approx(-1.0)
    assert center.alternative_qualified_power_greater == 1
    assert center.uniform_qualified_power_greater == 1
    assert set(pairs.group_dimension) >= {
        "climate", "composition_id", "leave_one_climate",
        "leave_one_composition_id",
    }
    leave_warm = pairs[
        pairs.alternative_control.eq("personal_center")
        & pairs.group_dimension.eq("leave_one_climate")
        & pairs.group_value.eq("Warm")
    ].iloc[0]
    assert leave_warm.pair_count == 1
    assert leave_warm.alternative_minus_uniform_qualified_power_mean_kw == pytest.approx(-10.0)
    grouped = pd.read_csv(
        tmp_path / "summary" / "EVENT_RESPONSE_GROUP_SUMMARY.csv"
    )
    center_group = grouped[
        grouped.event_service_limit_factor.eq(1.0)
        & grouped.group_dimension.eq("control")
        & grouped.group_value.eq("personal_center")
    ].iloc[0]
    assert center_group.raw_positive_event_power_mean_kw == pytest.approx(8.0)
    assert center_group.service_qualified_event_power_mean_kw == pytest.approx(4.0)
    assert set(grouped.group_dimension) >= {"climate", "composition_id"}
    recovery_summary = pd.read_csv(
        tmp_path / "summary" / "EVENT_TERMINAL_RECOVERY_GROUP_SUMMARY.csv"
    )
    final = recovery_summary[
        recovery_summary.audit_window.eq("final_recovery_hour")
        & recovery_summary.group_dimension.eq("overall")
    ].iloc[0]
    assert final.unique_event_count == 6
    assert final.facility_electricity_kw_abs_median == pytest.approx(0.1)
    assert "equal weight" in report["paired_definition"]["mean_definition"]
    assert (tmp_path / "summary" / "EVENT_RESPONSE_SUMMARY.json").is_file()


def test_build_summary_rejects_a_run_without_final_qualification(
    tmp_path: Path,
) -> None:
    score, recovery, event, run = _inputs(tmp_path)
    runs = pd.read_csv(run)
    runs.loc[0, "final_engineering_qualified"] = False
    runs.to_csv(run, index=False)
    with pytest.raises(ValueError, match="final qualification gate"):
        build_summary(
            score, recovery, event, run, tmp_path / "summary",
            expected_runs=6, expected_events=6, expected_factors=2,
        )
