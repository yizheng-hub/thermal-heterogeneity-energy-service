from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


@pytest.fixture
def synthetic_release(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame(
        {
            "unit": ["u1", "u1", "u2", "u3"],
            "building": ["b1", "b1", "b1", "b2"],
            "city": ["c1", "c1", "c1", "c2"],
            "country": ["k1", "k1", "k1", "k2"],
            "reference_center_c": [24.0, 25.0, 26.0, 27.0],
        }
    ).to_csv(data / "profiles.csv", index=False)
    pd.DataFrame(
        {
            "window": [1, 2],
            "compressed_burden": [0.0, 0.5],
            "full_burden": [1.0, 0.5],
            "jensen_gap": [1.0, 0.0],
        }
    ).to_csv(data / "windows.csv", index=False)
    pd.DataFrame(
        [
            {"city": "c1", "service_fraction": 0.9, "delta": delta, "lambda": spread, "value": 10 + delta + 2 * spread + delta * spread}
            for delta in (0.0, 1.0)
            for spread in (0.0, 1.0)
        ]
    ).to_csv(data / "effects.csv", index=False)
    annual = []
    for representation in ("compressed", "full"):
        annual.extend(
            [
                {"city": "c1", "seed": 1, "service_fraction": 0.9, "representation": representation, "physical_action_id": "a", "site_kwh": 10.0, "mean_c": 0.4 if representation == "compressed" else 1.2, "fixed_tail_c": 0.4 if representation == "compressed" else 1.2, "mean_budget_c": 1.0, "fixed_tail_budget_c": 1.0},
                {"city": "c1", "seed": 1, "service_fraction": 0.9, "representation": representation, "physical_action_id": "b", "site_kwh": 12.0, "mean_c": 0.5, "fixed_tail_c": 0.5, "mean_budget_c": 1.0, "fixed_tail_budget_c": 1.0},
            ]
        )
    pd.DataFrame(annual).to_csv(data / "annual.csv", index=False)
    dynamic = []
    for representation in ("compressed", "full"):
        for action, response in (("x", 5.0), ("y", 3.0)):
            bad = representation == "full" and action == "x"
            row = {"city": "c1", "seed": 1, "month": 7, "service_fraction": 0.9, "representation": representation, "event_action_id": action, "event_average_reduction_kw": response, "event_mean_budget_c": 1.0, "event_fixed_tail_budget_c": 1.0, "recovery_mean_budget_c": 1.0, "recovery_fixed_tail_budget_c": 1.0}
            for period in ("event", "recovery"):
                for trajectory in ("baseline", "perturbed"):
                    row[f"{period}_{trajectory}_mean_c"] = 1.2 if bad else 0.5
                    row[f"{period}_{trajectory}_fixed_tail_c"] = 1.2 if bad else 0.5
            dynamic.append(row)
    pd.DataFrame(dynamic).to_csv(data / "dynamic.csv", index=False)
    pd.DataFrame(
        {"mean_available": [True, True, False], "full_available": [True, True, False], "selection_matches": [True, False, False]}
    ).to_csv(data / "phase.csv", index=False)
    pd.DataFrame(
        {"certified_eligible": [True, False, False], "full_service_eligible": [True, True, False]}
    ).to_csv(data / "cert.csv", index=False)
    pd.DataFrame(
        {"split": ["city", "city", "source", "source"], "feedback_n": [0, 40, 0, 40], "crps": [0.2, 0.15, 0.21, 0.18], "tail_mae": [0.4, 0.3, 0.42, 0.35]}
    ).to_csv(data / "update.csv", index=False)

    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs" / "paper.yaml").read_text(encoding="utf-8"))
    config["output"] = "output"
    config["data_root"] = "."
    config["assets"] = {
        "human_profiles": "data/profiles.csv",
        "shared_room_windows": "data/windows.csv",
        "centre_spread_scores": "data/effects.csv",
        "annual_scores": "data/annual.csv",
        "dynamic_scores": "data/dynamic.csv",
        "phase_decisions": "data/phase.csv",
        "quantile_certification": "data/cert.csv",
        "local_update": "data/update.csv",
    }
    config["verification"] = {"claim_checks": []}
    config_path = tmp_path / "paper.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, tmp_path / "output"
