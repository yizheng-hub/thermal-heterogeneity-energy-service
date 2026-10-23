import pandas as pd

from thermal_heterogeneity.building.annual import choose_minimum_energy, qualify
from thermal_heterogeneity.building.carriers_carbon import account_carriers
from thermal_heterogeneity.building.centre_spread import decompose_two_by_two


def test_annual_selection_uses_lexical_tie_break():
    frame = pd.DataFrame({"physical_action_id": ["b", "a"], "site_kwh": [10.0, 10.0], "service_eligible": [True, True]})
    assert choose_minimum_energy(frame)["physical_action_id"] == "a"


def test_qualification_is_simultaneous_mean_and_tail():
    frame = pd.DataFrame({"mean_c": [0.5, 0.5], "fixed_tail_c": [0.5, 1.5], "mean_budget_c": [1.0, 1.0], "fixed_tail_budget_c": [1.0, 1.0]})
    assert qualify(frame).tolist() == [True, False]


def test_orthogonal_decomposition_recovers_interaction():
    frame = pd.DataFrame([{"delta": d, "lambda": s, "value": 10 + d + 2 * s + 3 * d * s} for d in (0.0, 1.0) for s in (0.0, 1.0)])
    result = decompose_two_by_two(frame, keys=[])
    row = result[(result["delta"] == 1) & (result["lambda"] == 1)].iloc[0]
    assert row["interaction"] == 3.0


def test_carbon_accounting_preserves_energy_carriers():
    data = pd.DataFrame({"city": ["x"], "electricity_kwh": [10.0], "gas_kwh_fuel": [5.0]})
    factors = pd.DataFrame({"city": ["x"], "electricity_kgco2_per_kwh": [0.4], "gas_kgco2_per_kwh_fuel": [0.2]})
    assert account_carriers(data, factors).iloc[0]["operational_co2_kg"] == 5.0
