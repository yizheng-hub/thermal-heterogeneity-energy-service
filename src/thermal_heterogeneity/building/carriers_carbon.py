"""Energy-carrier, operational-carbon and affine economic accounting."""

from __future__ import annotations

import pandas as pd


def account_carriers(
    frame: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    geography: str = "city",
    electricity: str = "electricity_kwh",
    gas: str = "gas_kwh_fuel",
) -> pd.DataFrame:
    required = {geography, "electricity_kgco2_per_kwh", "gas_kgco2_per_kwh_fuel"}
    if not required.issubset(factors.columns):
        raise ValueError(f"factor table missing {sorted(required - set(factors.columns))}")
    output = frame.merge(factors, on=geography, how="left", validate="many_to_one")
    if output[["electricity_kgco2_per_kwh", "gas_kgco2_per_kwh_fuel"]].isna().any().any():
        raise ValueError("missing carbon factor after geography join")
    output["operational_co2_kg"] = (
        output[electricity] * output["electricity_kgco2_per_kwh"]
        + output[gas] * output["gas_kgco2_per_kwh_fuel"]
    )
    return output


def affine_cost_boundary(
    electricity_change_kwh: float,
    gas_change_kwh: float,
    electricity_price: float,
    gas_price: float,
) -> float:
    """Signed annual operating-cost change before management resources."""

    return electricity_change_kwh * electricity_price + gas_change_kwh * gas_price


def management_headroom(energy_cost_change: float) -> float:
    """Maximum extra management expenditure consistent with net cost savings."""

    return max(0.0, -energy_cost_change)
