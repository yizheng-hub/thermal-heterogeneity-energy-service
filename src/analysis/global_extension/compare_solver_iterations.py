"""Compare cached probe trajectories produced with two solver iteration limits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ("climate", "target_c")
REQUIRED_ARRAYS = (
    "zone_ids", "temperatures", "people", "occupied",
    "hvac_electric_kwh", "gas_hhv_kwh",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        data = {name: np.asarray(saved[name]) for name in saved.files}
    missing = set(REQUIRED_ARRAYS) - set(data)
    if missing:
        raise ValueError(f"Compact trajectory lacks {sorted(missing)}: {path}")
    return data


def _relative_difference(first: float, second: float) -> float:
    denominator = abs(first)
    if denominator <= 1e-12:
        return 0.0 if abs(second) <= 1e-12 else float("inf")
    return abs(second - first) / denominator


def compare_probe_manifests(
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    maximum_temperature_difference_c: float = 0.01,
    maximum_energy_relative_difference: float = 0.001,
) -> pd.DataFrame:
    """Return an auditable row-by-row comparison without changing either run."""
    required = {*KEYS, "hourly_path"}
    for label, frame in (("first", first), ("second", second)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{label} manifest lacks {sorted(missing)}")
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"{label} manifest duplicates a probe key")
    joined = first.merge(
        second, on=list(KEYS), how="outer", suffixes=("_first", "_second"),
        indicator=True, validate="one_to_one",
    )
    if not joined._merge.eq("both").all():
        missing = joined.loc[joined._merge.ne("both"), [*KEYS, "_merge"]]
        raise ValueError(f"Solver manifests do not contain the same probes:\n{missing}")
    if maximum_temperature_difference_c <= 0:
        raise ValueError("Temperature tolerance must be positive")
    if maximum_energy_relative_difference <= 0:
        raise ValueError("Energy tolerance must be positive")

    rows: list[dict] = []
    for item in joined.to_dict("records"):
        first_data = _load(Path(item["hourly_path_first"]))
        second_data = _load(Path(item["hourly_path_second"]))
        if not np.array_equal(first_data["zone_ids"], second_data["zone_ids"]):
            raise ValueError(f"Zone identities changed for {item['climate']}/{item['target_c']}")
        if not np.array_equal(first_data["occupied"], second_data["occupied"]):
            raise ValueError(f"Occupancy clock changed for {item['climate']}/{item['target_c']}")
        if not np.array_equal(first_data["people"], second_data["people"]):
            raise ValueError(f"Native occupancy changed for {item['climate']}/{item['target_c']}")
        temperature_difference = np.abs(
            first_data["temperatures"].astype(float)
            - second_data["temperatures"].astype(float)
        )
        if temperature_difference.size == 0 or not np.isfinite(temperature_difference).all():
            raise ValueError("Temperature comparison is empty or non-finite")
        first_electricity = float(first_data["hvac_electric_kwh"].sum())
        second_electricity = float(second_data["hvac_electric_kwh"].sum())
        first_gas = float(first_data["gas_hhv_kwh"].sum())
        second_gas = float(second_data["gas_hhv_kwh"].sum())
        first_total = first_electricity + first_gas
        second_total = second_electricity + second_gas
        maximum_temperature = float(temperature_difference.max())
        electricity_relative = _relative_difference(
            first_electricity, second_electricity
        )
        gas_relative = _relative_difference(first_gas, second_gas)
        total_relative = _relative_difference(first_total, second_total)
        rows.append({
            "climate": item["climate"],
            "target_c": float(item["target_c"]),
            "first_maximum_hvac_iterations": item.get(
                "maximum_hvac_iterations_first", np.nan
            ),
            "second_maximum_hvac_iterations": item.get(
                "maximum_hvac_iterations_second", np.nan
            ),
            "operative_temperature_max_abs_difference_c": maximum_temperature,
            "operative_temperature_p99_abs_difference_c": float(
                np.quantile(temperature_difference, 0.99)
            ),
            "hvac_electricity_relative_difference": electricity_relative,
            "gas_hhv_relative_difference": gas_relative,
            "total_purchased_energy_relative_difference": total_relative,
            "trajectory_stable_within_declared_tolerance": bool(
                maximum_temperature <= maximum_temperature_difference_c
                and electricity_relative <= maximum_energy_relative_difference
                and gas_relative <= maximum_energy_relative_difference
                and total_relative <= maximum_energy_relative_difference
            ),
            "declared_temperature_tolerance_c": maximum_temperature_difference_c,
            "declared_energy_relative_tolerance": maximum_energy_relative_difference,
            "first_treatment_warning_profile": item.get(
                "treatment_warning_profile_first", ""
            ),
            "second_treatment_warning_profile": item.get(
                "treatment_warning_profile_second", ""
            ),
        })
    return pd.DataFrame(rows).sort_values(list(KEYS), kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-manifest", type=Path, required=True)
    parser.add_argument("--second-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temperature-tolerance-c", type=float, default=0.01)
    parser.add_argument("--energy-relative-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    result = compare_probe_manifests(
        pd.read_csv(args.first_manifest),
        pd.read_csv(args.second_manifest),
        maximum_temperature_difference_c=args.temperature_tolerance_c,
        maximum_energy_relative_difference=args.energy_relative_tolerance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
