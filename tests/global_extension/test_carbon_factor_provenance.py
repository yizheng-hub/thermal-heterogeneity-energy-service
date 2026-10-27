import json
from pathlib import Path

import pytest

from src.analysis.global_extension.analyse_global_results import (
    DEFAULT_GAS_CO2_KG_KWH_HHV,
)


ROOT = Path(__file__).resolve().parents[2]


def test_default_gas_factor_matches_tracked_epa_provenance() -> None:
    metadata = json.loads(
        (ROOT / "data/metadata/natural_gas_co2_factor_epa_2025.json").read_text(
            encoding="utf-8"
        )
    )
    expected = (
        metadata["source_factor_kg_co2_per_mmbtu"]
        / metadata["conversion_kwh_per_mmbtu"]
    )
    assert metadata["basis"] == "higher heating value"
    assert metadata["derived_factor_kg_co2_per_kwh_hhv"] == pytest.approx(expected)
    assert DEFAULT_GAS_CO2_KG_KWH_HHV == pytest.approx(expected)
