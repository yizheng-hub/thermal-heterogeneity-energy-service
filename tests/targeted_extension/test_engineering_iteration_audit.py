from pathlib import Path
import sys


MODULE = Path(__file__).resolve().parents[2] / "src" / "analysis" / "targeted_extension"
sys.path.insert(0, str(MODULE))

from run_demand_aware_energyplus import set_maximum_hvac_iterations  # noqa: E402


def test_set_maximum_hvac_iterations_changes_only_second_field():
    source = "ConvergenceLimits, 1, 20;\nBuilding, Example;"
    assert set_maximum_hvac_iterations(source, 40) == (
        "ConvergenceLimits, 1, 40;\nBuilding, Example;"
    )


def test_set_maximum_hvac_iterations_can_be_disabled():
    source = "ConvergenceLimits,1,20;"
    assert set_maximum_hvac_iterations(source, None) == source
