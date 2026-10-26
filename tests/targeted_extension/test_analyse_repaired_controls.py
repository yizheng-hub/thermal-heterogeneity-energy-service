import pandas as pd

from analysis.targeted_extension.analyse_repaired_controls import choose_energy


def test_choose_energy_is_deterministic_at_tie():
    rows = pd.DataFrame(
        {
            "annual_site_kwh": [10.0, 10.0, 11.0],
            "run_id": ["b", "a", "c"],
            "variant": [0, 1, 2],
        }
    )
    assert choose_energy(rows)["run_id"] == "a"
