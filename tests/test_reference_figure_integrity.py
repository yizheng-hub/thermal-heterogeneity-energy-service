"""Checks use the supplied numerical inputs and native-source outputs."""

import hashlib
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import ast

ROOT = Path(__file__).resolve().parents[1]

def notebook_definitions(cell_id):
    """Load a figure's visible definitions without drawing it."""
    notebook = json.loads((ROOT / "plot/Plot.ipynb").read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == cell_id)
    code = "".join(cell["source"]).split("# Read the data, draw this figure and save it.")[0]
    namespace = {"__name__": __name__}
    exec(compile(code, "Plot.ipynb:" + cell_id, "exec"), namespace)
    return namespace


def frame(name):
    from src.thermal_heterogeneity.figure_inputs import resolve
    return pd.read_csv(resolve(name))


@unittest.skipUnless((ROOT / "results/figure_data").is_dir(), "Extract Numerical_Results.zip first")
class IntegrityTests(unittest.TestCase):
    def test_supplied_hashes(self):
        records = json.loads((ROOT / "results/figure_data/index.json").read_text())
        for row in records:
            p = ROOT / row["file"]
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), row["sha256"])

    def test_observation_counts(self):
        cities = frame("CITY_MAP_POINTS.csv")
        countries = frame("COUNTRY_DEMAND_SUMMARY.csv")
        self.assertEqual(len(cities), 49)
        self.assertEqual(len(countries), 15)
        self.assertEqual(cities.record_count.sum(), 33845)
        self.assertEqual(countries.record_count.sum(), 33845)

    def test_room_counts(self):
        d = frame("SHARED_ROOM_SUMMARY.csv")
        d = d[d.minutes.eq(30)].set_index("dataset")
        np.testing.assert_array_equal(
            d.loc["NUS", ["Same", "Mixed", "Opposed"]].to_numpy(), [14, 43, 6]
        )
        np.testing.assert_array_equal(
            d.loc["ENTH", ["Same", "Mixed", "Opposed"]].to_numpy(), [12, 20, 2]
        )

    def test_operating_choices(self):
        d = frame("NOMINAL_QUALIFICATION_COUNTS.csv")
        self.assertEqual(len(d), 54)
        self.assertEqual(d.choices_differ.sum(), 24)
        self.assertEqual(d.qualified_operation_sets_differ.sum(), 54)
        self.assertLess(d.zone_mean_selected_full_service_excess_c.max(), 1)

    def test_hvac_denominators(self):
        d = frame("independent_pthp_MATCHED_CROSS_SYSTEM_EFFECTS.csv")
        d = d[d.layout.eq("within_floor_relocation_25pct")]
        self.assertEqual(len(d), 105)
        self.assertEqual(d.control.eq("personal_center").sum(), 52)
        self.assertEqual(d.control.eq("personal_range").sum(), 53)

    def test_response_means(self):
        d = frame("PAIRED_EVENT_INFORMATION_REVIEW.csv")
        d = d[
            d.control.eq("building_search")
            & d.normal_energy_reference_factor.eq(1)
            & d.policy.eq("maximum_power")
        ]
        self.assertEqual(len(d), 432)
        checked = np.where(d.mean_selected_full_service_qualified, d.mean_available_power_kw, 0.0)
        np.testing.assert_allclose(
            [d.mean_available_power_kw.mean(), checked.mean(), d.full_available_power_kw.mean()],
            [11.532, 2.494, 2.514],
            atol=0.001,
        )

    def test_native_floor_counts(self):
        namespace = notebook_definitions("main-3")
        study = namespace["main_Study"](ROOT / "results/figure_data", ROOT / "plot/main", True)
        panels, bounds, meta = namespace["main_native_floor_panels"](study)
        self.assertEqual([len(x["points"]) for x in panels], [89] * 6)
        self.assertEqual(bounds, (17.0, 32.0))
        self.assertEqual(meta["mode"], "native_CSV_and_geometry_JSON")

    def test_native_curve_provenance(self):
        meta = json.loads(
            (ROOT / "plot/main/Figure_06.json").read_text(encoding="utf-8")
        )
        self.assertEqual(meta["panel_ab"]["mode"], "native_hourly_NPZ")
        curves = pd.read_csv(
            ROOT / "plot/main/Figure_06_example_curves.csv"
        )
        self.assertTrue(curves["source_mode"].eq("native_hourly_NPZ").all())
        self.assertEqual(len(curves), 106)

    def test_final_render_checks(self):
        for i in range(1, 7):
            p = ROOT / "plot/main" / f"Figure_{i:02}.json"
            if not p.exists():
                self.skipTest("Run reproduce_publication.py first for output-level checks")
            report = json.loads(p.read_text(encoding="utf-8"))["checks"]
            self.assertTrue(all(v["passed"] for v in report.values()))
            meta = json.loads(
                (ROOT / "plot/main" / f"Figure_{i:02}.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(meta["canvas_overflow_text"], [])
            self.assertEqual(len(meta["heading_alignment"]), 4)
            self.assertTrue(all(row["single_text_artist"] for row in meta["heading_alignment"]))
            self.assertTrue(all(row["text"].startswith(f"({letter}) ")
                                for letter, row in zip("abcd", meta["heading_alignment"])))


if __name__ == "__main__":
    unittest.main()
