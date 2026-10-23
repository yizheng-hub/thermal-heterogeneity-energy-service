"""Independent small cases for the service accounting, including native weights."""

import unittest

import numpy as np

from thermal_service import score_service


class ThermalServiceTest(unittest.TestCase):
    def test_opposing_references_do_not_cancel(self):
        result = score_service([[24]], [[2]], [0, 0], [22, 26], [21, 25], [23, 27], [1])
        summary = result["summary"]
        self.assertEqual(summary["mean_reference_mismatch_c"], 2)
        self.assertEqual(summary["fixed_tail_reference_mismatch_c"], 2)
        self.assertEqual(summary["modeled_interval_person_hour_coverage"], 0)
        self.assertEqual(summary["modeled_interval_excess_native_person_c_hours"], 2)

    def test_native_occupancy_weights_are_not_profile_counts(self):
        result = score_service([[24, 26]], [[10, 2]], [0, 1, 1], [24, 24, 24], [23]*3, [25]*3, [1, 2])
        summary = result["summary"]
        self.assertAlmostEqual(summary["mean_reference_mismatch_c"], 1/3)
        self.assertAlmostEqual(summary["fixed_tail_reference_mismatch_c"], 2)
        self.assertAlmostEqual(summary["modeled_interval_person_hour_coverage"], 5/6)
        self.assertEqual(summary["occupied_native_person_hours"], 12)
        self.assertEqual(summary["modeled_interval_excess_native_person_c_hours"], 2)

    def test_unoccupied_excursion_does_not_reduce_service(self):
        result = score_service([[24], [40]], [[2], [0]], [0], [24], [23], [25], [0])
        self.assertEqual(result["summary"]["modeled_interval_person_hour_coverage"], 1)
        self.assertTrue(result["summary"]["all_profiles_without_interval_violation"])
        self.assertTrue(np.isnan(result["hourly"]["modeled_interval_coverage"][1]))

    def test_fixed_tail_preserves_equal_profile_definition(self):
        result = score_service([[24, 26]], [[10, 2]], [0, 1, 1], [24]*3, [23]*3, [25]*3, [0, 1])
        self.assertEqual(result["summary"]["fixed_tail_reference_mismatch_c"], 1)

    def test_interval_boundaries_are_inclusive_and_time_is_explicit(self):
        result = score_service([[23], [25], [26]], [[2], [2], [2]], [0], [24], [23], [25], [0], 0.5)
        self.assertAlmostEqual(result["summary"]["modeled_interval_person_hour_coverage"], 2/3)
        self.assertEqual(result["summary"]["modeled_interval_excess_native_person_c_hours"], 1)
        self.assertEqual(result["summary"]["occupied_native_person_hours"], 3)


if __name__ == "__main__":
    unittest.main()
