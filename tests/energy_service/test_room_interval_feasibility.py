import unittest

from room_interval_feasibility import room_interval_feasibility


class RoomIntervalTest(unittest.TestCase):
    def test_conflicting_shared_room_intervals(self):
        row = room_interval_feasibility([0, 0], [21, 25], [23, 27])[0]
        self.assertFalse(row["all_profile_intersection_exists"])
        self.assertEqual(row["max_simultaneously_covered_profiles"], 1)

    def test_touching_boundaries_allow_both(self):
        row = room_interval_feasibility([0, 0], [21, 23], [23, 27])[0]
        self.assertTrue(row["all_profile_intersection_exists"])
        self.assertEqual(row["max_simultaneously_covered_profiles"], 2)

    def test_existing_zones_are_not_merged_by_control_binding(self):
        rows = room_interval_feasibility([0, 1], [21, 25], [23, 27])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["all_profile_intersection_exists"] for row in rows))


if __name__ == "__main__":
    unittest.main()
