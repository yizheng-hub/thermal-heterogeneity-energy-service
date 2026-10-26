from analysis.targeted_extension.common_baseline_dr import action_library, split_action_block


def test_schedule_block_has_precondition_and_event_shifts():
    fields = ["For: Weekdays", "Until: 24:00", "25"]
    result = split_action_block(fields, -0.5, 1.0)
    assert result == [
        "For: Weekdays",
        "Until: 12:00", "25.0",
        "Until: 14:00", "24.5",
        "Until: 16:00", "26.0",
        "Until: 24:00", "25.0",
    ]


def test_native_zone_library_contains_building_actions_and_bounded_extensions():
    actions = action_library("Example", ["z1", "z2"])
    building = {row["action_id"] for row in actions if row["permission"] == "building"}
    zone = {row["action_id"] for row in actions if row["permission"] == "native_zone"}
    assert building <= zone
    assert len(building) == 6
    assert len(zone) == 14
