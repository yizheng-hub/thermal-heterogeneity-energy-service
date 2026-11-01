"""Test an exogenous weather-driven supply reset on the original VAV hardware."""

from standard_supply_reset import main


def coordination(objects):
    result = []
    count = 0
    for obj in objects:
        if obj[0].lower() == "setpointmanager:scheduled" and obj[1].endswith("SAT setpoint"):
            result.append(
                [
                    "SetpointManager:OutdoorAirReset",
                    obj[1] + " weather reset",
                    "Temperature",
                    "15.56",
                    "10.0",
                    "12.8",
                    "26.0",
                    obj[4],
                ]
            )
            count += 1
        else:
            result.append(obj)
    assert count == 3
    return result


if __name__ == "__main__":
    main("vav_coordinated_outdoor_reset", coordination, "vav_outdoor_reset")
