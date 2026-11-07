"""Prepare the study reference input from the user-supplied DOE/PNNL prototype."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import re


SOURCE_SHA256 = "96cce0363333ef67b5b277d0ddc2526938e8fe4e05c146f484c8c7ec8bcba04e"
REFERENCE_SHA256 = "c0740b1ed872b2a5a0550be6d205a44013752219a4eb05359e4cc4ac0cccf516"


def parse_objects(content: bytes) -> list[list[str]]:
    text = re.sub(r"!.*", "", content.decode("utf-8-sig"))
    return [[value.strip() for value in block.split(",")]
            for block in text.split(";") if block.strip()]


def reference_schedule(schedule: list[str]) -> list[str]:
    """Apply the frozen +2.5 C reference shift during the control windows."""
    result = schedule[:4]
    index = 4
    while index < len(schedule):
        day_type = schedule[index]
        index += 1
        entries = []
        while index < len(schedule) and not schedule[index].lower().startswith("for"):
            entries.append((schedule[index], float(schedule[index + 1])))
            index += 2
        result.append(day_type)
        weekday = "weekdays" in day_type.lower()
        saturday = "saturday" in day_type.lower()
        if weekday or saturday:
            endpoints = [(int(until.split(":")[1]), value) for until, value in entries]
            for hour in range(1, 25):
                value = next(value for end, value in endpoints if hour <= end)
                # The setback variant has unoccupied breaks at these three hours.
                occupied_break = schedule[1].lower().endswith("_w_sb") and hour in (11, 13, 15)
                if 7 <= hour <= (22 if weekday else 18) and not occupied_break:
                    value += 2.5
                result.extend([f"Until: {hour:02d}:00", f"{value:.8f}"])
        else:
            for until, value in entries:
                result.extend([until, f"{value:.8f}"])
    return result


def reference_bytes(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Expected the ASHRAE 90.1-2019 OfficeMedium Miami 22.1 prototype.")
    objects = parse_objects(source)
    lookup = {(obj[0].lower(), obj[1].lower()): obj for obj in objects if len(obj) > 1}
    thermostats = [obj for obj in objects if obj[0].lower() == "zonecontrol:thermostat"]
    schedule_names = {obj[field].lower() for obj in objects
                      if obj[0].lower() == "thermostatsetpoint:dualsetpoint"
                      for field in (2, 3)}
    for name in schedule_names:
        schedule = lookup["schedule:compact", name]
        schedule[:] = reference_schedule(schedule)

    source_actuators = {obj[2].lower(): obj[1] for obj in objects
                        if obj[0].lower() == "energymanagementsystem:actuator"
                        and obj[3].lower() == "schedule:compact"}
    independent_actuators: dict[str, list[str]] = {}
    for thermostat in thermostats:
        zone = thermostat[2]
        dual = lookup["thermostatsetpoint:dualsetpoint", thermostat[5].lower()]
        for field, mode in ((2, "heating"), (3, "cooling")):
            original_name = dual[field].lower()
            schedule = deepcopy(lookup["schedule:compact", original_name])
            schedule[1] = f"{zone} independent {mode}"
            actuator = f"{zone}_independent_{field}_actuator"
            objects.append(schedule)
            objects.append(["EnergyManagementSystem:Actuator", actuator, schedule[1],
                            "Schedule:Compact", "Schedule Value"])
            independent_actuators.setdefault(source_actuators[original_name], []).append(actuator)
            dual[field] = schedule[1]

    for obj in objects:
        if obj[0].lower() != "energymanagementsystem:program":
            continue
        if obj[1] in ("pacu_vav_bot_MAX_OA_FRAC", "pacu_vav_mid_MAX_OA_FRAC",
                      "pacu_vav_top_MAX_OA_FRAC"):
            obj[2:] = [f"SET {obj[1]} = NULL"]
        elif obj[1] == "OA_BASED_OPTIMUM_START_EMS_Program":
            body = []
            for line in obj[2:]:
                body.append(line)
                match = re.fullmatch(r"SET (\S+) = (.+)", line)
                if match:
                    for actuator in independent_actuators.get(match[1], []):
                        body.append(f"SET {actuator} = {match[2]}")
            obj[2:] = body

    objects.append(["Output:SQLite", "SimpleAndTabular"])
    for name in ("Zone Mean Air Temperature", "Zone Operative Temperature",
                 "Zone People Occupant Count", "Zone Thermostat Heating Setpoint Temperature",
                 "Zone Thermostat Cooling Setpoint Temperature",
                 "Zone Mechanical Ventilation Standard Density Volume Flow Rate",
                 "Zone Target Voz Ventilation Flow Rate",
                 "Zone Heating Setpoint Not Met While Occupied Time",
                 "Zone Cooling Setpoint Not Met While Occupied Time"):
        objects.append(["Output:Variable", "*", name, "Hourly"])
    for name in ("Electricity:Facility", "Electricity:HVAC", "Electricity:Plant",
                 "Heating:Electricity", "Cooling:Electricity", "Fans:Electricity",
                 "Pumps:Electricity", "Heating:NaturalGas"):
        objects.append(["Output:Meter", name, "Hourly"])
    objects.extend([
        ["Output:Variable", "*", "Transformer Distribution Electricity Loss Energy", "Hourly"],
        ["Output:Meter", "Transformer:ExteriorEquipment:Electricity", "Hourly"],
        ["Output:Variable", "*", "Air System Outdoor Air Mass Flow Rate", "Hourly"],
        ["Output:Variable", "*",
         "Air System Outdoor Air Mechanical Ventilation Requested Mass Flow Rate", "Hourly"],
    ])
    content = ("\r\n".join(",".join(obj) + ";" for obj in objects) + "\r\n").encode("utf-8")
    if hashlib.sha256(content).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Prepared input does not match the study reference model.")
    return content


def prepare_reference_model(source_idf: Path, output: Path) -> None:
    content = reference_bytes(source_idf.read_bytes())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-idf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare_reference_model(args.source_idf, args.output)
    print(f"Prepared reference model: {args.output}")


if __name__ == "__main__":
    main()
