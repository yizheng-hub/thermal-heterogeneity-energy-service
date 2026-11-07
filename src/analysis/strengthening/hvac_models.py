"""Construct explicit HVAC contrasts using the supplied EnergyPlus22.1 templates."""

from common import *
import re
import subprocess
import sqlite3
import pandas as pd
from global_matrix import parse_idf, serialize_idf
from run_global_matrix import ZONES

SIM = ROOT / "simulations"


def idd_fields():
    result = {}
    kind = None
    current = None
    for line in (RUNTIME.parent / "Energy+.idd").read_text(errors="replace").splitlines():
        obj = re.match(r"^([A-Za-z][\w:]*),\s*$", line)
        if obj:
            kind = obj[1].lower()
            result[kind] = []
            current = None
            continue
        field = re.match(r"^\s*[AN]\d+\s*[;,]", line)
        if field and kind:
            result[kind].append("")
            current = len(result[kind]) - 1
        name = re.search(r"\\field\s+(.+)", line)
        if name and kind and current is not None:
            result[kind][current] = name[1].strip()
    return result


def coordination(objects):
    result = []
    changed = []
    for obj in objects:
        if obj[0].lower() == "setpointmanager:scheduled" and obj[1].endswith("SAT setpoint"):
            loop = obj[1].replace(" SAT setpoint", "")
            result.append(
                [
                    "SetpointManager:Warmest",
                    obj[1] + " coordinated",
                    "Temperature",
                    loop,
                    "12.8",
                    "18.0",
                    "MaximumTemperature",
                    obj[4],
                ]
            )
            changed.append(obj[1])
        else:
            result.append(obj)
    assert len(changed) == 3
    return result


def heatpump(objects):
    example = parse_idf((RUNTIME.parent / "ExampleFiles/HVACTemplate-5ZonePTHP.idf").read_text())
    template = next(x for x in example if x[0].lower() == "hvactemplate:zone:pthp")
    fields = idd_fields()["hvactemplate:zone:pthp"]

    def setfield(obj, name, value):
        i = fields.index(name) + 1
        while len(obj) <= i:
            obj.append("")
        obj[i] = str(value)

    retained_programs = {
        o[1].lower()
        for o in objects
        if o[0].lower() == "energymanagementsystem:program"
        and (
            "light_ems_program" in o[1].lower()
            or o[1].lower() in ["transformer_load_prog", "oa_based_optimum_start_ems_program"]
        )
    }
    program_text = " ".join(
        " ".join(o).lower()
        for o in objects
        if o[0].lower() == "energymanagementsystem:program" and o[1].lower() in retained_programs
    )
    result = []
    for obj in objects:
        kind = obj[0].lower()
        name = obj[1].lower() if len(obj) > 1 else ""
        remove = kind.startswith(
            (
                "airloophvac",
                "airterminal:",
                "zonehvac:",
                "coil:",
                "fan:",
                "controller:",
                "availabilitymanager",
                "outdoorair:",
            )
        ) or kind in [
            "sizing:system",
            "sizing:zone",
            "nodeList".lower(),
            "zonecontrol:thermostat",
            "thermostatsetpoint:dualsetpoint",
        ]
        if kind == "branch" or kind == "branchlist":
            remove = "pacu" in name
        if kind.startswith("setpointmanager"):
            remove = "swh" not in name
        if kind == "energymanagementsystem:program":
            remove = name not in retained_programs
        if kind == "energymanagementsystem:programcallingmanager":
            remove = not any(x.lower() in retained_programs for x in obj[3:])
        if kind == "energymanagementsystem:actuator":
            remove = len(obj) > 3 and obj[3].lower() not in ["lights", "schedule:compact"]
        if kind == "energymanagementsystem:sensor":
            remove = not bool(re.search(r"\b" + re.escape(name) + r"\b", program_text))
        if not remove:
            result.append(obj)
    existing = {x[1] for x in objects if x[0].lower().startswith("schedule:")}
    assert "HVACOperationSchd" in existing
    schedule = "Strengthening continuous fan"
    result.append(["Schedule:Constant", schedule, "Any Number", "1"])
    for zone in ZONES:
        heating = next(
            x[1]
            for x in objects
            if x[0].lower().startswith("schedule:")
            and x[1].lower() == zone.lower() + " independent heating"
        )
        cooling = next(
            x[1]
            for x in objects
            if x[0].lower().startswith("schedule:")
            and x[1].lower() == zone.lower() + " independent cooling"
        )
        thermostat = zone + " PTHP thermostat"
        result.append(["HVACTemplate:Thermostat", thermostat, heating, "", cooling, ""])
        pthp = list(template)
        setfield(pthp, "Zone Name", zone)
        setfield(pthp, "Template Thermostat Name", thermostat)
        setfield(pthp, "Outdoor Air Method", "DetailedSpecification")
        setfield(pthp, "Outdoor Air Flow Rate per Person", "")
        setfield(pthp, "Design Specification Outdoor Air Object Name", "SZ DSOA " + zone)
        setfield(pthp, "Supply Fan Operating Mode Schedule Name", schedule)
        setfield(pthp, "System Availability Schedule Name", "HVACOperationSchd")
        result.append(pthp)
    return result


def invariant_objects(objects):
    kinds = [
        "building",
        "zone",
        "buildingsurface:detailed",
        "fenestrationsurface:detailed",
        "people",
        "lights",
        "electricequipment",
        "zoneinfiltration:designflowrate",
        "designspecification:outdoorair",
        "daylighting:controls",
        "wateruse:equipment",
    ]
    return sorted((json.dumps(o) for o in objects if o[0].lower() in kinds))


def prepare():
    pilot = json.loads((ROOT / "config/PILOT_SETTINGS.json").read_text())
    models = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())
    rows = []
    for climate in pilot["climates"]:
        source = parse_idf(Path(models[climate]["prepared_idf"]).read_text())
        for system, transform in [
            ("vav_coordinated", coordination),
            ("independent_pthp", heatpump),
        ]:
            converted = transform(source)
            assert invariant_objects(source) == invariant_objects(converted)
            folder = SIM / "hvac_models_v2" / system / climate
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "model_template.idf"
            path.write_text(serialize_idf(converted))
            if system == "independent_pthp":
                # ExpandObjects is part of the locked EnergyPlus distribution.
                (folder / "in.idf").write_bytes(path.read_bytes())
                (folder / "Energy+.idd").write_bytes((RUNTIME.parent / "Energy+.idd").read_bytes())
                result = subprocess.run(
                    [str(RUNTIME.parent / "ExpandObjects.exe")],
                    cwd=folder,
                    capture_output=True,
                    timeout=60,
                )
                (folder / "EXPAND.log").write_bytes(result.stdout + result.stderr)
                assert result.returncode == 0 and (folder / "expanded.idf").exists()
                expanded = parse_idf((folder / "expanded.idf").read_text())
                # Expander may rewrite case; compare field values case-insensitively.
                a = [x.lower() for x in invariant_objects(source)]
                b = [x.lower() for x in invariant_objects(expanded)]
                assert sorted(a) == sorted(
                    b
                ), "Expansion altered physical-load or ventilation specifications"
                path = folder / "expanded.idf"
            rows.append(
                {
                    "climate": climate,
                    "system": system,
                    "prepared_idf": str(path),
                    "prepared_sha256": digest(path),
                    "weather": models[climate]["weather_path"],
                    "state": "prepared_not_yet_validated",
                    "physical_load_invariants": "identical",
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "config/HVAC_PREPARATION.csv", index=False)
    write_json(
        ROOT / "config/HVAC_SPECIFICATION.json",
        {
            "supply_reset": "Warmest-zone supply-air reset12.8–18C, maximum-temperature strategy; three original airloops",
            "pthp": "EnergyPlus22.1 supplied PTHP template; single-speed DX cooling COP3, heating COP2.75, electric supplemental heat; original outdoor-air specs and all geometry, loads, lighting EMS retained",
            "capacity": "Must size common25C operation then explicitly freeze all equipment AUTOSIZE values before strategy comparisons",
            "validation": "Prepared only; no performance conclusions until ventilation,load,capacity and numerical checks pass",
        },
    )
    print("Prepared", len(rows), "unvalidated system-climate models", flush=True)


if __name__ == "__main__":
    prepare()
