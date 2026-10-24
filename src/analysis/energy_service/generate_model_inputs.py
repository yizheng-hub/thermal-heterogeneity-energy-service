"""Build annual and event IDFs from locally supplied frozen complete models.

The four source IDFs and weather files retain their original source terms.
This program changes only the declared thermostat schedules and verifies every
generated normalized IDF against the final engineering manifest. No executable
or third-party model is bundled with this source module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def objects(text):
    return [[value.strip() for value in part.split(",")]
            for part in re.sub(r"!.*", "", text).split(";") if part.strip()]


def signature(text):
    return hashlib.sha256("\n".join(",".join(x) + ";" for x in objects(text)).encode()).hexdigest()


def physical_hash(text):
    parsed = objects(text)
    schedules = {x[i] for x in parsed if x[0] == "ThermostatSetpoint:DualSetpoint" for i in (2, 3)}
    retained = [x for x in parsed if not x[0].startswith("Output")
                and not (x[0] == "Schedule:Compact" and x[1] in schedules)]
    return hashlib.sha256(json.dumps(retained).encode()).hexdigest()


def annual_input(text, targets, half):
    parsed = objects(text)
    duals = {x[1]: x for x in parsed if x[0] == "ThermostatSetpoint:DualSetpoint"}
    controlled = {}
    for thermostat in [x for x in parsed if x[0] == "ZoneControl:Thermostat"]:
        for index in (2, 3):
            controlled[duals[thermostat[5]][index]] = (
                targets[thermostat[2]], 25 + (1 if index == 3 else -1) * half)
    for obj in parsed:
        if obj[0] != "Schedule:Compact" or obj[1] not in controlled:
            continue
        target, old = controlled[obj[1]]
        eligible = False
        for index in range(3, len(obj)):
            if obj[index].lower().startswith("for:"):
                eligible = any(day in obj[index].split(":", 1)[1].lower().split()
                               for day in ("weekdays", "saturday"))
            elif (eligible and obj[index - 1].lower().startswith("until:")
                  and abs(float(obj[index]) - old) < 1e-7):
                obj[index] = str(float(obj[index]) + target - 25)
    return "\n".join(",".join(x) + ";" for x in parsed)


def split_block(fields, delta):
    result, previous, eligible = [], 0, False
    i = 0
    while i < len(fields):
        token = fields[i]
        if token.lower().startswith("for"):
            result.append(token)
            previous = 0
            eligible = any(x in token.lower() for x in ("weekdays", "saturday"))
            i += 1
        elif token.lower().startswith("until:"):
            hour, minute = map(int, token.split(":", 1)[1].strip().split(":"))
            end = hour * 60 + minute
            value = float(fields[i + 1])
            cuts = sorted({previous, end} | {v for v in (840, 960) if previous < v < end})
            for start, stop in zip(cuts[:-1], cuts[1:]):
                shift = delta if eligible and start >= 840 and stop <= 960 else 0.
                result.extend([f"Until: {stop // 60:02d}:{stop % 60:02d}", str(value + shift)])
            previous = end
            i += 2
        else:
            raise ValueError(f"Unsupported compact schedule token: {token}")
    return result


def event_input(text, events, amplitude):
    parsed = objects(text)
    names = {x[i] for x in parsed if x[0] == "ThermostatSetpoint:DualSetpoint" for i in (2, 3)}
    for obj in parsed:
        if obj[0] != "Schedule:Compact" or obj[1] not in names:
            continue
        if obj[3].lower() != "through: 12/31" or sum(v.lower().startswith("through:") for v in obj) != 1:
            raise ValueError("Expected a single original annual thermostat block")
        original = obj[4:]
        expanded = obj[:3]
        previous = date(2001, 1, 1)
        for event in sorted(events, key=lambda e: (e["month"], e["day"])):
            current = date(2001, event["month"], event["day"])
            if previous < current:
                end = current - timedelta(days=1)
                expanded += [f"Through: {end.month}/{end.day}"] + original
            expanded += [f"Through: {current.month}/{current.day}"] + split_block(
                original, event["sign"] * amplitude)
            previous = current + timedelta(days=1)
        if previous <= date(2001, 12, 31):
            expanded += ["Through: 12/31"] + original
        obj[:] = expanded
    return "\n".join(",".join(x) + ";" for x in parsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--commands", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    models = json.loads((args.model_root / "MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    generated = json.loads(args.commands.read_text(encoding="utf-8"))
    keys = ["context", "composition_id", "layout", "permission", "control", "variant"]
    commands = {tuple(row[k] for k in keys): row["targets_c"] for row in generated}
    annual = pd.read_csv(args.input_root / "engineering/ANNUAL_RUN_MANIFEST.csv", keep_default_na=False)
    events = pd.read_csv(args.input_root / "engineering/EVENT_RUN_MANIFEST.csv", keep_default_na=False)
    dates = json.loads((args.input_root / "engineering/EVENT_DATES_BEFORE_RUNS.json").read_text(encoding="utf-8"))
    texts, physics = {}, {}
    args.output_root.mkdir(parents=True, exist_ok=False)
    for context, cfg in models.items():
        path = args.model_root / cfg["baseline_idf"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["baseline_idf_sha256"]:
            raise ValueError(f"Frozen complete model bytes changed: {context}")
        weather = args.model_root / cfg["weather"]
        if hashlib.sha256(weather.read_bytes()).hexdigest() != cfg["weather_sha256"]:
            raise ValueError(f"Frozen weather bytes changed: {context}")
        texts[context] = path.read_text(encoding="utf-8")
        physics[context] = physical_hash(texts[context])
    unique = {}
    for row in annual.to_dict("records"):
        context = row["context"]
        text = annual_input(texts[context], commands[tuple(row[k] for k in keys)],
                            models[context]["half_deadband_c"])
        if signature(text) != row["normalized_idf_sha256"] or physical_hash(text) != physics[context]:
            raise ValueError("Generated annual model differs from locked physical input")
        unique[row["run_id"]] = text
    for row in events.to_dict("records"):
        if row["engineering_status"] == "unavailable_baseline":
            continue
        text = event_input(unique[row["baseline_run_id"]], dates[row["context"]], row["amplitude_c"])
        if signature(text) != row["normalized_idf_sha256"] or physical_hash(text) != physics[row["context"]]:
            raise ValueError("Generated event model differs from locked physical input")
        unique[row["run_id"]] = text
    for run_id, text in unique.items():
        (args.output_root / f"{run_id}.idf").write_bytes(text.encode())
    report = {"status": "PASS", "annual_logical_inputs": len(annual), "event_logical_inputs": len(events),
              "unique_physical_inputs": len(unique), "all_normalized_inputs_match": True,
              "all_noncontrol_physical_fingerprints_unchanged": True,
              "historical_source_module_required": False, "physical_simulations_run": 0,
              "runtime_identity": {c: cfg["runtime_sha256"] for c, cfg in models.items()},
              "redistribution_rights_asserted": False}
    (args.output_root / "MODEL_GENERATION_QA.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
