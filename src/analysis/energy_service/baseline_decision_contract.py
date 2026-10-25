"""Verify own-baseline identity across harmless floating-point CSV differences.

The original pre-event CSV checksum remains immutable. A derived sidecar binds
its exact 324 decisions and service numbers; the existing 1e-9 service tolerance
is retained. Changed identifiers, variants, states or service values are rejected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


KEYS = ["context", "composition_id", "layout", "permission", "control"]
DECISIONS = KEYS + ["selection_status", "run_id", "variant"]
SERVICE = ["service_fraction", "mean_budget_c", "fixed_tail_budget_c",
           "mean_reference_mismatch_c", "fixed_tail_reference_mismatch_c"]
SERVICE_TOL = 1e-9


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path, expected_rows=324):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for name in ("composition_id", "variant"):
            if row[name] and not float(row[name]).is_integer():
                raise ValueError("Baseline composition and variant must be integer-valued")
            row[name] = int(float(row[name])) if row[name] else ""
    if len(result) != expected_rows or len({tuple(row[k] for k in KEYS) for row in result}) != expected_rows:
        raise ValueError(f"Expected exactly {expected_rows} unique own-baseline decisions")
    return sorted(result, key=lambda row: tuple(row[k] for k in KEYS))


def decision_sha256(records):
    decisions = [{name: row[name] for name in DECISIONS} for row in records]
    return hashlib.sha256(json.dumps(decisions, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def verify_baseline(path, generation, contract, expected_rows=324):
    if contract["source_baseline_csv_sha256"] != generation["baseline_selection_sha256"]:
        raise ValueError("Derived decision contract does not bind the original pre-event evidence")
    current = rows(path, expected_rows)
    if decision_sha256(current) != contract["decision_sha256"]:
        raise ValueError("Own-baseline run identifiers, states or variants changed")
    if len(contract["service_references"]) != len(current):
        raise ValueError("Incomplete own-baseline service references")
    differences = [abs(float(row[name]) - reference[name])
                   for row, reference in zip(current, contract["service_references"])
                   for name in SERVICE]
    if not all(math.isfinite(value) for value in differences):
        raise ValueError("Own-baseline service values must remain finite")
    maximum = max(differences)
    if maximum > SERVICE_TOL:
        raise ValueError("Own-baseline service values changed beyond the existing tolerance")
    return {"baseline_csv_byte_identical": sha256(path) == generation["baseline_selection_sha256"],
            "canonical_baseline_decisions_identical": True, "baseline_decisions_verified": len(current),
            "baseline_service_max_difference": maximum, "baseline_service_tolerance": SERVICE_TOL,
            "original_pre_event_csv_sha256": generation["baseline_selection_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=324)
    args = parser.parse_args()
    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    source_sha = sha256(args.baseline)
    if source_sha != generation["baseline_selection_sha256"]:
        raise ValueError("Derive the sidecar only from the unchanged original pre-event baseline CSV")
    source = rows(args.baseline, args.expected_rows)
    result = {"source_baseline_csv_sha256": source_sha,
              "source_event_generation_sha256": sha256(args.generation),
              "derived_after_generation_from_verified_original": True,
              "decision_sha256": decision_sha256(source), "decision_rows": len(source),
              "service_tolerance": SERVICE_TOL,
              "service_references": [{name: float(row[name]) for name in SERVICE} for row in source]}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "service_references"}, indent=2))


if __name__ == "__main__":
    main()
