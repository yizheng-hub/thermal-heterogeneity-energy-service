"""Cross-runtime baseline verification preserves exact discrete decisions."""

import csv
import hashlib
import itertools
import math

import pytest

from baseline_decision_contract import KEYS, SERVICE, decision_sha256, rows, verify_baseline


def fixture(tmp_path):
    records = []
    for index, key in enumerate(itertools.product(range(4), range(3), range(3), range(3), range(3))):
        item = dict(zip(KEYS, [f"context{key[0]}", key[1], f"layout{key[2]}", f"scope{key[3]}", f"control{key[4]}"]))
        item.update(selection_status="selected", run_id=f"{index:032x}", variant=3)
        item.update({name: 1.25 for name in SERVICE})
        records.append(item)
    source = tmp_path / "baselines.csv"
    write(source, records)
    parsed = rows(source)
    generation = {"baseline_selection_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    contract = {"source_baseline_csv_sha256": generation["baseline_selection_sha256"],
                "decision_sha256": decision_sha256(parsed),
                "service_references": [{name: float(row[name]) for name in SERVICE} for row in parsed]}
    return source, records, generation, contract


def write(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_exact_pre_event_csv_passes(tmp_path):
    source, _, generation, contract = fixture(tmp_path)
    result = verify_baseline(source, generation, contract)
    assert result["baseline_csv_byte_identical"]
    assert result["baseline_decisions_verified"] == 324


def test_only_rounding_bits_can_differ(tmp_path):
    source, records, generation, contract = fixture(tmp_path)
    records[0]["mean_budget_c"] = math.nextafter(1.25, math.inf)
    write(source, records)
    result = verify_baseline(source, generation, contract)
    assert not result["baseline_csv_byte_identical"]
    assert result["canonical_baseline_decisions_identical"]
    assert result["baseline_service_max_difference"] < 1e-9


@pytest.mark.parametrize("field,value", [("run_id", "changed"),
                                         ("selection_status", "service_unavailable"),
                                         ("variant", 2)])
def test_changed_run_state_or_variant_is_rejected(tmp_path, field, value):
    source, records, generation, contract = fixture(tmp_path)
    records[0][field] = value
    write(source, records)
    with pytest.raises(ValueError, match="identifiers, states or variants"):
        verify_baseline(source, generation, contract)


def test_service_value_changes_are_rejected_at_existing_tolerance(tmp_path):
    source, records, generation, contract = fixture(tmp_path)
    records[0]["fixed_tail_budget_c"] += 1e-7
    write(source, records)
    with pytest.raises(ValueError, match="existing tolerance"):
        verify_baseline(source, generation, contract)


def test_reference_order_is_stable_across_csv_row_order(tmp_path):
    source, records, generation, contract = fixture(tmp_path)
    write(source, records[::-1])
    assert verify_baseline(source, generation, contract)["canonical_baseline_decisions_identical"]
