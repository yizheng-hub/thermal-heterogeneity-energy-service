"""Compare scientific CSVs while retaining exact discrete decisions.

Service and fraction comparisons retain the existing 1e-9 absolute tolerance.
Purchased-resource sums use the declared 1e-6 absolute / 1e-9 relative energy
tie scale. Runtime and path provenance JSONs are reported separately and are
never substituted for a comparison of the scientific numerical products.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = ("annual", "events", "costs", "portfolio", "nested_permissions", "common_layout", "dense_uniform_bound")
DISCRETE = {"composition_id", "variant", "service_fraction", "amplitude_c", "event_month", "event_day",
            "hour_index", "hour_from_event_start", "source_row_index"}


def compare(reference, calculated):
    records = []
    for group in GROUPS:
        expected = {p.name: p for p in (reference / group).glob("*.csv*")}
        actual = {p.name: p for p in (calculated / group).glob("*.csv*")}
        if set(expected) != set(actual):
            raise ValueError(f"Scientific file set differs: {group}")
        for name, path in sorted(expected.items()):
            other = actual[name]
            a = pd.read_csv(path, keep_default_na=False, dtype=str)
            b = pd.read_csv(other, keep_default_na=False, dtype=str)
            if a.shape != b.shape or not a.columns.equals(b.columns):
                raise ValueError(f"Scientific schema or row count differs: {group}/{name}")
            differences = []
            for field in a:
                left, right = a[field], b[field]
                if left.equals(right):
                    continue
                if not left.eq("").equals(right.eq("")):
                    raise ValueError(f"Scientific missing-value states differ: {group}/{name}:{field}")
                nonempty = left.ne("")
                try:
                    x = pd.to_numeric(left[nonempty], errors="raise").to_numpy(dtype=float)
                    y = pd.to_numeric(right[nonempty], errors="raise").to_numpy(dtype=float)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"Discrete scientific values differ: {group}/{name}:{field}") from error
                if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                    raise ValueError(f"Scientific finite states differ: {group}/{name}:{field}")
                finite = np.isfinite(x)
                if not np.array_equal(x[~finite], y[~finite], equal_nan=True):
                    raise ValueError(f"Scientific nonfinite values differ: {group}/{name}:{field}")
                delta = np.abs(x[finite] - y[finite])
                if field in DISCRETE or field.endswith("_count") or field.endswith("_candidates"):
                    tolerance = np.zeros(len(delta))
                elif field.endswith("_kw") or "_kw_" in field:
                    tolerance = np.full(len(delta), 1e-9)
                elif any(token in field.lower() for token in ("kwh", "co2", "usd", "currency", "inr")):
                    tolerance = np.maximum(1e-6, 1e-9 * np.abs(x[finite]))
                else:
                    tolerance = np.full(len(delta), 1e-9)
                if np.any(delta > tolerance):
                    raise ValueError(f"Scientific numerical values differ beyond the existing scale: {group}/{name}:{field}")
                differences.append({"column": field, "maximum_absolute_difference": float(delta.max(initial=0)),
                                    "discrete_values_unchanged": field in DISCRETE,
                                    "allowed_absolute_scale": "declared resource tie scale" if tolerance.max(initial=0) > 1e-9 else "1e-9 or exact"})
            byte_same = path.read_bytes() == other.read_bytes()
            records.append({"path": f"{group}/{name}", "rows": len(a), "columns": len(a.columns),
                            "byte_identical": byte_same, "scientific_values_match": True,
                            "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "calculated_sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
                            "floating_column_differences": differences})
    return {"status": "PASS", "scientific_files_compared": len(records),
            "byte_identical_files": sum(row["byte_identical"] for row in records),
            "all_scientific_values_match": True, "all_discrete_decisions_match": True,
            "numerical_products_ignored": 0,
            "provenance_json_excluded_from_csv_comparison": True, "files": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--calculated-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(args.reference_root, args.calculated_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
