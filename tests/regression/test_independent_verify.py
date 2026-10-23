import json
from pathlib import Path

import pandas as pd

from thermal_heterogeneity.io import sha256
from thermal_heterogeneity.verify import verify_manifest


def test_independent_claim_check_reads_regenerated_table(tmp_path: Path):
    table = tmp_path / "table.csv"
    pd.DataFrame({"count": [3]}).to_csv(table, index=False)
    manifest = {
        "scientific_files": [{"path": "table.csv", "sha256": sha256(table)}],
        "claim_checks": [
            {
                "claim_id": "three",
                "path": "table.csv",
                "reducer": "value",
                "column": "count",
                "expected": 3,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    result = verify_manifest(path)
    assert result["passed"]
    assert result["checked_claims"] == 1
