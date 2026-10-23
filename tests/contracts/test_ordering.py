import pandas as pd

from thermal_heterogeneity.io import canonical_table_hash


def test_canonical_hash_is_input_order_invariant():
    frame = pd.DataFrame({"id": [2, 1], "value": [4.0, 3.0]})
    assert canonical_table_hash(frame, ["id"]) == canonical_table_hash(frame.iloc[::-1], ["id"])
