"""Bit-exact storage against shared normal histories, without shortening runs."""

import hashlib
import os
import sqlite3
import tempfile
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def retain_sql_metadata(path):
    """Keep SQL metadata after required hourly values have been exported exactly."""
    path = Path(path)
    expected = sha(path)
    target = path.with_name("simulation_metadata.sql")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="metadata_", suffix=".pending", delete=False) as file:
        pending = Path(file.name)
    with sqlite3.connect(path) as source, sqlite3.connect(pending) as dest:
        tables = source.execute(
            "select name,sql from sqlite_master where type='table' and name not like 'sqlite_%'"
        ).fetchall()
        counts = {}
        for name, schema in tables:
            quoted = '"' + name.replace('"', '""') + '"'
            dest.execute(schema)
            count = source.execute("select count(*) from " + quoted).fetchone()[0]
            counts[name] = count
            if name == "ReportData":
                continue
            cursor = source.execute("select * from " + quoted)
            marks = ",".join("?" for _ in cursor.description)
            while rows := cursor.fetchmany(10000):
                dest.executemany("insert into " + quoted + " values(" + marks + ")", rows)
            cursor.close()
            assert dest.execute("select count(*) from " + quoted).fetchone()[0] == count
        dest.commit()
    source.close()
    dest.close()
    pending.replace(target)
    info = {
        "source_sql_sha256": expected,
        "sql_metadata_path": str(target),
        "sql_metadata_sha256": sha(target),
        "source_table_row_counts": counts,
        "raw_reportdata_retained": False,
        "hourly_primary_data": "All required operative temperatures,occupancy,energy meters and full air/humidity/surface states exported as float64 NPZ; other legacy report variables can be regenerated from locked IDF.",
    }
    try:
        path.unlink()
        info["source_sql_cleanup_deferred"] = False
    except PermissionError:
        info["source_sql_cleanup_deferred"] = True
    return info


def save_states_against(path, state, base_path):
    base_path = Path(base_path)
    with np.load(base_path, allow_pickle=False) as saved:
        assert "values" in saved and np.array_equal(state["keys"], saved["keys"])
        a = np.ascontiguousarray(state["values"], dtype=np.float64)
        b = np.ascontiguousarray(saved["values"], dtype=np.float64)
    assert a.shape == b.shape
    delta = np.bitwise_xor(a.view(np.uint64), b.view(np.uint64))
    assert np.array_equal(np.bitwise_xor(delta, b.view(np.uint64)).view(np.float64), a)
    np.savez_compressed(
        path,
        keys=state["keys"],
        xor_uint64=delta,
        base_path=np.array(os.path.relpath(base_path, Path(path).parent)),
        base_sha256=np.array(sha(base_path)),
    )


def load_states(path):
    with np.load(path, allow_pickle=False) as saved:
        if "values" in saved:
            return dict(saved)
        base = Path(str(saved["base_path"]))
        if not base.is_absolute():
            base = Path(path).parent / base
        assert sha(base) == str(saved["base_sha256"])
        with np.load(base, allow_pickle=False) as normal:
            value = np.bitwise_xor(
                saved["xor_uint64"], np.ascontiguousarray(normal["values"]).view(np.uint64)
            ).view(np.float64)
        return {"keys": saved["keys"].copy(), "values": value}
