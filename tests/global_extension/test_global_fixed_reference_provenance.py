import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "metadata" / "global_fixed_reference_weather_sources.csv"
PROVENANCE_PATH = (
    ROOT / "data" / "metadata" / "global_fixed_reference_inputs.provenance.json"
)
CONFIG_PATH = ROOT / "src" / "analysis" / "global_extension" / "GLOBAL_CLIMATES.json"
WEATHER_DIR = ROOT / "data" / "upstream" / "global_climate_matrix" / "weather"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weather_rows() -> list[dict[str, str]]:
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _idf_object_counts(path: Path) -> Counter[str]:
    uncommented = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        uncommented.append(line.split("!", 1)[0])
    objects = Counter()
    for block in "\n".join(uncommented).split(";"):
        object_type = block.strip().split(",", 1)[0].strip().lower()
        if object_type:
            objects[object_type] += 1
    return objects


def test_weather_metadata_is_complete_and_matches_global_config() -> None:
    rows = _weather_rows()
    assert len(rows) == 9
    assert len({row["city"] for row in rows}) == 9
    assert len({row["local_file"] for row in rows}) == 9
    assert all(len(row["sha256"]) == 64 for row in rows)
    assert all(row["acquisition_url"].startswith("https://") for row in rows)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["climates"]
    configured = {
        row["climate"]: (row["weather"], row["climate_zone"]) for row in config
    }
    recorded = {
        row["city"]: (row["local_file"], row["ashrae_climate_zone"])
        for row in rows
    }
    assert recorded == configured
    assert recorded["Malmo"][1] == "5A"
    assert recorded["Tunis"][1] == "2A"
    assert recorded["Sydney"][1] == "3A"


def test_metadata_hash_and_optional_local_weather_hashes() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert _sha256(CSV_PATH) == provenance["weather_metadata"]["csv_sha256"]
    for row in _weather_rows():
        local_path = WEATHER_DIR / row["local_file"]
        if local_path.exists():
            assert _sha256(local_path) == row["sha256"]
            location = local_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[0].split(",")
            assert location[0] == "LOCATION"
            assert abs(float(location[6]) - float(row["latitude"])) < 1e-6
            assert abs(float(location[7]) - float(row["longitude"])) < 1e-6


def test_optional_local_prototype_matches_recorded_identity() -> None:
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    prototype = provenance["prototype"]
    fixed_path = ROOT / prototype["fixed_reference_idf"]
    official_path = ROOT / prototype["official_extracted_idf"]
    if not fixed_path.exists() or not official_path.exists():
        return

    assert _sha256(fixed_path) == prototype["fixed_reference_idf_sha256"]
    assert _sha256(official_path) == prototype["official_extracted_idf_sha256"]

    fixed = _idf_object_counts(fixed_path)
    official = _idf_object_counts(official_path)
    expected = {
        "zone": 18,
        "buildingsurface:detailed": 79,
        "fenestrationsurface:detailed": 24,
        "airloophvac": 3,
        "fan:variablevolume": 3,
        "coil:cooling:dx:twospeed": 3,
        "coil:heating:fuel": 3,
        "airterminal:singleduct:vav:reheat": 15,
        "coil:heating:electric": 15,
    }
    for object_type, count in expected.items():
        assert fixed[object_type] == count
        assert official[object_type] == count
