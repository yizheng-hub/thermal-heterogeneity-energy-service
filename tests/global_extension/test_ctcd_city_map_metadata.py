import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "metadata" / "ctcd_city_map_metadata.csv"
PROVENANCE_PATH = (
    ROOT / "data" / "metadata" / "ctcd_city_map_metadata.provenance.json"
)


def test_ctcd_city_map_metadata_has_declared_scopes() -> None:
    cities = pd.read_csv(CSV_PATH)
    assert len(cities) == 11
    assert cities.city_normalized.is_unique
    assert cities.geonames_id.is_unique
    assert cities.metadata_version.eq("1.0.0").all()
    assert cities.overlap_screened_office_records.sum() == 14_725
    assert cities.active_conditioning_records.sum() == 12_427
    assert cities.executable_profiles.sum() == 12_404
    assert cities.latitude.between(-90, 90).all()
    assert cities.longitude.between(-180, 180).all()


def test_ctcd_city_map_metadata_preserves_scope_boundaries() -> None:
    cities = pd.read_csv(CSV_PATH).set_index("city_normalized")
    assert cities.loc["xian", "display_name"] == "Xi'an"
    assert cities.loc["yantai", "overlap_screened_office_records"] == 1
    assert cities.loc["yantai", "active_conditioning_records"] == 0
    assert cities.loc["yantai", "executable_profiles"] == 0
    assert (
        cities.active_conditioning_records
        <= cities.overlap_screened_office_records
    ).all()
    assert (cities.executable_profiles <= cities.active_conditioning_records).all()


def test_ctcd_city_map_provenance_matches_csv() -> None:
    cities = pd.read_csv(CSV_PATH)
    provenance = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    totals = provenance["scope_totals"]
    assert provenance["metadata_version"] == cities.metadata_version.iat[0]
    assert totals["overlap_screened_office_records"] == (
        cities.overlap_screened_office_records.sum()
    )
    assert totals["active_conditioning_records"] == cities.active_conditioning_records.sum()
    assert totals["executable_profiles"] == cities.executable_profiles.sum()
    assert provenance["dataset_citation"]["doi"] == "10.1038/s41597-023-02568-3"
    assert provenance["terminology"]["required_short_name"] == "CTCD"
