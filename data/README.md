# Data boundary

This directory contains release metadata, not raw human or third-party source files.

The current complete replay uses an externally supplied **877-file** package with
`legacy_analysis`, `source_building` and `energy_service` components. Verify it
against `paper_bundle_manifest.csv` and `paper_bundle_source.json`, then pass its
unpacked directory to `ctl/reproduce.py --data-root`. The archive is supplied
separately. A permanent public download address is not asserted here. Model,
weather and original survey/scene files are separate external dependencies.
The repository README distinguishes the complete replay, model-input generation
and optional upstream human-demand reconstruction.

The sections below retain component-specific field, provenance and archive
verification details. Their older component counts are not counts of the
complete 877-file package.

- `source_manifest.csv` records the locked decision for all 28 asset classes.
- `analysis_bundle_manifest.csv` records byte hashes, row counts, keys and provenance for the 46 minimized Level-1 products generated under `staging/data_v3/`.
- `analysis_bundle_release.json` binds the current minimized products to one immutable tar+zstd archive hash. Its publication locator remains pending until the archive is deposited.
- `FIELD_CODEBOOK.csv` documents every released CSV field.
- `TRANSFORM_PROVENANCE.csv` links each release product to a relative frozen-source locator and both source and release SHA-256 hashes.
- `source_building_bundle_manifest.csv` and `source_building_bundle_source.json`
  bind the separate robustness extension to its immutable archive and 425
  constituent files. The original 46-file Level-1 bundle is unchanged.

The regional holdout table includes three identity-free sampling ranks, and
`implementation/local_update_sampling_plan.csv.gz` freezes the exact first-40
feedback draws used by the locked local-update experiment. Reproduction uses
that plan directly; it does not regenerate a random stream from anonymized
group identifiers. Raw group/unit identifiers, random-number seeds and an ID
crosswalk are not released.

Run `scripts/materialize_release_data.py` to rebuild the products and `scripts/validate_release_data.py` to verify the schema, privacy, paths, keys, source hashes and release hashes. Build and test the immutable archive with:

```console
uv run python scripts/build_analysis_bundle.py --input staging/data_v3 --output staging/analysis_bundle.tar.zst
set THERMAL_HETEROGENEITY_ANALYSIS_BUNDLE_URL=<HTTPS-OR-LOCAL-ARCHIVE>
uv run python scripts/fetch_data.py --tier analysis --manifest data/source_manifest.csv --dest clean_data
uv run python scripts/verify_data_manifest.py --root clean_data --manifest data/analysis_bundle_manifest.csv
```

The fetch command verifies the archive hash before extraction, rejects links and path traversal, then verifies the exact 46-file set, every file hash and size, and every declared table row count, column count and primary key. The panel-support JSON contains aggregate method locks, intervals and anonymized window statistics needed for deterministic panel-data v2 materialization. `staging/`, fetched data and the archive itself are generated local products and remain outside Git. Raw inputs, identity crosswalks, exact participant times, exact coordinates and local machine paths remain outside the release products.

Unpack the separate source-building robustness archive with:

```console
uv run python scripts/unpack_source_building_data.py --bundle source_building_robustness.tar.zst --dest source_building_data
uv run python scripts/unpack_source_building_data.py --root source_building_data
```

The unpacker rejects unexpected entries, links and path traversal, then verifies
every declared byte count and checksum. Reproduction uses 416 hourly NumPy
arrays, minimized executable demand units, cohort metric/identifier strata,
fixed candidate supports, zone capacities and a protocol locked before outcome
scoring. These are frozen trajectories and archive-derived counterfactual
cohorts. Original human records and identity crosswalks are not distributed.

## Typical-day temporal supplement

The final replay additionally verifies `typical_day_dr_bundle_manifest.csv` and
`typical_day_dr_bundle_source.json`. They identify a separate externally supplied
ZIP with 44 declared files (32 normalized full-year physical trajectories, four
source calendars and eight locked protocol records), plus its manifest. Pass the
unpacked directory with `--typical-day-data-root`. These are retained outputs
from new physical simulations; the public analysis replay rescores them and
does not claim to regenerate their full EnergyPlus input-generation history.
No trajectory, manuscript or data ZIP is stored in this code repository.
