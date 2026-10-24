# Human thermal heterogeneity: computational source

Numerical methods, control generation, scientific comparisons, figure renderers, source-data exports and tests for **Human thermal heterogeneity reshapes building energy, carbon and flexibility**. Manuscript and supplementary prose, Word templates, completed documents, data archives, building/weather assets and private identifier crosswalks are excluded.

## Complete scientific replay

Use Python **3.11.15** with the frozen environment:

```console
uv sync --frozen --extra test
uv run python ctl/reproduce.py --plan
uv run python ctl/reproduce.py --data-root ../paper_data --typical-day-data-root ../typical_day_dr_data --output-root outputs/replay
```

`../paper_data` is an externally supplied, unpacked 877-file computational package: `legacy_analysis` (46 files), `source_building` (425), and `energy_service` (406). Archive and constituent checksums are recorded in `data/paper_bundle_source.json` and `data/paper_bundle_manifest.csv`. The archive is supplied separately. No permanent public download locator is currently asserted by this repository. To verify and unpack a supplied archive:

```console
uv run python scripts/unpack_paper_data.py --bundle ../paper_data_energy_service_20261023.tar.zst --dest ../paper_data
```

The final temporal extension also requires the separately supplied normalized typical-day supplement. Its 44 declared files contain 32 retained full-year event trajectories, four source calendars and eight locked protocol/verification records, with a separate manifest file. The 19.6-MB ZIP and all building/weather assets stay outside Git. Verify and unpack it with:

```console
uv run python scripts/unpack_typical_day_dr_data.py --bundle ../typical_day_dr_supplement_v1.zip --dest ../typical_day_dr_data
```

Choose a new output directory. The complete replay verifies the inputs, recomputes the human and three-control analyses, regenerates 1,296 annual control vectors, evaluates annual/event service, energy, carbon, synchronized portfolios and cost scenarios, and produces **six main and 24 supplementary figures**. It writes all result tables, traceable source-data CSVs and their index, the figure-to-source map, logs and `SCIENTIFIC_REPLAY_RUN.json`. Monthly and four-season service, heating/cooling carriers and carbon are recomputed from all 8,760 annual hours. Typical-day response analyses recompute the original peak-day comparison and two additional July date classes from the retained physical outputs; this replay does not regenerate those 32 EnergyPlus trajectories. No manuscript or Word template is read. No Word document is built.

Times New Roman normal and bold fonts must be installed locally to render the figures. Fonts are not redistributed. Missing fonts stop rendering. On Windows, short external data/output paths avoid legacy path-length limits.

## Organization

| Directory | Purpose |
| --- | --- |
| `src/` | Human-demand, building-result and implementation modules |
| `ctl/` | Complete scientific replay entry point |
| `plot/` | Figure redraw entry point and renderer guide |
| `data/` | Input checksums, fields, source metadata and retrieval boundaries |
| `configs/` | Frozen scientific and model-execution contracts |
| `publication/` | Retained numerical/plotting modules and small audited support tables |
| `scripts/` | Input verification, optional upstream reconstruction and release checks |
| `tests/` | Numerical, contract, regression, packaging and synthetic smoke tests |
| `outputs/`, `results/`, `intermediate_files/` | Ignored local products; only README files are tracked |

The retained `publication` paths preserve imports and provenance. They contain no manuscript prose or Word-authoring modules. Small aggregate support tables remain in source because their checksums and licensing boundaries form part of the calculation.

## Redraw all figures

After a successful replay, redraw all 30 figures and refresh source-data exports without rerunning the numerical analyses:

```console
uv run python plot/reproduce.py --data-root ../paper_data --typical-day-data-root ../typical_day_dr_data --replay-root outputs/replay
```

`plot/README.md` lists the renderers. Historical input-panel names and the `_word.png` suffix identify frozen inputs and the 300-dpi image variant. They do not require Word. Current displayed numbering and source routes are recorded in generated manifests.

## Reproduction boundaries

The main replay starts from retained operative temperatures, occupancy, calendars, thermostat commands and purchased energy carriers. It recalculates outcomes from these frozen inputs. It does **not** run EnergyPlus or recalibrate building physics. Demand profiles are references, not confirmed prototype occupants; physical gains remain unchanged. Service describes reconstructed temperature mismatch and model-derived interval coverage, not measured satisfaction. The 108-office portfolios are declared aggregation scenarios, not real city stocks.

The current model-input generator reconstructs 489 normalized annual/event inputs when four externally supplied models and weather files satisfy `data/energy_service_model_contract.json`:

```console
uv run python publication/energy_service/generate_model_inputs.py --input-root ../paper_data/energy_service --model-root ../local_models --commands outputs/replay/energy_service/controls/CONTROL_COMMANDS.json --output-root outputs/model_inputs
```

Running those models additionally requires the specified EnergyPlus 22.1 build and assets supplied under their original terms. The older supporting action library has a historical snapshot boundary in `protocol/HISTORICAL_CODE_PROVENANCE.md`. No unavailable scene-generation history is claimed to have been recovered.

The optional upstream tool reconstructs human-demand numbers from three external checksum-matched inputs:

```console
uv run python scripts/generate_demand_inputs.py --measurements ../external_inputs/measurements.csv --metadata ../external_inputs/metadata.csv --scene-pool ../external_inputs/scene_pool.json --output-root outputs/upstream_numeric
```

The required measurement, metadata and scene-pool SHA-256 hashes are enforced by the script. These assets are not distributed here. This tool does not download or reconstruct the scene pool, export question text, recreate private crosswalks or reproduce original release identifiers. It is separate from the complete replay. A different source-file hash is rejected rather than silently substituted.

An optional building-identity audit uses the checksum-matched measurements and metadata without requiring the scene pool:

```console
uv run python scripts/audit_building_identity.py --measurements ../external_inputs/measurements.csv --metadata ../external_inputs/metadata.csv --public-profiles ../paper_data/legacy_analysis/human/human_profiles_minimized.csv.gz --source-inventory ../paper_data/source_building/SOURCE_BUILDING_INVENTORY.csv --output-root outputs/building_identity
```

The metadata distinguishes 150 original-source building identifiers from 37 inferred groups in the office cohort. Building identity establishes neither room membership nor simultaneous presence. The audit recomputes 36 variance decompositions and reports identity origins using only release labels. It changes no frozen simulation input. The ten supporting source-group pools include three original and seven inferred identifiers.

## Tests and CI

After a complete local replay, run all applicable tests with actual panel fixtures:

```powershell
$env:THERMAL_HETEROGENEITY_PANEL_DATA = (Resolve-Path 'outputs/replay/supporting_evidence/panel_data').Path
uv run pytest -q
```

For POSIX, set `export THERMAL_HETEROGENEITY_PANEL_DATA="$PWD/outputs/replay/supporting_evidence/panel_data"` before the same test command. Figure tests require these inputs and installed fonts.

GitHub CI runs source/privacy checks, numerical/package contracts and the synthetic analysis smoke test on Windows and Linux. The source-only CI test command explicitly excludes three rendering test files that require local fonts and external panel inputs. It does not silently assume that the separate data archive or licensed fonts exist on a hosted runner. The complete local test and scientific replay remain separate checks.

```console
uv run ruff check .
uv run python scripts/check_release_layout.py --root .
uv run python scripts/release_scan.py --root . --mode licence-privacy
uv run python scripts/release_scan.py --root . --mode paths-identifiers
uv run python scripts/release_scan.py --root . --mode portability
```

The exact Git allowlist denies manuscript files, generated products, external assets and credentials. This source-only repository starts with new Git history. Original software is BSD-3-Clause; author-created documentation and eligible derived outputs are CC BY 4.0. Third-party materials retain their terms. See `DATA_AND_SOFTWARE_LICENSES.md` and `THIRD_PARTY_NOTICES.md`.
