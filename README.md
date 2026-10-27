# Human thermal heterogeneity: computational source

This repository contains the analysis and figure code for **Human thermal heterogeneity reshapes building energy, carbon and flexibility**. It compares three controls under fixed building physics, weather, equipment and physical occupancy loads:

- **Uniform average** uses one population-average temperature target at the permitted command scale.
- **Personal centre** minimizes person-hour-weighted absolute temperature mismatch.
- **Personal range** minimizes distance outside personal ranges estimated with the published elicitation method used by the study.

The common 25 °C operation is an external service reference. It is not an internal fallback candidate for the three controls. Annual controller candidates use five declared offsets: −1, −0.5, 0, +0.5 and +1 °C.

Controller targets are optimized only within the declared 20–27 °C probe domain; the analysis does not infer optima beyond those bounds. `target_domain_sensitivity.py` reports boundary contact and repeats common-service and matched-service comparisons using only simulated candidates whose zone targets all lie within 21–26 °C. It also recomputes the offset-zero optima on the 21–26 °C subset of the same cached probes, without rerunning EnergyPlus.

```console
uv run --frozen python src/analysis/global_extension/target_domain_sensitivity.py --help
```

The repository is source-only. Manuscripts, Word-generation code, licensed building and weather assets, participant-level source archives, retained simulation trajectories and generated outputs are outside Git.

## Reproduction modes

Use Python 3.11.15 and the frozen environment:

```console
uv sync --frozen --extra test --extra notebook
uv run --frozen python reproduce.py --plan
```

The root entry point uses the final nine-climate evidence only:

```console
# Verify retained final tables, hashes and declared denominators
uv run --frozen python reproduce.py --mode verify

# Rebuild the final story and implementation tables, then verify them
uv run --frozen python reproduce.py --mode reassemble

# Six main figures and 21 supplementary figures from existing result tables
uv run --frozen python reproduce.py --mode figures

# Reassembly followed by all figures
uv run --frozen python reproduce.py --mode full
```

`--mode analysis` reserves the complete compact-trajectory rescoring DAG. It
currently stops with a clear error because the portable, checksum-indexed
annual, seasonal, spatial and event trajectory bundle has not yet been
declared. The `--plan` output lists that DAG without presenting final-table
reassembly as a full EnergyPlus or trajectory reconstruction.

On Windows, the equivalent wrapper is:

```powershell
.\reproduce.ps1 --mode full
```

To redraw selected figures without rerunning analysis:

```console
uv run --frozen python reproduce.py --mode figures --figure 2 5 S8 S17
```

`plot/Plot.ipynb` contains one executable cell for each publication figure. Run its setup cell once, edit the relevant figure cell, and run that cell to change one figure. The command-line renderer executes the same cells without storing notebook outputs.

## Required local inputs

Verification and reassembly expect one checksum-indexed package at
`data/inputs/final_analysis/global_extension/`. It contains the retained final
annual, seasonal, spatial, response, personal-range and information-compression
tables used by the nine-climate story. A different location can be supplied
with `--global-input-root`. In the complete local project, the entry point also
detects the same structure under `results/global_extension/`.

Reassembly reads retained numerical tables and does not run EnergyPlus. A full
trajectory reconstruction additionally requires portable manifests plus the
annual, seasonal, spatial and event compact trajectories, annual SQL evidence,
profiles, commands and protocol files. Until that separately distributed
bundle has a public relative-path manifest, `--mode analysis` remains disabled
rather than silently replaying the older four-context workflow.

These computational inputs are excluded from Git. The public ASHRAE Database II and CTCD archive are the upstream observational sources. The upstream rebuild tools verify their declared files and regenerate the aligned numerical demand inputs. Regenerating building trajectories additionally requires the declared EnergyPlus 22.1 runtime, prototype models and weather files under their original terms.

Final figure rendering expects Times New Roman to be installed on the host system; this affects typography only, not the numerical tables.

Reassembly produces the final publication tables in:

```text
results/global_extension/final_story/
results/global_extension/implementation_value_final/
```

`FINAL_STORY_HASHES.json` records the SHA-256 hashes of every input and output
used in the nine-climate annual, seasonal, spatial and multi-event story. The
entry point also enforces the declared 810 annual candidates, 216 seasonal
pairs, 516 new spatial runs, 972 spatial effects, 72 events, 810 response
trajectories and 6,480 strict event-action evaluations. Main Figures 2–5 will
not render if this manifest or any manifested output is missing or changed.

## Output structure

| Directory | Contents |
| --- | --- |
| `src/` | Analysis methods, control construction, simulation interfaces and release checks |
| `data/` | Local external inputs plus tracked checksum metadata |
| `output/` | Recomputed analyses, logs, run manifests and temporary intermediate products |
| `results/` | Figure-ready numerical tables and source-data indexes |
| `plot/` | `Plot.ipynb`, command-line renderer and generated figures |
| `tests/` | Numerical, contract, packaging and synthetic smoke tests |

The final figures are written to `plot/main/` and `plot/supplementary/`. The sibling `../manuscript/` directory contains the working Word files; it is neither read nor written by the reproduction entry point.

## Final analysis design

The shared annual design uses one fixed Medium Office prototype across nine weather contexts spanning eight climate zones. Six fixed 267-profile demand compositions, three controls and five declared offsets produce 810 annual candidates under the same building physics and native-zone permission. Separate spatial experiments compare three occupant placements and building, floor and native-zone command permissions without changing the annual-design denominator.

Two annual comparisons answer different questions:

- **Direct operation** measures how service and resources change together when each controller runs its own selected target.
- **Same-service selection** compares resource use when every controller meets the same annual population and fixed-tail service limits.

Seasonal tables retain spring, summer, autumn and winter changes before annual aggregation. Spatial tables separate occupant placement from command permission.

The response analysis has two declared baselines:

- **Common baseline:** all demand representations select from the same physical response library built from a common 25 °C operation. Building-mean, zone-mean and full-distribution information are reviewed against the same event and 48-hour recovery service limits.
- **Own baseline:** each control begins from its offset-zero annual native-zone operation. A fixed protocol selects typical- and high-load occupied events in each local meteorological season. Five declared actions cover uniform shifting, preconditioning and service-protected zone control; every result includes the two-hour event, any two-hour preconditioning period and the following 48-hour recovery.

Implementation tables combine signed electricity and gas changes with declared price and management-resource scenarios. They do not assign an unobserved demand-response revenue.

## Upstream and simulation rebuilds

The upstream demand tools accept the original public archive files supplied separately:

```console
uv run --frozen python src/utilities/generate_demand_inputs.py --help
uv run --frozen python src/analysis/targeted_extension/ctcd_alignment.py --help
```

EnergyPlus regeneration is optional for users who have the licensed/local assets:

```console
uv run --frozen python src/analysis/targeted_extension/run_demand_aware_energyplus.py --help
uv run --frozen python src/analysis/targeted_extension/common_baseline_dr.py --help
uv run --frozen python src/analysis/targeted_extension/run_own_baseline_dr.py --help
```

The final global reassembler performs no simulation and changes no
qualification rule:

```console
uv run --frozen python src/analysis/global_extension/reproduce_global_extension.py \
  --mode reassemble \
  --input-root data/inputs/final_analysis/global_extension \
  --output-root results/global_extension
```

## Verification

Run the source and numerical checks before release:

```console
uv run ruff check .
uv run pytest -q
uv run python reproduce.py --mode full --plan
uv run python src/utilities/check_release_layout.py --root .
uv run python src/utilities/release_scan.py --root . --mode licence-privacy
uv run python src/utilities/release_scan.py --root . --mode paths-identifiers
uv run python src/utilities/release_scan.py --root . --mode portability
```

`src/provenance/PUBLIC_FILE_ALLOWLIST.txt` defines the exact Git contents. The deny-by-default `.gitignore` and release checks exclude manuscripts, generated figures, output tables, external datasets, EnergyPlus assets, archives and credentials. Source code is BSD-3-Clause. Author-created documentation and eligible derived source-data products are CC BY 4.0; third-party materials retain their own terms.
