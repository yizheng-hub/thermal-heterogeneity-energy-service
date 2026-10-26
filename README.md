# Human thermal heterogeneity: computational source

This repository contains the analysis and figure code for **Human thermal heterogeneity reshapes building energy, carbon and flexibility**. It compares three controls under fixed building physics, weather, equipment and physical occupancy loads:

- **Uniform average** uses one population-average temperature target at the permitted command scale.
- **Personal centre** minimizes person-hour-weighted absolute temperature mismatch.
- **Personal range** minimizes distance outside model-generated personal temperature ranges.

The common 25 °C operation is an external service reference. It is not an internal fallback candidate for the three controls. Annual controller candidates use five declared offsets: −1, −0.5, 0, +0.5 and +1 °C.

The repository is source-only. Manuscripts, Word-generation code, licensed building and weather assets, participant-level source archives, retained simulation trajectories and generated outputs are outside Git.

## Reproduction modes

Use Python 3.11.15 and the frozen environment:

```console
uv sync --frozen --extra test --extra notebook
uv run --frozen python reproduce.py --plan
```

The root entry point has three modes:

```console
# Numerical analyses, final evidence tables and cost scenarios
uv run --frozen python reproduce.py --mode analysis

# Six main figures and 21 supplementary figures from existing result tables
uv run --frozen python reproduce.py --mode figures

# Analysis followed by all figures
uv run --frozen python reproduce.py --mode full
```

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

The analysis mode expects three separately supplied, checksum-indexed computational inputs:

1. `data/inputs/` contains the harmonized demand inputs and retained annual/event calculations used by the base replay, including the aligned ASHRAE and CTCD cohorts.
2. `data/typical_days/` contains the retained response and recovery trajectories.
3. `data/inputs/final_analysis/Reviewer_repair/` contains the final retained annual and event trajectories and their primary and strict analysis tables. A different location can be passed with `--final-analysis-root`.

In the complete local project, the third package may already be present at `results/Reviewer_repair/`; the entry point detects that location automatically. The source-only release expects it under `data/inputs/final_analysis/Reviewer_repair/` after the separately distributed data package is unpacked.

These computational inputs are excluded from Git. The public ASHRAE Database II and CTCD archive are the upstream observational sources. The upstream rebuild tools verify their declared files and regenerate the aligned numerical demand inputs. Regenerating building trajectories additionally requires the declared EnergyPlus 22.1 runtime, prototype models and weather files under their original terms.

Final figure rendering expects Times New Roman to be installed on the host system; this affects typography only, not the numerical tables.

The complete analysis produces the final tables in:

```text
results/final_submission/evidence/
results/final_submission/costs/
```

`FINAL_EVIDENCE_QA.json` records the denominators, headline counts and SHA-256 hashes of every table used in the final evidence assembly.

## Output structure

| Directory | Contents |
| --- | --- |
| `src/` | Analysis methods, control construction, simulation interfaces and release checks |
| `data/` | Local external inputs plus tracked checksum metadata |
| `output/` | Recomputed analyses, logs and run manifests |
| `results/` | Figure-ready numerical tables and source-data indexes |
| `plot/` | `Plot.ipynb`, command-line renderer and generated figures |
| `tests/` | Numerical, contract, packaging and synthetic smoke tests |
| `intermediate_files/` | Local temporary products excluded from Git |

The final figures are written to `plot/main/` and `plot/supplementary/`. The sibling `../manuscript/` directory contains the working Word files; it is neither read nor written by the reproduction entry point.

## Final analysis design

The shared annual design uses three controls, four representative office-weather contexts, common random-mixing and demand-ordered layouts, and building, floor and native-zone command permissions. Primary engineering qualification includes isolated HVAC iteration warnings when the physical and accounting checks pass; the strict sensitivity excludes those warnings.

Two annual comparisons answer different questions:

- **Direct operation** measures how service and resources change together when each controller runs its own selected target.
- **Same-service selection** compares resource use when every controller meets the same annual population and fixed-tail service limits.

Seasonal tables retain spring, summer, autumn and winter changes before annual aggregation. Spatial tables separate occupant placement from command permission.

The response analysis has two declared baselines:

- **Common baseline:** all demand representations select from the same physical response library built from a common 25 °C operation. Building-mean, zone-mean and full-distribution information are reviewed against the same event and 48-hour recovery service limits.
- **Own baseline:** each control begins from its direct annual native-zone operation. The same 0.5 and 1 °C response actions are evaluated in February and July through the 48-hour recovery period.

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

The final evidence assembler performs no simulation and changes no qualification rule:

```console
uv run --frozen python src/analysis/targeted_extension/assemble_final_evidence.py \
  --results-root data/inputs/final_analysis/Reviewer_repair \
  --output results/final_submission/evidence
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
