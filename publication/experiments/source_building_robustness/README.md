# Source-building-preserving robustness experiment

This extension tests whether room-mean demand qualification still omits full-demand mismatch when all 267 archive units in a scenario are sampled from one measured source building. It uses the frozen annual and July action libraries and performs no new EnergyPlus run or building-physics change.

## Run

```powershell
python run_robustness.py `
  --data-root <paper-data-root> `
  --input-root <source-building-extension-root> `
  --output-root <output-directory>
```

`--data-root` is the paper data directory containing `building/`, `population/`, and `expected/`. `--input-root` contains the protocol and support CSV files at its root plus `hourly_inputs/` with 416 NPZ files. `INPUT_INDEX_PORTABLE.csv` is preferred; the local provenance index is accepted as a fallback, but only the basename of each cache path is used.

The program first recomputes one complete frozen group: 56 annual actions and 6 July actions for the multinational reference, Miami, seed 2026102001, both λ values, and both demand representations. New outcome scoring starts only if all scores are within 1e-8 °C and all fixed-tail identities, eligibility counts, and final selections match the frozen references.

## Locked design

- 10 eligible measured source buildings, with no result-based deletion.
- 3 fixed composition seeds and 267 units sampled without replacement per source building and seed.
- 4 frozen building-weather contexts.
- λ = 0 and 1, four nested service budgets, and mean/full demand representations.
- 960 annual and 960 July design-cell selections.
- One UK/Midland building contains 300 known-ID units; its results are reported separately. The other nine buildings contain unknown-ID record units and are not described as confirmed people.
- Four source buildings use operative-temperature records and six use air-temperature fallback; both strata remain explicit.

## Main outputs

| File | Expected content |
|---|---|
| `REPRODUCTION_PRECONDITION.json` | Frozen-score and selection reproduction gate |
| `SOURCE_BUILDING_SAMPLES.csv.gz` | 8,010 sampled source-building/seed archive-unit rows |
| `SOURCE_BUILDING_ASSIGNMENTS.csv.gz` | 32,040 deterministic zone assignments |
| `ANNUAL_SCORES.csv.gz` | 22,520 action × cohort × λ × representation scores |
| `ANNUAL_CANDIDATE_ELIGIBILITY.csv.gz` | 90,080 action × cohort × λ × representation × budget rows |
| `ANNUAL_DECISIONS.csv` | 960 annual design-cell selections |
| `JULY_SCORES.csv.gz` | 2,720 event-action × cohort × λ × representation scores |
| `JULY_CANDIDATE_ELIGIBILITY.csv.gz` | 10,880 event-action × cohort × λ × representation × budget rows |
| `JULY_DECISIONS.csv` | 960 July design-cell selections |
| `TABLE_S19_SOURCE_BUILDING_ROBUSTNESS.csv` | Per-source Table S19 values, including λ=1 results |
| `METRIC_AND_ID_STRATA_SUMMARY.csv` | Operative/air-fallback and known/unknown-ID strata |
| `QA.json` and `RUN_MANIFEST.json` | Checks, input hashes, output hashes, and local provenance |

The gzip outputs use a fixed timestamp so clean runs can be compared byte for byte. `QA.json` and `RUN_MANIFEST.json` contain local absolute paths and are excluded from cross-path byte-identity contracts.

## Interpretation boundary

These are correlated design-grid evaluations on a finite frozen action library. They are not independent prevalence observations, local workforce estimates, or an observed simultaneous intervention. The source-building restriction tests sensitivity to archive mixing; it does not remove selection bias within each source study.
