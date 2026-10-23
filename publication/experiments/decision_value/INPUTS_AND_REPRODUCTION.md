# Frozen inputs and reproduction

This experiment re-evaluates an existing annual candidate library. It does not run EnergyPlus, change physical parameters, regenerate controls, retune service budgets or optimize a new protection threshold.

Run from the public repository's Python environment:

```powershell
python run_decision_value.py --data-root <repository>/staging/data_v3 --output-root <output-directory>
python audit_assignment_provenance.py --data-root <repository>/staging/data_v3 --output-root <output-directory>
```

The first script imports three small existing functions from the repository `src/thermal_heterogeneity`. It reads these anonymous minimized inputs:

| Input | Role |
|---|---|
| `implementation/representation_scores.csv.gz` | Full and compressed annual mismatch scores, common-25 °C anchors, fixed plan IDs and site energy |
| `building/annual_action_library.csv.gz` | The operational CO2 attached to each fixed annual physical plan |
| `population/population_assignments_minimized.csv.gz` | Anonymous profile-to-composition assignments for checking cross-country separation |
| `human/human_profiles_minimized.csv.gz` | Anonymous archive-unit and source-building identifiers for overlap assertions |

The three existing approximation rules are five bins with training q95, five bins with q90 and three bins with q90. Bins are the original equal-count groups represented by their means. Protection margins use the existing one-sided empirical higher-quantile rule. Full and protected plans use the same finite action library and the same mean and fixed-tail budgets. Site-energy ties use the original absolute/relative tolerance and lexical plan-ID rule.

Only after all calculated CSVs are written, `expected/quantile_decisions_reference.csv.gz` is read for regression checking. It is never used to select a plan or calculate a margin. All 6,048 existing reference decisions are checked. The new 2,016 country-transfer decisions have no historical reference and are evaluated under the protocol declared before their calculation.

`DECISIONS.csv` contains every one of 8,064 design-case evaluations. `CANDIDATE_EVALUATIONS.csv.gz` retains all candidate tests, including rejection and false acceptance. Missing selections remain missing and have no numeric regret. `MAIN_FIG6G_Q5_REGRET.csv` contains all 1,008 primary q5-q95 cases, including 64 protection-induced gaps and 402 cases without a full-qualified plan. Plot regret only for `status=matched_full_qualified`. Their 245 zero regrets are real matched outcomes. The relative denominator is the site energy of the full-information, energy-minimizing plan in that same design case.

The carbon difference compares the carbon of the two energy-selected plans. It is not regret from a separately carbon-optimized solution. Candidate counts are repeated evaluations over compositions, spreads and budgets, not independent physical trajectories or people.

`TABLE_S17_PROTECTED_QUANTILE_TRANSFER.csv` gives the primary holdout, each original composition-label holdout and both archive-unit-disjoint country directions. `INPUT_MANIFEST.json` hashes all inputs, the regression reference and the script. `QA.json` records equality with the existing reference and zero intersections of training/test archive units and source buildings in the two new transfer directions.

The second script independently reports the actual source-building mixing and the unit overlap in the original composition-label holdouts. Source-building labels identify where a profile was originally measured. They do not mean that a simulated zone preserves a real building's original occupant composition.
