# Candidate support and implementation headroom

This analysis reselects the frozen retargeted-control candidates under three
declared action supports. It changes neither building parameters nor simulated
trajectories. Inputs include all 7,740 candidate evaluations, including their
engineering admission flags, and 180 fixed service anchors.

The primary support gives uniform control its finite target grid and gives each
alternative control family its four specified variants. The equal-four support
uses four variants for every family. The nested support additionally permits
both alternative families to select any uniform option. All three supports use
the same mean and fixed-tail limits and minimum purchased space-HVAC energy
objective. No rejected candidate can be selected.

Run from any working directory:

```text
python reproduce_support.py --input-root inputs --output-root results
```

`TableS18.csv` reports both default-matched three-coefficient classifications and
all-definition matched comparisons. These have different denominators.
`support_selections.csv` retains all 6,480 rows, including unavailable choices.
The three-coefficient stability test uses 1e-6 percentage points; the
all-definition signs use the archived accounting tolerance of 1e-8 percentage
points. These are descriptive repeated design comparisons, not independent
field experiments or population prevalence estimates.

Energy-bill headroom uses the original nominal US electricity and HHV gas price
inputs in `regional_factors.csv`. It is the uniform energy charge minus the
alternative charge. A light implementation package annualizes eight initial
manager hours over five years at 3% and adds four manager hours annually, at
US$51.28 per hour. Its annual cost is US$294.697779. New Delhi is excluded from
this USD count because the paper retains its separate INR gas-price boundary.
Minimum total purchased energy does not imply minimum energy charge when the
electricity/gas mix changes.

The minimized inputs remove local paths, source subject identifiers and raw
human records. `INPUT_MANIFEST.json` records source and minimized checksums.
An independent chief-review implementation reproduced all 6,480 selections;
the 432 primary USD bill headrooms also match the frozen accounting ledger
within US$1e-6. These comparisons validate the portable recalculation.
