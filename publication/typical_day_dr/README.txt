Supplementary July date-sensitivity analysis replay

This directory extends the existing publication repository. It imports the
published energy_service scorer by a repository-relative path. Python, NumPy,
pandas and the existing repository dependencies are required; plotting also
requires matplotlib and exact Times New Roman normal/bold fonts.

The separate typical_day_dr_supplement_v1.zip contains 32 normalized complete
annual hourly trajectories, four source clocks and eight small protocol files.
Its 44 declared files are locked by the repository manifest. MANIFEST.csv is an
additional archive entry. Keep the ZIP and extracted numerical inputs outside
Git, as with the existing paper data package. No IDF, weather, executable, raw
SQL, console log, original person identifier or absolute user path is included.

Verify and extract the sidecar into a new directory:

  python scripts/unpack_typical_day_dr_data.py --bundle <sidecar.zip> --dest <supplement-root>

Verify an already extracted sidecar:

  python scripts/unpack_typical_day_dr_data.py --root <supplement-root>

After completing the main publication replay, recompute the supplementary data:

  python publication/typical_day_dr/replay.py --paper-data-root <paper-data-root> --annual-root <fresh-main-output/energy_service/annual> --supplement-root <supplement-root> --output-root <new-replay-output>

The paper-data root may contain energy_service/, or may be that directory.
The CLI verifies the existing paper manifest, every consumed paper file, the
sidecar manifest and the fresh annual baseline/anchor/tail outputs. It reproduces
the source-day median/90th-percentile selection, checks the actual source weekday
calendar, then recalculates 192 candidate budget rows, 96 selections and 2,400
hourly rows for 48 logical amplitude cases. The original peak-date response is
also recomputed from the existing main package, giving a 144-row three-date
comparison. Direct weighted endpoint arithmetic independently checks the new
scores, all eight service constraints and amplitude selections.

Prelocked dates remain authoritative when their distance to the quantile target
is within 1e-12 kWh of the minimum. CSV rounding can reverse the near-equal median
distances for Miami July 20/26 and Seattle July 21/14. Replay validates the locked
date within that tolerance instead of selecting a new date from rounded ties.

Render the four-context comparison:

  python publication/typical_day_dr/render.py --replay-root <new-replay-output> --output-root <new-figure-output>

Outputs use the stable stem TYPICAL_DAY_DR: PDF/SVG, 600-dpi PNG, 300-dpi Word PNG,
the 36-row figure source CSV, objective caption and text/font geometry QA. The
caption currently identifies Supplementary Fig. 13; the final paper assembler
controls numbering. Faint dashed curves are physical +1 C electrical reductions;
solid curves are service-qualified available response among the two tested
amplitudes. Qualified no-DR baseline response zero is distinct from absence of
physical load-reduction capability. Contexts are individual simulated buildings
with separate y scales, never regional totals.

Reproduction boundary

This is an analysis replay from retained physical outputs. It runs no EnergyPlus
simulation and changes no physical parameter. The local original physical runs,
full engineering evidence and pre-run locks remain preserved separately. The
normalized sidecar records their hashes and qualification decisions, but does
not implement or claim a portable full-model input generator or physical rerun.
The original pre-run date/protocol lock is retained by hash; normalization into
this shareable sidecar occurred after the physical simulations.

The result is a bounded exploratory July date sensitivity: one composition,
random mixing, native-zone permission, three source load classes and two fixed
setpoint amplitudes. It does not establish annual DR capacity, a distribution
over all dates, local workforce representativeness or invariant controller ranks.
