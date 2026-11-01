# Reproduce from the contained project

The assembled Windows project contains the original observational inputs,
model, weather files, numeric demand specification, source code, locked Python
packages, EnergyPlus 22.1 runtime and figure fonts. It does not read a historical
assignment table or an older project's result directory to generate results.
The source-only Git repository and this complete local research package have
different contents; third-party inputs retain their original terms.

From the assembled directory, use its own launcher:

```powershell
.\run.ps1 --output-root output/from_raw --plan
.\run.ps1 --output-root output/from_raw --workers 6
```

`run.ps1` resolves all paths from its own location. Its isolated Python runtime
does not use a host Python installation, user site-packages or a virtual
environment in another directory. Times New Roman is loaded from `tools/fonts`.
Windows remains the required operating system for these bundled executables.

The default stages run in order:

1. `demand`: reconstruct record-level demand inputs and all 1,602 seeded profile
   assignments from raw archives and the physical model's occupant slots.
2. `observations`: regenerate source summaries, shared-room analyses, geographic
   figure data, document-specific policy comparisons and leave-one-source variance.
3. `matrix`: prepare the nine weather contexts, run probes, generate annual
   commands and simulate all 810 original logical annual candidates.
4. `annual`: reconstruct individual service scores and operating choices.
5. `seasonal`: derive source-season demand shifts, run seasonal commands and
   calculate monthly electricity and facility gas.
6. `spatial`: generate arrangements and command scopes, run and score them.
7. `response`: reconstruct the original event protocol and response trajectories.
8. `secondary`: reproduce the existing matching, tolerance, tail, domain,
   information, equipment and response diagnostics from the fresh trajectories.
9. `strengthening`: execute the frozen validation, candidate refinement, strong
   baselines, constrained arrangements, HVAC and response capability analyses.

The matrix entrypoint retains full hourly SQL with each operation's compact
arrays, exact input and run QA. Later equipment and ventilation diagnostics
read this SQL directly. A fresh output directory therefore creates its own
diagnostic sources; the earlier technical replay registry is not a prerequisite.

Every attempt writes commands, logs, source/input/runtime hashes and status
under the selected output directory. Completion of program execution is
separate from engineering qualification, feasibility and statistical support;
inspect the stage reports and `strengthening/PHASE_EXECUTION_STATUS.json`.
Prespecified unsuccessful engineering conditions remain in their run logs and
reports. The protocol files are in `data/metadata/strengthening_protocol`.

The original matrix has 279 logical probe operations and 810 annual operations.
The first candidate refinement adds 648 logical operations. Its declared
conditional extra level adds 1,296. Strong baselines, constrained arrangements,
alternative HVAC and response operations have separate budgets in the frozen
protocols; duplicates are reused with their logical mappings retained. These
are simulation workloads, not counts of independent observed participants.
The launcher checks free space before large simulation stages; runtime depends
on the machine and physical deduplication, and is recorded per operation.
The initial free-space check is 125 GiB when generating the raw matrix and
250 GiB when the full strengthening stage is requested. These are conservative
startup allowances for retained hourly outputs and later analyses; monitor
available space during execution.

To resume, name only the stages whose prerequisites already exist:

```powershell
.\run.ps1 --output-root output/from_raw --stages annual seasonal spatial response secondary strengthening --workers 6
```

Use a fresh output directory for a fresh raw demand rebuild. After moving a
project, run into a fresh directory so all generated operation manifests refer
to the current location. Existing outputs and logs are retained as evidence;
they are not substitutes for the raw inputs.

To verify a moved project and its reconstructed demand products:

```powershell
.\tools\python\python.exe src/utilities/verify_project_relocation.py
.\run.ps1 --output-root output/move_check --stages demand observations --workers 2
.\tools\python\python.exe src/utilities/verify_project_relocation.py --raw-rebuild output/move_check/demand
.\tools\python\python.exe src/utilities/technical_relocation_simulation.py --output-root output/move_simulation
```

The relocation check compares every packaged file with
`SELF_CONTAINED_PROJECT_MANIFEST.json`, then checks eleven reconstructed data
products against recorded hashes and verifies the generator version. The three
reference simulations are a technical portability test, separate from the
complete scientific analysis. Their successful execution alone does not mean
that every analysis or engineering quality check has completed.

The final figure and manuscript artifacts are collected after the scientific
analyses finish. Their versions and supporting evidence are recorded separately
from the input/runtime manifest.

Figure 1 can be regenerated directly from the completed `demand` and
`observations` stages, without legacy panel tables:

```powershell
.\tools\python\python.exe plot/reproduce.py --figure 1 --raw-run-root output/from_raw
```

Country-city locations shared by the two observation archives are pooled.
Display-coordinate overrides and policy matching declarations are retained in
`data/metadata`; plotting does not query an online geocoder or policy service.
