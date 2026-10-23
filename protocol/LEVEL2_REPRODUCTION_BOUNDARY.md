# Level 2 frozen EnergyPlus reproduction boundary

## Scope

Level 2 optionally reruns the exact building trajectories used by the frozen analysis. It contains 412 unique execution records: 346 annual trajectories and 66 event trajectories. The 66 event trajectories use 33 annual baselines, each reused twice, so those baselines are references rather than 33 additional EnergyPlus runs.

Level 2 does not create actions, change controls, retune building physics, or reconstruct the historical annual action-library generator. Level 1 remains the required route for reproducing the paper's scored action tables, decisions and figures.

## Locked files

- `configs/energyplus_runs.csv`: one record per EnergyPlus execution, including the run class, logical city, content-addressed IDF asset, EPW asset, exact hashes, EnergyPlus version, command template, historical engineering-QA hash and gate identifier.
- `configs/energyplus_dynamic_baseline_reuse.csv`: the 33 annual baselines and their two linked event trajectories.
- `configs/energyplus_level2_contract.json`: counts, version, gate and generator-provenance lock.
- `src/thermal_heterogeneity/building/energyplus.py`: fail-closed package validation and single-run adapter.
- `scripts/validate_energyplus_manifest.py`: offline manifest validator.

The manifest has 412 unique run IDs and source locators. It references 384 unique IDF byte streams because 28 annual execution records intentionally retain byte-identical frozen inputs under distinct historical action/run identities. Four EPW assets are used.

## Runtime and command

The runtime is pinned to EnergyPlus `22.1.0-ed759b17ee`. Each command is equivalent to:

```console
energyplus -w <input-package>/epw/<epw-asset>.epw -d <output-root>/<run-id> <input-package>/idf/<idf-asset>.idf
```

The adapter builds the argument vector from the locked asset fields; it does not execute the command-template string or search the filesystem for alternative files.

## Rights-cleared local input package

IDF and EPW files are excluded from this repository. A local Level 2 input package must contain the exact content-addressed files and a user-supplied rights manifest:

```text
level2-local-assets-v1/
  LEVEL2_RIGHTS.json
  idf/
    idf_<sha256>.idf
  epw/
    <epw_asset_id>.epw
```

`LEVEL2_RIGHTS.json` has this minimal form:

```json
{
  "package_id": "level2-local-assets-v1",
  "local_use_authorized": true,
  "assets": {
    "idf_<sha256>": {
      "sha256": "<sha256>",
      "rights_basis": "<user-recorded local-use basis>"
    },
    "<epw_asset_id>": {
      "sha256": "<sha256>",
      "rights_basis": "<user-recorded local-use basis>"
    }
  }
}
```

If the package directory, exact IDF, exact EPW, rights manifest, local-use authorization or per-asset rights declaration is absent, the public adapter returns `LEVEL2_INPUT_PACKAGE_UNAVAILABLE`. A present but malformed package or hash mismatch returns `LEVEL2_INPUT_PACKAGE_INVALID`. No nearby prototype, weather file or trajectory is substituted.

## Engineering gate

`LEVEL2_TRAJECTORY_GATE_V1` requires:

1. EnergyPlus version exactly `22.1.0-ed759b17ee`;
2. a zero process exit code;
3. `eplusout.end` reporting `EnergyPlus Completed Successfully`;
4. zero `Severe` and zero `Fatal` entries in `eplusout.err`.

Every manifest row also retains the SHA-256 of the historical engineering-QA record that admitted that trajectory to the frozen analysis. Passing the Level 2 runtime gate establishes exact-asset execution and EnergyPlus completion. It does not establish empirical building calibration or regenerate the original strict engineering analysis.

## Historical generator boundary

The frozen simulation package recorded the annual generator SHA-256 as `c1b0c7a94cfd906568aa9f9f3367120badd299ac2bc7daa73e58a4469ab96c63`. The later working-tree file with the same historical role has SHA-256 `433e5cc2702df509a6e93d74acbe18abd19d0a76de82506295cf3ee259387070`. The original recorded snapshot is unavailable. These hashes are deliberately retained as different provenance objects. Level 2 therefore reruns frozen IDF/EPW assets and must not claim to regenerate the action library.

## Validation

From the release root:

```console
python scripts/validate_energyplus_manifest.py configs/energyplus_runs.csv --baseline-reuse configs/energyplus_dynamic_baseline_reuse.csv --contract configs/energyplus_level2_contract.json
python -m pytest tests/contracts/test_energyplus_boundary.py -q
```

The contract tests include missing-package, missing-exact-asset, rights, hash-tampering and engineering-gate cases.
