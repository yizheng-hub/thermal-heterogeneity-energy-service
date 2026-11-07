"""Score the complete existing uniform-command grid as a strong shared baseline."""

from common import *
from score_new_controls import score_manifest
from run_global_matrix import ZONES
import pandas as pd


def main():
    probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    profiles = pd.read_csv(MATRIX / "GLOBAL_PROFILES.csv")
    models = json.loads((MATRIX / "models/MODEL_MANIFEST.json").read_text())
    rows = []
    for row in probes.itertuples():
        assert row.completed and row.relative_engineering_pass
        targets = dict.fromkeys(ZONES, float(row.target_c))
        for composition in sorted(profiles.composition_id.unique()):
            rows.append(
                {
                    "climate": row.climate,
                    "context": row.climate,
                    "composition_id": int(composition),
                    "layout": "random_mixing",
                    "permission": "building",
                    "control": "uniform_command_search",
                    "offset_c": float(row.target_c) - 25,
                    "uniform_target_c": float(row.target_c),
                    "physical_id": row.input_sha256,
                    "run_id": row.run_id,
                    "input_sha256": row.input_sha256,
                    "targets_sha256": hashlib.sha256(
                        json.dumps(targets, sort_keys=True).encode()
                    ).hexdigest(),
                    "input_idf_path": str(MATRIX / "probe_runs/inputs" / (row.run_id + ".idf")),
                    "weather_path": models[row.climate]["weather_path"],
                    "qa_path": str(MATRIX / "probe_runs/runs" / row.run_id / "RUN_QA.json"),
                    "hourly_path": row.hourly_path,
                    "hourly_sha256": row.hourly_sha256,
                    "completed": True,
                    "relative_engineering_pass": True,
                }
            )
    manifest = pd.DataFrame(rows)
    assert len(manifest) == 1674
    manifest.to_csv(ROOT / "results/UNIFORM_SEARCH_FULL_MANIFEST.csv", index=False)
    score_manifest("uniform_search")
    metrics = pd.read_csv(ROOT / "results/UNIFORM_SEARCH_METRICS.csv")
    metrics["uniform_target_c"] = metrics.run_id.map(probes.set_index("run_id").target_c)
    metrics.to_csv(ROOT / "results/UNIFORM_SEARCH_METRICS.csv", index=False)
    anchor = pd.read_csv(ROOT / "results/base/annual/GLOBAL_25C_SERVICE_ANCHORS.csv")
    common = metrics[metrics.uniform_target_c.eq(25)].merge(
        anchor, on=["climate", "composition"], validate="one_to_one"
    )
    assert abs(common.mean_reference_mismatch_c - common.anchor_mean_mismatch_c).max() < 1e-9
    assert abs(common.fixed_tail_reference_mismatch_c - common.anchor_tail_mismatch_c).max() < 1e-9
    write_json(
        ROOT / "audit/UNIFORM_COMMAND_SEARCH_COMPLETE.json",
        {
            "status": "completed",
            "logical_scores": len(metrics),
            "physical_trajectories_reused": len(probes),
            "new_simulations": 0,
            "role": "Explicit common physical search library,including25C; no change to the original controller candidate-family diagnostic.",
        },
    )


if __name__ == "__main__":
    main()
