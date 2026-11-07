from common import *
import pandas as pd
import numpy as np
from run_global_matrix import warning_profile, relative_engineering_qa


def audit(name):
    f = pd.read_csv(ROOT / "results" / name)
    probes = pd.read_csv(MATRIX / "probe_runs/PROBE_RUN_MANIFEST.csv")
    rows = []
    for row in f.drop_duplicates("physical_id").itertuples():
        anchor = probes[probes.climate.eq(row.climate) & probes.target_c.eq(25)].iloc[0]
        parent = Path(anchor.hourly_path).parents[1] / "runs" / anchor.run_id
        qa = json.loads(Path(row.qa_path).read_text())
        profile = warning_profile(Path(row.qa_path).parent / "eplusout.err")
        relative = relative_engineering_qa(qa, profile, warning_profile(parent / "eplusout.err"))
        categories = set(relative["new_warning_categories"]) | set(
            relative["critical_warning_count_increases"]
        )
        stable = (
            json.loads(row.solver_stability)
            if isinstance(getattr(row, "solver_stability", None), str)
            else {}
        )
        ordinary = relative["engineering_pass_relative_to_prototype_baseline"]

        def truth(value):
            return str(value).strip().lower() in ["true", "1", "1.0"]

        reused = truth(getattr(row, "reused_existing_physical_operation", False))
        reused_original_validation = bool(
            reused
            and truth(row.relative_engineering_pass)
            and (
                truth(getattr(row, "raw_relative_engineering_pass", False))
                or truth(getattr(row, "annual_solver_stability_pass", False))
            )
        )
        exception = bool(
            categories <= {"hvac_iterations"}
            and not qa.get("unexplained_recurring_groups")
            and stable.get("warning_categories_stable")
            and stable.get("temperature_delta_c", np.inf) <= 0.02
            and stable.get("relative_energy_delta", np.inf) <= 0.001
            and stable.get("people_delta", np.inf) <= 1e-10
        )
        passed = bool(
            row.completed
            and row.relative_engineering_pass
            and (ordinary or exception or reused_original_validation)
        )
        rows.append(
            {
                "physical_id": row.physical_id,
                "strict_warning_review_pass": passed,
                "new_categories": json.dumps(sorted(categories)),
                "hvac_only_stability_exception": bool(not ordinary and exception),
                "reused_original_validation": reused_original_validation,
                "inherited_service_water_warning_policy": "retained and compared separately; not thermostat numerical failure",
            }
        )
    output = ROOT / "results" / (Path(name).stem + "_STRICT_WARNING_REVIEW.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    report = {
        "manifest": name,
        "physical_operations": len(rows),
        "strict_passed": sum(x["strict_warning_review_pass"] for x in rows),
        "status": "completed" if all(x["strict_warning_review_pass"] for x in rows) else "partial",
    }
    write_json(ROOT / "audit" / (Path(name).stem + "_STRICT_WARNING_REVIEW.json"), report)
    return report


if __name__ == "__main__":
    print(audit(sys.argv[1]))
