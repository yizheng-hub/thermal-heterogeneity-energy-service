"""Run the original matrix while retaining full hourly SQL for later diagnostics."""

import json
import run_global_matrix as matrix


original_run_energyplus = matrix.run_energyplus


def run_with_retained_sql(*args, **kwargs):
    kwargs["cleanup_raw"] = False
    return original_run_energyplus(*args, **kwargs)


def main():
    previous = matrix.run_energyplus
    matrix.run_energyplus = run_with_retained_sql
    try:
        print(json.dumps({"storage_policy": "Retain full hourly SQL for the declared downstream mechanism diagnostics",
                          "physical_and_scoring_changes": False}), flush=True)
        matrix.main()
    finally:
        matrix.run_energyplus = previous


if __name__ == "__main__":
    main()
