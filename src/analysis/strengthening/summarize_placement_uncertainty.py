"""Describe setting-level placement sensitivities without treating draws as independent people."""
from common import *
import pandas as pd


def main():
    for name in ["placement_uncertainty_study_shared_identity_hvac", "placement_uncertainty_single_record_identity_hvac"]:
        folder = ROOT / "results" / name
        source = folder / "PAIRED_CONSTRAINED_PLACEMENT_SUMMARY.csv"
        audit = json.loads((folder / "AUDIT.json").read_text())
        if audit["status"] != "completed":
            raise ValueError("Complete the declared fixed-trajectory sensitivity before summarising")
        frame = pd.read_csv(source)
        summary = frame.groupby(["system", "control", "layout", "scenario"]).agg(
            settings=("climate", "size"),
            population_gain_median_c=("population_gain_median_c", "median"),
            tail_gain_median_c=("fixed_tail_gain_median_c", "median"),
            minimum_setting_joint_positive_fraction=("conditional_draw_fraction_both_service_gains", "min"),
            median_setting_joint_positive_fraction=("conditional_draw_fraction_both_service_gains", "median"),
            fixed_energy_median_pct=("energy_change_pct_fixed_operations", "median"))
        target = folder / "DESCRIPTIVE_SETTING_SUMMARY.csv"
        summary.to_csv(target)
        write_json(folder / "DESCRIPTIVE_SETTING_SUMMARY_AUDIT.json", {
            "source_sha256": digest(source), "output_sha256": digest(target),
            "summary_cells": len(summary),
            "interpretation": "Descriptive summaries of setting-level conditional sensitivities; no independent-sample confidence interval. Systems retain their own declared engineering availability; use matched outputs for architecture contrasts."})
        print(name, len(summary), flush=True)


if __name__ == "__main__":
    main()
