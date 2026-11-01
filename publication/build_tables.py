"""Build manuscript tables from retained scientific outputs without rerunning models."""

from pathlib import Path
import argparse
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONTROL = {
    "uniform_average": "Average",
    "personal_center": "Personal center",
    "personal_range": "Personal range",
}
DESIGN = {
    "timed_multi": "Timed, repeated",
    "timed_single": "Timed, single",
    "undated_multi": "Untimed, repeated",
    "undated_single": "Untimed, single",
}
MODEL = {
    "raw_center": "Individual center",
    "shrink_center": "Shrunk center",
    "building_mean": "Building mean",
    "building_median": "Building median",
    "pooled_mean": "Pooled mean",
    "pooled_median": "Pooled median",
    "study_mean": "Study mean",
    "study_median": "Study median",
    "source_prior_only": "Source prior",
}
MENU = {"building_search": "Building", "zone_summary": "Zone summary", "zone_full": "Full demand"}


def quarter_reassignment_rows(interaction):
    """Summarize only matched operations with the stated quarter-move budget."""
    quarter = interaction.loc[interaction["layout"].eq("within_floor_relocation_25pct")]
    rows = []
    for control, group in quarter.groupby("control"):
        for suffix, system in [("vav", "VAV"), ("alternative", "Zone heat pump")]:
            rows.append(
                [
                    system,
                    CONTROL[control],
                    len(group),
                    group["population_gain_c_" + suffix].median(),
                    group["fixed_tail_gain_c_" + suffix].median(),
                    group["energy_change_pct_" + suffix].median(),
                ]
            )
    return rows


def build(raw, analysis, publication, resource_scenarios=None):
    tables = []
    sources = {}

    def read(relative, base=analysis):
        path = (base / relative).resolve()
        sources[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    def add(number, title, frame, note):
        tables.append(
            {
                "number": number,
                "title": title,
                "columns": list(frame.columns),
                "rows": frame.fillna("—").astype(str).values.tolist(),
                "note": note,
            }
        )

    def f(value, digits=3):
        return "—" if pd.isna(value) else f"{value:.{digits}f}"

    metrics = read("results/group_information_validation/ALL_FINAL_METRICS.csv")
    support = (
        metrics.groupby(["design", "endpoint"])[["records", "participants"]].first().reset_index()
    )
    support.design = support.design.map(DESIGN)
    if support.design.isna().any():
        raise ValueError("Unmapped validation design")
    support.endpoint = support.endpoint.map(
        {"no_change": "No-change preference", "acceptability": "Acceptability"}
    )
    support.columns = ["Evaluation", "Response", "Records", "Participants"]
    add(
        1,
        "Final response-evaluation support",
        support,
        "Timed evaluations use later responses after personal calibration. Untimed evaluations use the independent participant split without a verified time order.",
    )
    shared = metrics[metrics.width_policy.eq("shared_width")].copy()
    shared["Evaluation"] = shared.design.map(DESIGN)
    shared["Center"] = shared.center_model.map(MODEL)
    if shared[["Evaluation", "Center"]].isna().any().any():
        raise ValueError("Unmapped model label")
    shared["Half-width (°C)"] = shared.half_width_c.map(lambda v: f(v, 2))
    shared["Balanced accuracy"] = shared.apply(
        lambda r: f"{r.balanced_accuracy:.3f} [{r.ba_lower95:.3f}, {r.ba_upper95:.3f}]", axis=1
    )
    shared["AUROC"] = shared.auroc.map(f)
    shared["Brier score"] = shared.brier.map(f)
    add(
        2,
        "Independent response prediction at a shared width",
        shared[
            ["Evaluation", "Center", "Half-width (°C)", "Balanced accuracy", "AUROC", "Brier score"]
        ],
        "Bracketed values are 95% participant-cluster bootstrap intervals. Widths are selected in development data. Tuned-width and Personal range metrics and all paired contrasts are retained in Supplementary Data 1.",
    )
    old = read("results/decision_effect_sizes/DECISION_EFFECT_SIZES.csv", raw)
    old = old[old.selection_disagreement.eq(True)].copy()
    composition_labels = {
        f"{source} {letter}": str(offset + i + 1)
        for source, offset in [("Multinational", 0), ("China", 3)]
        for i, letter in enumerate("ABC")
    }
    old["Composition"] = old.composition.map(composition_labels)
    old["Population excess (°C)"] = (-old.zone_mean_selected_full_population_margin_c).map(f)
    old["Tail excess (°C)"] = (-old.zone_mean_selected_full_tail_margin_c).map(f)
    old["Extra energy (MWh)"] = (old.delta_purchased_kwh_full_minus_zone_mean / 1000).map(f)
    old = old.rename(columns={"climate": "Climate"})
    add(
        3,
        "Original-library settings with different selected operations",
        old[
            [
                "Climate",
                "Composition",
                "Population excess (°C)",
                "Tail excess (°C)",
                "Extra energy (MWh)",
            ]
        ],
        "Signed excess is the mean-selected operation's full-demand mismatch minus the common limit; negative values indicate a margin below the limit. Extra energy is full-demand selection minus mean selection. Composition labels identify sampled demand groups, not countries of simulated occupants.",
    )
    grid = read("results/aligned_grid_sensitivity/ALIGNED_GRID_SUMMARY.csv")
    compact = grid[
        [
            "level",
            "settings",
            "qualified_set_changes",
            "choice_changes",
            "mean_selected_full_service_exceedances",
        ]
    ].copy()
    compact.level = compact.level.map(
        {"half": "0.5 °C", "quarter": "0.25 °C", "eighth": "0.125 °C"}
    )
    compact.columns = [
        "Grid",
        "Settings",
        "Set differences",
        "Choice differences",
        "Mean choices exceeding service limits",
    ]
    add(
        4,
        "Selection stability across aligned offset grids",
        compact,
        "All grids use the same full-demand reference requirements and retain the frozen uniform building search. These are finite-library results.",
    )
    pooled = read("results/information_uncertainty_summary/NOMINAL_POOLED_CHOICES_WITH_ORIGINS.csv")
    rows = []
    for climate, g in pooled.groupby("climate"):
        changed = g[g.choices_differ.eq(True)]
        rows.append(
            [
                climate,
                len(g),
                len(changed),
                f(changed.zone_mean_selected_full_service_excess_c.median()),
                f(changed.extra_energy_kwh.median() / 1000),
            ]
        )
    add(
        5,
        "Expanded-library choice differences by climate",
        pd.DataFrame(
            rows,
            columns=[
                "Climate",
                "Settings",
                "Different choices",
                "Median excess (°C)",
                "Median extra energy (MWh)",
            ],
        ),
        "Excess and extra energy summarize only settings with different choices. A dash denotes no such setting, not a zero service difference.",
    )
    interaction = read(
        "results/placement_hvac_interaction/independent_pthp_MATCHED_CROSS_SYSTEM_EFFECTS.csv"
    )
    rows = [
        row[:3] + [f(value) for value in row[3:]] for row in quarter_reassignment_rows(interaction)
    ]
    add(
        6,
        "Quarter-reassignment effects on common cross-system support",
        pd.DataFrame(
            rows,
            columns=[
                "System",
                "Control",
                "Pairs",
                "Population gain (°C)",
                "Least-served gain (°C)",
                "Energy change (%)",
            ],
        ),
        "All entries are medians of within-system paired changes. The same 52 Personal center and 53 Personal range comparisons are retained for both systems.",
    )
    resource = read("results/dr_capability_summary/RESOURCE_SUMMARY_ALL_SETTINGS.csv")
    resource = resource[resource.normal_energy_reference_factor.eq(1)].copy()
    response = pd.DataFrame(
        {
            "Normal menu": resource.control.map(MENU),
            "Events": resource.planned_events,
            "Positive events": resource.selected_positive_response_events,
            "All-event power (kW)": resource.available_event_power_mean_kw.map(f),
            "Positive-net policy power (kW)": resource.available_positive_net_event_power_mean_kw.map(
                f
            ),
        }
    )
    add(
        7,
        "One-action response under the reference energy budget",
        response,
        "Each mean uses all 432 opportunities, including resolved cases without a positive service-preserving response as zero. The positive-net policy maximizes event power subject to positive full-window electricity savings.",
    )
    decisions = read(
        "results/information_compression_final/GLOBAL_INFORMATION_COMPRESSION_DECISIONS.csv", raw
    )
    decisions = decisions[decisions.service_fraction.eq(1)]
    rows = []
    for name, g in decisions.groupby("representation"):
        selected = g.selected_plan_id.notna()
        rows.append(
            [
                name.replace("_", " "),
                len(g),
                int(selected.sum()),
                int(g.loc[selected, "selected_plan_passes_full_review"].sum()),
                int(g.qualified_under_both_count.sum()),
                int(g.full_qualified_plan_count.sum()),
            ]
        )
    add(
        8,
        "Compact demand representations at the reference service requirement",
        pd.DataFrame(
            rows,
            columns=[
                "Representation",
                "Settings",
                "Choices",
                "Individually qualifying selections",
                "Qualifying plans retained",
                "Full qualifying plans",
            ],
        ),
        "Qualification and exact selection agreement are different endpoints. Cross-context error margins are identified by the representation name. Plan counts sum over settings.",
    )
    costs = (
        read(resource_scenarios.name, resource_scenarios.parent)
        if resource_scenarios is not None
        else read("support/implementation/IMPLEMENTATION_RESOURCE_SCENARIOS.csv", publication)
    )
    rows = []
    for wage, g in costs.groupby("wage_profile"):
        rows.append(
            [
                wage.capitalize(),
                len(g),
                f(g.annual_resource_cost_usd.min(), 2),
                f(g.annual_resource_cost_usd.median(), 2),
                f(g.annual_resource_cost_usd.max(), 2),
            ]
        )
    add(
        9,
        "Annualized implementation-resource scenarios",
        pd.DataFrame(
            rows,
            columns=[
                "Wage assumption",
                "Scenarios",
                "Minimum (US$)",
                "Median (US$)",
                "Maximum (US$)",
            ],
        ),
        "Illustrative inputs are not observed building-management costs. Capital is annualized over five years at 3%; reassignment and command scope remain quantities for local valuation.",
    )
    destination = publication / "tables"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "TABLES.json").write_text(
        json.dumps({"tables": tables, "input_sha256": sources}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    for t in tables:
        if "number" in t:
            pd.DataFrame(t["rows"], columns=t["columns"]).to_csv(
                destination / f"Table_S{t['number']:02}.csv", index=False
            )
    print(json.dumps({"tables_written": len(tables)}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=ROOT / "output/from_raw")
    p.add_argument("--analysis-root", type=Path, default=ROOT / "output/strengthening")
    p.add_argument("--publication-root", type=Path, default=ROOT / "output/publication")
    p.add_argument("--resource-scenarios", type=Path)
    args = p.parse_args()
    if not all(
        x.resolve().is_relative_to(ROOT)
        for x in [args.raw_root, args.analysis_root, args.publication_root]
    ):
        raise ValueError("Publication inputs and outputs must stay inside the project")
    if args.resource_scenarios is not None and not args.resource_scenarios.resolve().is_relative_to(
        ROOT
    ):
        raise ValueError("Resource inputs must stay inside the project")
    build(args.raw_root, args.analysis_root, args.publication_root, args.resource_scenarios)
