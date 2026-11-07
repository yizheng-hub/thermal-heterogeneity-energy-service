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
MENU = {"building_search": "Building", "zone_summary": "Zone summary", "zone_full": "Full zone"}


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
        if not path.is_file() and (ROOT / "Data/index.json").is_file():
            from publication.standalone_plotting import resolve
            path = resolve(path)
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
        "Records count held-out responses and participants count distinct source-qualified identities in the final evaluation. Timed evaluations use later responses after personal calibration. Untimed evaluations use a participant split without verified time order. Repeated calibration uses the available early records; single calibration uses one record. Acceptability and no-change preference are separate response endpoints.",
    )
    shared = metrics[metrics.width_policy.eq("shared_width")].copy()
    # The complete nine-model table accompanies the plotting inputs.
    shared = shared[shared.center_model.isin(["raw_center", "shrink_center", "building_median"])].copy()
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
        "Individual and building-median prediction at a shared width",
        shared[
            ["Evaluation", "Center", "Half-width (°C)", "Balanced accuracy", "AUROC", "Brier score"]
        ],
        "Balanced accuracy is the mean of sensitivity and specificity; each entry reports its point estimate followed by the lower and upper bounds of a 95% participant-cluster bootstrap confidence interval (CI) in brackets (500 resamples within the source–building structure). The area under the receiver operating characteristic curve (AUROC) measures discrimination. The Brier score is mean squared probability error; lower is better, whereas higher balanced accuracy and AUROC are better. AUROC and Brier entries are point estimates. Half-widths are selected using development data and shared across center models within each design. This table shows the individual, shrinkage and building-median models. All nine center baselines, tuned-width and Personal range metrics, and paired contrasts are supplied in Supplementary Software 1.",
    )
    tables[-1]["columns"][3] = "Balanced accuracy [95% CI]"
    old = read("results/decision_effect_sizes/DECISION_EFFECT_SIZES.csv", raw)
    old = old[old.selection_disagreement.eq(True)].copy()
    composition_labels = {
        f"{source} {letter}": str(offset + i + 1)
        for source, offset in [("Multinational", 0), ("China", 3)]
        for i, letter in enumerate("ABC")
    }
    old["Composition"] = old.composition.map(composition_labels)
    old["Population excess (°C)"] = old.zone_mean_selected_full_population_margin_c.map(f)
    old["Tail excess (°C)"] = old.zone_mean_selected_full_tail_margin_c.map(f)
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
        "Signed excess is the mean-selected operation’s full-demand mismatch minus the corresponding 25 °C reference limit; positive values exceed the limit and negative values remain below it. Tail denotes the fixed 27 least-served profiles among 267, identified under the reference. Extra energy is full-demand selection minus mean selection, in annual purchased MWh. These ten settings come from the 810-entry library. Compositions 1–3 sample the ASHRAE Global Thermal Comfort Database II and 4–6 the Chinese Thermal Comfort Dataset; they are demand samples rather than occupant nationalities.",
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
        "Grid denotes the temperature-offset spacing. Set differences count settings with different qualifying operation sets under zone-mean and individual-demand scoring; choice differences count different lowest-energy physical selections. The last column checks mean-selected operations against individual-demand service limits. Each row covers the same 54 climate–composition settings and full-demand reference requirements, with the frozen uniform building search retained. Minima refer to each evaluated finite library.",
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
        "Results use the expanded 8,586-entry logical library, with six compositions per climate. Excess is the larger of population and fixed least-served mismatch minus its own reference limit. Excess and extra energy summarize only settings with different choices; extra energy is individual-demand minus mean-selected annual purchased energy. A dash denotes no changed choice. Compositions sample the ASHRAE Global Thermal Comfort Database II and Chinese Thermal Comfort Dataset.",
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
        "Variable-air-volume (VAV) equipment is compared with independent packaged terminal heat pumps (zone heat pumps). All entries are medians of within-system paired changes from mixed to within-floor arranged placement under the 25% move allowance. Service gains are mixed minus arranged mismatch; percentage energy changes are arranged minus mixed energy divided by mixed energy. The same 52 Personal center and 53 Personal range comparisons meet the engineering criteria in both systems. Purchased energy sums facility electricity and gas.",
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
        "Building, zone-summary and full-zone menus provide alternative normal-operation command libraries under the same individual-demand population and fixed-group service limits. The annual energy budget equals reference operation. Each power mean uses all 432 opportunities (nine climates × six compositions × eight events), including resolved cases without a positive service-preserving response as zero. One of five actions is selected per event. The positive-net policy maximizes event power subject to positive electricity savings over preparation, event and 48-hour recovery.",
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
                "Full-demand qualifying selections",
                "Qualifying plans retained",
                "Full qualifying plans",
            ],
        ),
        "Choices count settings returning a selection; qualifying selections count those also meeting the full-demand service requirements. Retained plans count setting–operation instances qualifying under both representations; the denominator is 700 full-demand qualifying instances in this library. These are qualification endpoints, distinct from exact selection agreement. Five-bin summaries group demand centers; cross-context error margins are identified in the representation name. Counts sum over 54 settings.",
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
        "The two wage assumptions each contain 108 illustrative resource scenarios, giving 216 scenarios in total. The table summarizes annual costs in United States dollars (US$) within each wage assumption rather than observed management-cost distributions. Capital is annualized over five years at 3%; reassignment counts and command scope remain quantities for local valuation. Setup, maintenance, feedback, software and optional interfaces use the declared resource inputs.",
    )
    add(
        10,
        "Equipment operation acceptance and diagnostic definitions",
        pd.DataFrame(
            [
                ["Annual completion", "Acceptance", "Complete 8,760-hour trajectory for 15 conditioned zones; no reported severe or fatal errors."],
                ["Zone-feedback warmup", "Acceptance", "EnergyPlus warmup retains a minimum of six and maximum of 25 days, a temperature-convergence tolerance of 0.20 °C and a loads-convergence tolerance of 0.04. Reported warmup-nonconvergence severe errors exclude an operation even if its annual trajectory completes."],
                ["Zone-feedback expansion", "Acceptance and stopping rule", "Each climate must pass all seven technical-pilot operations (one reference and three controls for each of two compositions) before the remaining twelve operations are run. All seven pilot operations failed in each of Brasília and Singapore, leaving twelve operations per climate unrun (24 total)."],
                ["Warnings and numerical stability", "Acceptance", "Coil operating-range warnings exclude an operation. HVAC-iteration, wet-bulb or psychrometric-temperature warnings trigger a 100-iteration rerun. Maximum hourly zone-temperature difference must be ≤0.02 °C, relative annual purchased-energy difference ≤0.001, and maximum occupancy difference ≤10⁻¹⁰, with no new warning category. Warning-count changes are recorded."],
                ["Electricity accounting", "Acceptance", "At every hour, heating, cooling and fan electricity plus metered transformer loss must close to HVAC electricity within 10⁻⁶ kWh."],
                ["Outdoor-air delivery", "Acceptance and separate diagnostic", "No occupied hour may deliver less than 99% of the area-based outdoor-air request after applying its fractional schedule. The separate VAV analysis checks actual air-loop delivery against the native mechanical-ventilation request. Occupied-loop hours and occupied-building hours are separate denominators."],
                ["Equipment capacity", "Fixed comparison input", "Autosized equipment and airflow fields are resolved using the common 25 °C design run for each system and climate, then held fixed across strategies. VAV reset alternatives retain original VAV capacities; the heat-pump architecture uses its own sizing run."],
                ["Setpoint tracking", "Reported diagnostic", "Heating and cooling use the EnergyPlus reporting tolerance of 0.556 °C and EnergyPlus While Occupied durations. Annual zone-hours sum these durations over 15 conditioned zones and 8,760 hourly records. No unmet-hour cutoff is used for engineering exclusion. Service-window and outside-window sums partition this annual total."],
            ],
            columns=["Item", "Role", "Executed rule or definition"],
        ),
        "Heating, ventilation and air-conditioning (HVAC) operation is assessed using the rules above. Variable-air-volume (VAV) candidates use the reference equipment. Relative annual energy tolerance 0.001 equals 0.1%. Alternative systems use the acceptance rules above; reference VAV candidates also undergo prototype-relative warning checks. Numerical acceptance, setpoint-tracking diagnostics and thermal-service selection are separate. Thermal-service selection uses population and fixed-group mismatch. The hourly maximum of zone unmet durations is the largest unmet duration among zones in that hour.",
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
