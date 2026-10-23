"""Render the frozen Nature main Figures 4--6 from panel-data v2.

This module is deliberately limited to presentation. It verifies the approved
panel files, computes only descriptive plotting summaries and never reruns a
building, event or human-demand experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from thermal_heterogeneity.figures import (
    CATEGORY_STYLES,
    CONTEXT_COLORS,
    COUNT_CMAP,
    PALETTE,
    REPRESENTATION_COLORS,
    SIGNED_CMAP,
    ExportBundle,
    create_panel_figure,
    save_figure_bundle,
    style_panel,
)


CONTEXTS = ["Miami", "NewDelhi", "PortAngeles_small2019", "Seattle_small2019"]
CONTEXT_LABELS = {
    "Miami": "Miami",
    "NewDelhi": "New Delhi",
    "PortAngeles_small2019": "Port Angeles",
    "Seattle_small2019": "Seattle",
}
CONTEXT_COLOURS = CONTEXT_COLORS
LAMBDA_COLOURS = {
    0.0: PALETTE["blue"],
    0.5: PALETTE["sky"],
    1.0: PALETTE["orange"],
    1.5: PALETTE["vermillion"],
}
LAMBDA_MARKERS = {0.0: "o", 0.5: "s", 1.0: "^", 1.5: "D"}
CLASS_COLOURS = {
    "mean-safe": PALETTE["sky"],
    "distribution-required": PALETTE["vermillion"],
    "finite-library-unresolved": PALETTE["light_grey"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_panel_inputs(panel_directory: Path | str, figure_number: int) -> dict[str, str]:
    """Verify manifest hashes and critical frozen denominators for one figure."""

    panel_directory = Path(panel_directory)
    manifest_path = panel_directory / "PANEL_DATA_V2_MANIFEST.csv"
    manifest = pd.read_csv(manifest_path)
    selected = manifest[manifest["panel_id"].str.startswith(f"Fig{figure_number}")]
    if len(selected) != 8:
        raise ValueError(f"Figure {figure_number} requires exactly eight manifest rows")
    hashes: dict[str, str] = {}
    for row in selected.itertuples(index=False):
        path = panel_directory / row.relative_path
        observed = _sha256(path)
        if observed != row.sha256:
            raise ValueError(f"panel hash mismatch: {path.name}")
        hashes[row.panel_id] = observed

    if figure_number == 4:
        counts = pd.read_csv(panel_directory / "Fig4c.csv")
        full = counts[counts["representation"] == "full"].sort_values("lambda")
        mean = counts[counts["representation"] == "mean"].sort_values("lambda")
        if full["selected_count"].tolist() != [128, 120, 74, 21]:
            raise ValueError("Fig4c full-demand counts differ from the freeze")
        if mean["selected_count"].tolist() != [128, 128, 128, 128]:
            raise ValueError("Fig4c mean-demand counts differ from the freeze")
    elif figure_number == 5:
        case = json.loads((panel_directory / "Fig5bc_case_lock.json").read_text(encoding="utf-8"))
        if not case["mean_full_same_action"]:
            raise ValueError("Fig5b/c must show the same frozen action")
        boundary = pd.read_csv(panel_directory / "Fig5h.csv")
        feb = boundary[boundary["month"] == 2]
        if set(feb["reference_positive_selected_count"]) != {0}:
            raise ValueError("February central reference must remain 0/144")
        if set(feb["month_grid_positive_selected_count"]) != {210}:
            raise ValueError("February full grid must remain 210/2880")
    elif figure_number == 6:
        phase = pd.read_csv(panel_directory / "Fig6c.csv").set_index("classification")
        expected = {
            "mean-safe": 275,
            "distribution-required": 331,
            "finite-library-unresolved": 402,
        }
        if phase["count"].to_dict() != expected:
            raise ValueError("Fig6c phase counts differ from the freeze")
        gate = pd.read_csv(panel_directory / "Fig6f.csv")
        retained = gate["true_positive_candidates"] / gate["full_eligible_candidates"]
        covered = gate["full_available_and_certified_decisions"] / gate["full_available_decisions"]
        if not np.allclose(retained, gate["true_action_retention"], rtol=0, atol=1e-12):
            raise ValueError("Fig6f action-retention denominators do not reproduce")
        if not np.allclose(covered, gate["decision_coverage"], rtol=0, atol=1e-12):
            raise ValueError("Fig6f decision-coverage denominators do not reproduce")
    else:
        raise ValueError("only Figures 4, 5 and 6 are implemented here")
    return hashes


def _panel(axis: Axes, letter: str, title: str, *, grid_axis: str | None = "y") -> None:
    style_panel(axis, letter, title, grid_axis=grid_axis)


def _blank(axis: Axes, letter: str, title: str) -> None:
    _panel(axis, letter, title, grid_axis=None)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def _box(
    axis: Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = "#555555",
    hatch: str | None = None,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        hatch=hatch,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center")


def _arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    axis.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.8, color="#555555")
    )


def _child(parent: Axes, bounds: list[float]) -> Axes:
    axis = parent.inset_axes(bounds)
    axis.tick_params(width=0.65, length=2.5, pad=1, color="#333333")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.65)
        axis.spines[side].set_color("#333333")
    axis.grid(True, axis="y", color="#D9D9D9", linewidth=0.35, zorder=0)
    return axis


def _parent_for_children(axis: Axes, letter: str, title: str) -> None:
    _blank(axis, letter, title)


def _critical(text, name: str):
    """Tag a layout-critical text artist for figure-specific geometry tests."""

    text.set_gid(f"figures-4-6-layout-critical:{name}")
    return text


def _inset_panel_axis(
    axis: Axes,
    *,
    left: float = 0.0,
    right: float = 0.0,
    bottom: float = 0.0,
    top: float = 0.0,
) -> None:
    """Move one top-level data rectangle within its locked panel cell."""

    position = axis.get_position()
    axis.set_position(
        [
            position.x0 + left,
            position.y0 + bottom,
            position.width - left - right,
            position.height - bottom - top,
        ]
    )


def _quantiles(values: pd.Series | np.ndarray) -> tuple[float, float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    return tuple(float(x) for x in np.quantile(clean, [0.25, 0.5, 0.75]))


def _simple_density(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.ptp(values) < 1e-10:
        width = 0.10
    else:
        width = max(0.13, 0.25 * np.std(values, ddof=1))
    density = np.exp(-0.5 * ((grid[:, None] - values[None, :]) / width) ** 2).sum(axis=1)
    area = np.trapz(density, grid)
    return density / area if area > 0 else density


def _event_span(axis: Axes) -> None:
    axis.axvspan(0, 2, color=PALETTE["orange"], alpha=0.16, linewidth=0)
    axis.axvspan(2, 50, color=PALETTE["sky"], alpha=0.08, linewidth=0)


def build_figure_4(panel_directory: Path | str):
    """Build Figure 4: spread and annual service-qualified action space."""

    panel_directory = Path(panel_directory)
    verify_panel_inputs(panel_directory, 4)
    figure, axes = create_panel_figure()
    a, b, c, d, e, f, g, h = axes.ravel()

    # a: same action, different demand representations.
    _blank(a, "a", "Two service checks")
    _box(a, (0.03, 0.37), 0.23, 0.26, "Fixed action\nand trajectory", facecolor="#F2F2F2")
    _arrow(a, (0.27, 0.50), (0.39, 0.72))
    _arrow(a, (0.27, 0.50), (0.39, 0.28))
    _box(a, (0.40, 0.58), 0.27, 0.25, "Mean demand\nservice check", facecolor="#E6E6E6")
    _box(a, (0.40, 0.15), 0.27, 0.25, "Full demand\nservice check", facecolor="#DCEEF8")
    _arrow(a, (0.68, 0.705), (0.78, 0.705))
    _arrow(a, (0.68, 0.275), (0.78, 0.275))
    _critical(a.text(0.88, 0.705, "Pass / fail", va="center", ha="center", color=PALETTE["dark_grey"]), "4a-mean-outcome")
    _critical(a.text(0.88, 0.275, "Pass / fail", va="center", ha="center", color=PALETTE["blue"]), "4a-full-outcome")

    # b: spread manipulation at a fixed centre.
    spread = pd.read_csv(panel_directory / "Fig4b.csv")
    xmin = spread["perturbed_demand_c"].min() - 0.5
    xmax = spread["perturbed_demand_c"].max() + 0.5
    grid = np.linspace(xmin, xmax, 320)
    line_styles = {0.0: "-", 0.5: "--", 1.0: "-.", 1.5: ":"}
    for lam in [0.0, 0.5, 1.0, 1.5]:
        vals = spread.loc[spread["lambda"] == lam, "perturbed_demand_c"].to_numpy()
        density = _simple_density(vals, grid)
        b.plot(grid, density, color=LAMBDA_COLOURS[lam], linestyle=line_styles[lam], label=f"λ={lam:g}")
        q10, q90 = np.quantile(vals, [0.1, 0.9])
        b.plot([q10, q90], [0.003 + lam * 0.002] * 2, color=LAMBDA_COLOURS[lam], linewidth=2.0)
    common_mean = float(spread["perturbed_demand_c"].groupby(spread["lambda"]).mean().mean())
    b.axvline(common_mean, color="#333333", linewidth=0.7)
    _critical(b.text(0.98, 0.57, f"shared mean = {common_mean:.1f} °C", transform=b.transAxes, ha="right", va="top"), "4b-shared-mean")
    b.set_xlim(xmin, xmax + 0.8)
    b.set_ylim(0, b.get_ylim()[1] * 1.12)
    b.set_xticks([15, 20, 25, 30, 35])
    b.set_xlabel("Demand temperature (°C)")
    b.set_ylabel("Normalized density")
    _panel(b, "b", "Fixed-mean demand spread")
    b.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.5, columnspacing=0.8)
    _inset_panel_axis(b, left=0.030, bottom=0.010)

    # c: full-demand coverage falls while the mean line is flat.
    coverage = pd.read_csv(panel_directory / "Fig4c.csv")
    for rep, label in [("full", "Full demand"), ("mean", "Mean")]:
        rows = coverage[coverage["representation"] == rep].sort_values("lambda")
        style = CATEGORY_STYLES["full_demand" if rep == "full" else "mean"]
        c.plot(rows["lambda"], rows["selected_count"], label=label, **style)
        for row in rows.to_dict("records"):
            offset = -15 if rep == "full" and row["lambda"] <= 0.5 else (7 if rep == "full" else 11)
            if rep == "mean" and row["lambda"] <= 0.5:
                offset = 13
            _critical(c.annotate(f"{row['selected_count']}/144", (row["lambda"], row["selected_count"]), xytext=(0, offset), textcoords="offset points", ha="center", va="center", clip_on=True), f"4c-{rep}-{row['lambda']}")
    c.set_xlim(-0.20, 1.70)
    c.set_ylim(0, 172)
    c.set_xticks([0, 0.5, 1, 1.5])
    c.set_yticks([0, 50, 100, 150])
    c.set_ylabel("Selected cells (count)")
    _panel(c, "c", "Qualified selections")
    c.legend(frameon=False, loc="lower left", ncol=2, handlelength=1.5)

    # d: mean-selected actions that fail full review.
    false_accept = pd.read_csv(panel_directory / "Fig4d.csv").sort_values("lambda")
    bars = d.bar(false_accept["lambda"], false_accept["false_accept_count"], width=0.28, color=PALETTE["vermillion"], edgecolor="white")
    for bar, row in zip(bars, false_accept.itertuples(index=False)):
        d.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{row.false_accept_count}/128\n{100*row.false_accept_rate:.0f}%", ha="center", va="bottom")
    d.set_xlim(-0.25, 1.75)
    d.set_ylim(0, 150)
    d.set_xticks([0, 0.5, 1, 1.5])
    d.text(0.98, 0.04, "λ", transform=d.transAxes, ha="right", va="bottom")
    d.set_ylabel("False accepts (count)")
    _panel(d, "d", "Mean false acceptance")
    _inset_panel_axis(d, left=0.030, bottom=0.008)

    # e: action-set contraction before optimization.
    eligible = pd.read_csv(panel_directory / "Fig4e.csv")
    groups = [eligible.loc[eligible["lambda"] == lam, "full_eligible_action_count"].to_numpy() for lam in [0, 0.5, 1, 1.5]]
    boxes = e.boxplot(groups, positions=[0, 0.5, 1, 1.5], widths=0.28, patch_artist=True, showfliers=False, whis=(10, 90))
    for patch, color in zip(boxes["boxes"], [LAMBDA_COLOURS[x] for x in [0, 0.5, 1, 1.5]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    medians_for_display = []
    zeros_for_display = []
    for index, (lam, values) in enumerate(zip([0, 0.5, 1, 1.5], groups)):
        jitter = np.linspace(-0.11, 0.11, len(values))
        e.scatter(lam + jitter, values, s=5, color=LAMBDA_COLOURS[lam], alpha=0.52, linewidths=0)
        unresolved = int((values == 0).sum())
        medians_for_display.append(f"{np.median(values):.0f}")
        zeros_for_display.append(str(unresolved))
    e.set_xlim(-0.25, 1.75)
    e.set_ylim(-3, 82)
    e.set_xticks([0, 0.5, 1, 1.5])
    e.set_yticks([0, 20, 40, 60, 80])
    e.set_ylabel("Full-eligible actions (count)")
    _critical(e.text(0.03, 0.96, f"Median  {' · '.join(medians_for_display)}", transform=e.transAxes, ha="left", va="top"), "4e-medians")
    _critical(e.text(0.03, 0.87, f"Zero      {' · '.join(zeros_for_display)}", transform=e.transAxes, ha="left", va="top"), "4e-zeros")
    _panel(e, "e", "Eligible action counts")

    # f: energy effects only within common-service selections.
    energy = pd.read_csv(panel_directory / "Fig4f.csv")
    f.axvline(0, color="#333333", linewidth=0.65)
    y_base = np.arange(len(CONTEXTS))[::-1]
    offsets = {0.0: -0.20, 0.5: 0.0, 1.5: 0.20}
    for lam in [0.0, 0.5, 1.5]:
        for y, context in zip(y_base, CONTEXTS):
            rows = energy[(energy["context"] == context) & (energy["lambda"] == lam)]
            matched = rows[rows["bilateral_matched"]]
            q1, med, q3 = _quantiles(matched["delta_energy_mwh"])
            yy = y + offsets[lam]
            if np.isfinite(med):
                interval = f.plot([q1, q3], [yy, yy], color=LAMBDA_COLOURS[lam], linewidth=1.5)[0]
                interval.set_gid(f"figure-4f-data-interval:{context}:{lam}")
                marker = f.scatter(
                    [med],
                    [yy],
                    color=LAMBDA_COLOURS[lam],
                    marker=LAMBDA_MARKERS[lam],
                    s=22,
                    zorder=3,
                )
                marker.set_gid(f"figure-4f-data-marker:{context}:{lam}")
    denominator_rows = []
    for context in CONTEXTS:
        parts = []
        for lam in [0.0, 0.5, 1.5]:
            rows = energy[(energy["context"] == context) & (energy["lambda"] == lam)]
            matched = int(rows["bilateral_matched"].sum())
            parts.append(f"{matched}/{len(rows)-matched}")
        denominator_rows.append(" · ".join(parts))
    f.set_yticks(y_base, ["MIA", "NDL", "PTA", "SEA"])
    f.set_xlim(-70, 172)
    f.set_xticks([-60, -30, 0, 30, 60])
    f.set_xlabel("Energy change (MWh yr⁻¹)")
    f.set_ylabel("")
    f.axvline(50, color="#D2D2D2", linewidth=0.65)
    for y, label in zip(y_base, denominator_rows):
        _critical(f.text(168, y, label, ha="right", va="center"), f"4f-denominator-{y}")
    _critical(f.text(168, 4.00, "m/u by λ: 0 · .5 · 1.5", ha="right", va="bottom"), "4f-denominator-header")
    f.set_ylim(-0.5, 4.75)
    _panel(f, "f", "Matched annual energy", grid_axis="x")
    legend = f.legend(
        handles=[Line2D([], [], color=LAMBDA_COLOURS[x], marker=LAMBDA_MARKERS[x], linestyle="") for x in [0, 1.5, 0.5]],
        labels=["λ0", "λ1.5", "λ.5"],
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.0,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    legend.set_gid("figure-4f-legend")
    _inset_panel_axis(f, left=0.040, bottom=0.010)

    # g: CO2 forest plus a signed carrier heatmap inset.
    response = pd.read_csv(panel_directory / "Fig4g.csv")
    _parent_for_children(g, "g", "Carbon and carriers")
    main = _child(g, [0.05, 0.18, 0.37, 0.66])
    carrier = _child(g, [0.64, 0.18, 0.34, 0.66])
    main.axvline(0, color="#333333", linewidth=0.65)
    row_positions: list[float] = []
    row_labels: list[str] = []
    carrier_matrix: list[list[float]] = []
    for context_index, context in enumerate(CONTEXTS):
        base = (len(CONTEXTS) - 1 - context_index) * 3
        for lam_index, lam in enumerate([0.0, 0.5, 1.5]):
            rows = response[(response["context"] == context) & (response["lambda"] == lam) & response["bilateral_matched"]]
            y = base + (2 - lam_index)
            q1, med, q3 = _quantiles(rows["delta_co2_t"])
            main.plot([q1, q3], [y, y], color=LAMBDA_COLOURS[lam], linewidth=1.4)
            main.scatter([med], [y], s=18, color=LAMBDA_COLOURS[lam], marker=LAMBDA_MARKERS[lam], zorder=3)
            row_positions.append(y)
            context_code = {"Miami": "MIA", "NewDelhi": "NDL", "PortAngeles_small2019": "PTA", "Seattle_small2019": "SEA"}[context]
            row_labels.append(context_code if lam == 0.5 else "")
            carrier_matrix.append([
                float(rows["heating_electricity_delta_mwh"].median()),
                float(rows["cooling_electricity_delta_mwh"].median()),
                float(rows["fan_delta_mwh"].median()),
                float(rows["gas_hhv_delta_mwh"].median()),
            ])
    main.set_yticks(row_positions, row_labels)
    main.set_xticks([-10, 0, 10, 20, 30])
    main.set_xlabel("ΔCO₂ (tCO₂ yr⁻¹)")
    main.grid(True, axis="x", color="#D9D9D9", linewidth=0.35)
    matrix = np.asarray(carrier_matrix)[np.argsort(row_positions)]
    vmax = max(1.0, float(np.nanquantile(np.abs(matrix), 0.95)))
    carrier.imshow(matrix, cmap=SIGNED_CMAP, vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
    carrier.set_xticks(range(4), ["Heat", "Cool", "Fan", "Gas"], rotation=45, ha="right")
    carrier.set_yticks([])
    carrier.set_xlabel("Δ carrier (MWh yr⁻¹)")
    carrier.grid(False)
    carrier.text(0.98, 0.98, f"±{vmax:.0f}", transform=carrier.transAxes, ha="right", va="top")
    g.text(0.255, 0.89, "Carbon", ha="center", va="center")
    g.text(0.785, 0.89, "Carriers", ha="center", va="center")

    # h: context heatmaps of non-additive eligibility interaction.
    interaction = pd.read_csv(panel_directory / "Fig4h.csv")
    _parent_for_children(h, "h", "Centre×spread interaction")
    child_bounds = [[0.08, 0.66, 0.38, 0.18], [0.58, 0.66, 0.38, 0.18], [0.08, 0.42, 0.38, 0.18], [0.58, 0.42, 0.38, 0.18]]
    medians = interaction.groupby(["context", "lambda", "delta_c"], as_index=False)["eligible_count_interaction"].median()
    vlim = max(1.0, float(medians["eligible_count_interaction"].abs().max()))
    for panel_index, (context, bounds) in enumerate(zip(CONTEXTS, child_bounds)):
        sub = medians[medians["context"] == context].pivot(index="lambda", columns="delta_c", values="eligible_count_interaction").sort_index().sort_index(axis=1)
        child = _child(h, bounds)
        child.imshow(sub.to_numpy(), cmap=SIGNED_CMAP, vmin=-vlim, vmax=vlim, origin="lower", aspect="auto")
        child.set_xticks(range(len(sub.columns)), [f"{x:g}" for x in sub.columns])
        child.set_yticks(range(len(sub.index)), [f"{x:g}" for x in sub.index])
        if panel_index < 2:
            child.set_xticklabels([])
        if panel_index % 2 == 1:
            child.set_yticklabels([])
        child.grid(False)
        child.text(0.02, 0.98, CONTEXT_LABELS[context], transform=child.transAxes, va="top")
    gradient = np.linspace(-vlim, vlim, 256)[None, :]
    h.imshow(
        gradient,
        cmap=SIGNED_CMAP,
        vmin=-vlim,
        vmax=vlim,
        aspect="auto",
        extent=(0.20, 0.80, 0.23, 0.255),
        transform=h.transAxes,
        origin="lower",
    )
    h.add_patch(Rectangle((0.20, 0.23), 0.60, 0.025, fill=False, edgecolor="#555555", linewidth=0.65))
    h.text(0.18, 0.242, f"−{vlim:.0f}", ha="right", va="center")
    h.text(0.50, 0.242, "0", ha="center", va="center")
    h.text(0.82, 0.242, f"+{vlim:.0f}", ha="left", va="center")
    _critical(h.text(0.50, 0.010, "Eligibility ≠0: 1,261/1,728\nEnergy ≠0: 545/585; unresolved 1,143", ha="center", va="bottom"), "4h-denominators")
    return figure


def build_figure_5(panel_directory: Path | str):
    """Build Figure 5: event flexibility under service and recovery constraints."""

    panel_directory = Path(panel_directory)
    verify_panel_inputs(panel_directory, 5)
    figure, axes = create_panel_figure()
    a, b, c, d, e, f, g, h = axes.ravel()

    # a: frozen event and recovery contract.
    _blank(a, "a", "Event and recovery contract")
    a.plot([0.08, 0.93], [0.60, 0.60], color="#555555", linewidth=1.1)
    a.add_patch(Rectangle((0.10, 0.48), 0.16, 0.24, facecolor=PALETTE["orange"], alpha=0.35, edgecolor="none"))
    a.add_patch(Rectangle((0.26, 0.48), 0.64, 0.24, facecolor=PALETTE["sky"], alpha=0.18, edgecolor="none"))
    a.text(0.18, 0.60, "2-h event", ha="center", va="center")
    a.text(0.58, 0.60, "48-h recovery", ha="center", va="center")
    a.text(0.10, 0.37, "0 h", ha="center")
    a.text(0.26, 0.37, "2 h", ha="center")
    a.text(0.90, 0.37, "50 h", ha="center")
    a.text(0.18, 0.84, "response", ha="center", color=PALETTE["orange"])
    a.text(0.58, 0.84, "rebound + recovery service", ha="center", color=PALETTE["blue"])
    a.text(0.50, 0.17, "50-h net energy = event saving - recovery extra", ha="center")

    # b: same action baseline and perturbed power traces.
    power = pd.read_csv(panel_directory / "Fig5b_power_trace.csv")
    for state, color, linestyle in [("baseline", PALETTE["dark_grey"], "--"), ("perturbed", PALETTE["blue"], "-")]:
        rows = power[power["state"] == state].sort_values("hour")
        b.plot(rows["hour"], rows["hvac_electric_power_kw"], color=color, linestyle=linestyle, label=state.capitalize())
    _event_span(b)
    base = power[power["state"] == "baseline"].sort_values("hour")
    pert = power[power["state"] == "perturbed"].sort_values("hour")
    difference = base["hvac_electric_power_kw"].to_numpy() - pert["hvac_electric_power_kw"].to_numpy()
    response_kw = float(difference[:2].mean())
    rebound_kw = float((-difference[2:]).max())
    net_kwh = float(difference.sum())
    b.text(0.98, 0.96, f"same action\nresponse {response_kw:.1f} kW\nrebound {rebound_kw:.1f} kW\nnet {net_kwh:.1f} kWh", transform=b.transAxes, ha="right", va="top")
    b.set_xlim(0, 49)
    b.set_ylim(0, max(power["hvac_electric_power_kw"]) * 1.18)
    b.set_xlabel("Hour from event start")
    b.set_ylabel("HVAC power (kW)")
    _panel(b, "b", "Fixed-action power trajectory")
    b.legend(frameon=False, loc="lower right", ncol=2, handlelength=1.5)
    _inset_panel_axis(b, left=0.030, bottom=0.010)

    # c: temperature envelope and service trajectories on aligned child axes.
    trace = pd.read_csv(panel_directory / "Fig5c_temperature_service_trace.csv")
    _parent_for_children(c, "c", "Temperature and service")
    temp_ax = _child(c, [0.18, 0.55, 0.79, 0.31])
    service_ax = _child(c, [0.18, 0.12, 0.79, 0.31])
    for state, color, linestyle in [("baseline", PALETTE["dark_grey"], "--"), ("perturbed", PALETTE["blue"], "-")]:
        rows = trace[trace["state"] == state].sort_values("hour")
        temp_ax.fill_between(rows["hour"], rows["zone_min_c"], rows["zone_max_c"], color=color, alpha=0.10, linewidth=0)
        temp_ax.plot(rows["hour"], rows["occupancy_weighted_temp_c"], color=color, linestyle=linestyle)
        service_ax.plot(rows["hour"], rows["full_mean_service"], color=color, linestyle=linestyle)
        service_ax.plot(rows["hour"], rows["fixed27_tail_service"], color=color, linestyle=":")
    mean_budget = float(trace["mean_service_budget_c"].iloc[0])
    tail_budget = float(trace["tail_service_budget_c"].iloc[0])
    service_ax.axhline(mean_budget, color=PALETTE["dark_grey"], linewidth=0.65)
    service_ax.axhline(tail_budget, color=PALETTE["vermillion"], linewidth=0.65)
    for child in (temp_ax, service_ax):
        _event_span(child)
        child.set_xlim(0, 49)
        child.set_xticks([0, 10, 20, 30, 40])
    temp_ax.set_yticks([23, 25, 27])
    service_ax.set_yticks([2, 4, 6])
    temp_ax.set_ylabel("Temp. (°C)")
    temp_ax.set_xticklabels([])
    service_ax.set_ylabel("Service (°C)")
    service_ax.set_xlabel("Hour from event start")
    _critical(temp_ax.text(0.98, 0.93, "line: occupied mean\nshade: zone range", transform=temp_ax.transAxes, ha="right", va="top"), "5c-temperature-key")
    _critical(service_ax.text(0.98, 0.93, "mean service; tail dotted", transform=service_ax.transAxes, ha="right", va="top"), "5c-service-key")

    # d/e: population-level qualification and false acceptance.
    july = pd.read_csv(panel_directory / "Fig5d.csv")
    for rep, label in [("full", "Full demand"), ("mean", "Mean")]:
        rows = july[july["representation"] == rep].sort_values("lambda")
        style = CATEGORY_STYLES["full_demand" if rep == "full" else "mean"]
        d.plot(rows["lambda"], rows["positive_selected_count"], label=label, **style)
        for row in rows.to_dict("records"):
            offset: int | tuple[int, int] = -15 if rep == "full" and row["lambda"] <= 0.5 else (7 if rep == "full" else 12)
            horizontal_alignment = "center"
            if rep == "mean" and row["lambda"] <= 0.5:
                offset = 13
            if rep == "full" and row["lambda"] == 1.5:
                offset = (0, 11)
            xytext = (0, offset) if isinstance(offset, int) else offset
            _critical(d.annotate(f"{row['positive_selected_count']}/144", (row["lambda"], row["positive_selected_count"]), xytext=xytext, textcoords="offset points", ha=horizontal_alignment, va="center", clip_on=True), f"5d-{rep}-{row['lambda']}")
    d.set_xlim(-0.20, 1.70)
    d.set_ylim(0, 220)
    d.set_xticks([0, 0.5, 1, 1.5])
    d.set_xlabel("Demand spread λ")
    d.set_ylabel("Positive selections (count)")
    _panel(d, "d", "July response availability")
    legend = d.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        handlelength=1.5,
    )
    legend.set_gid("figure-5d-legend")
    d.set_yticks([0, 50, 100, 150])
    _inset_panel_axis(d, left=0.035, bottom=0.008)

    july_false = pd.read_csv(panel_directory / "Fig5e.csv").sort_values("lambda")
    bars = e.bar(july_false["lambda"], july_false["false_accept_count"], width=0.28, color=PALETTE["vermillion"], edgecolor="white")
    for bar, row in zip(bars, july_false.itertuples(index=False)):
        e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4, f"{row.false_accept_count}/122\n{100*row.false_accept_rate:.0f}%", ha="center", va="bottom")
    e.set_xlim(-0.25, 1.75)
    e.set_ylim(0, 140)
    e.set_xticks([0, 0.5, 1, 1.5])
    e.text(0.98, 0.04, "λ", transform=e.transAxes, ha="right", va="bottom")
    e.set_ylabel("False accepts (count)")
    _panel(e, "e", "False flexibility acceptance")

    # f: paired response changes in the 12 common cells.
    paired = pd.read_csv(panel_directory / "Fig5f.csv")
    wide = paired.pivot(index="decision_id", columns="lambda", values="selected_response_kw")
    for _, row in wide.sort_values(1.0).iterrows():
        color = PALETTE["blue"] if row[1.5] < row[1.0] - 1e-9 else PALETTE["grey"]
        f.plot([1, 1.5], [row[1.0], row[1.5]], color=color, alpha=0.75, linewidth=1.0)
        f.scatter([1, 1.5], [row[1.0], row[1.5]], color=color, s=14)
    deltas = wide[1.5] - wide[1.0]
    f.text(0.04, 0.96, f"5 decrease; 7 tie\nmax decrease {abs(deltas.min()):.3f} kW", transform=f.transAxes, ha="left", va="top")
    f.set_xlim(0.88, 1.62)
    f.set_xticks([1, 1.5], ["λ=1", "λ=1.5"])
    f.set_ylabel("Response (kW)")
    _panel(f, "f", "Response in common cells")
    _inset_panel_axis(f, left=0.025, bottom=0.005)

    # g: four recovery/energy endpoints in aligned microplots.
    recovery = pd.read_csv(panel_directory / "Fig5g.csv")
    _parent_for_children(g, "g", "Recovery and net energy")
    metrics = [
        ("delta_response_kw", "Response (kW)"),
        ("delta_recovery_peak_kw", "Recovery peak (kW)"),
        ("delta_recovery_extra_kwh", "Recovery extra (kWh)"),
        ("delta_50h_net_saving_kwh", "50-h net (kWh)"),
    ]
    bounds = [[0.16, 0.55, 0.32, 0.15], [0.61, 0.55, 0.36, 0.15], [0.16, 0.07, 0.32, 0.15], [0.61, 0.07, 0.36, 0.15]]
    header_positions = [(0.32, 0.90), (0.79, 0.90), (0.32, 0.40), (0.79, 0.40)]
    for metric_index, ((metric, label), bound) in enumerate(zip(metrics, bounds)):
        child = _child(g, bound)
        values = recovery[metric].to_numpy(float)
        yy = np.linspace(-0.12, 0.12, len(values))
        child.axvline(0, color="#333333", linewidth=0.65)
        child.scatter(values, yy, color=np.where(values < -1e-9, PALETTE["blue"], np.where(values > 1e-9, PALETTE["vermillion"], PALETTE["grey"])), s=13)
        child.set_yticks([])
        child.set_ylim(-0.25, 0.25)
        limit = max(1.0, float(np.max(np.abs(values))))
        tick_limit = float(np.ceil(limit / 5) * 5)
        child.set_xlim(-1.15 * tick_limit, 1.15 * tick_limit)
        child.set_xticks([-tick_limit, 0, tick_limit], [f"{-tick_limit:g}", "0", f"{tick_limit:g}"])
        child.set_xlabel("")
        lower = int((values < -1e-9).sum())
        tied = int((np.abs(values) <= 1e-9).sum())
        higher = int((values > 1e-9).sum())
        hx, hy = header_positions[metric_index]
        _critical(g.text(hx, hy, f"{label}\n{lower}− | {tied}= | {higher}+", ha="center", va="top"), f"5g-header-{metric_index}")
        child.grid(True, axis="x", color="#D9D9D9", linewidth=0.35)

    # h: July and February counterfactual grids.
    boundary = pd.read_csv(panel_directory / "Fig5h.csv")
    _parent_for_children(h, "h", "Season-location boundary")
    heat_bounds = [[0.05, 0.44, 0.41, 0.42], [0.56, 0.44, 0.41, 0.42]]
    vmax = int(boundary["full_positive_selected_count"].max())
    for month, bound, label in [(7, heat_bounds[0], "July"), (2, heat_bounds[1], "February")]:
        child = _child(h, bound)
        sub = boundary[boundary["month"] == month].pivot(index="lambda", columns="delta_c", values="full_positive_selected_count").sort_index().sort_index(axis=1)
        child.imshow(sub.to_numpy(), cmap=COUNT_CMAP, vmin=0, vmax=vmax, origin="lower", aspect="auto")
        child.set_xticks(range(len(sub.columns)), [f"{x:g}" for x in sub.columns])
        child.set_yticks(range(len(sub.index)), [f"{x:g}" for x in sub.index])
        child.text(0.03, 0.97, label, transform=child.transAxes, va="top", color="white")
        child.grid(False)
    h.text(0.50, 0.28, "Demand shift (°C); rows are λ", ha="center", va="center")
    h.text(0.50, 0.18, "July: ref 51/144; grid 1,247/2,880", ha="center", va="center")
    h.text(0.50, 0.02, "February: ref 0/144; grid 210/2,880", ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "#555555", "pad": 1})
    return figure


def build_figure_6(panel_directory: Path | str):
    """Build Figure 6: protected anonymous demand summaries."""

    panel_directory = Path(panel_directory)
    verify_panel_inputs(panel_directory, 6)
    figure, axes = create_panel_figure()
    a, b, c, d, e, f, g, h = axes.ravel()

    # a: conceptual sufficiency phase space.
    _blank(a, "a", "Is the mean sufficient?")
    a.add_patch(Polygon([[0.12, 0.14], [0.86, 0.14], [0.12, 0.82]], closed=True, facecolor=CLASS_COLOURS["mean-safe"], alpha=0.65, edgecolor="white"))
    a.add_patch(Polygon([[0.86, 0.14], [0.86, 0.82], [0.12, 0.82]], closed=True, facecolor=CLASS_COLOURS["distribution-required"], alpha=0.65, edgecolor="white"))
    a.add_patch(Rectangle((0.12, 0.70), 0.74, 0.12, facecolor=CLASS_COLOURS["finite-library-unresolved"], hatch="///", alpha=0.85, edgecolor="#777777"))
    a.plot([0.12, 0.86], [0.14, 0.14], color="#333333", linewidth=0.65)
    a.plot([0.12, 0.12], [0.14, 0.82], color="#333333", linewidth=0.65)
    a.text(0.49, 0.05, "Demand heterogeneity SD (°C)", ha="center")
    a.text(
        0.02,
        0.48,
        "Service-budget multiplier\n(looser allowance ↑)",
        rotation=90,
        va="center",
    )
    a.text(0.30, 0.31, "mean-safe", ha="center")
    a.text(0.64, 0.55, "distribution-required", ha="center")
    a.text(0.49, 0.76, "unresolved", ha="center")

    # b: exact simulated design cells in four contexts.
    phase = pd.read_csv(panel_directory / "Fig6b.csv")
    _parent_for_children(b, "b", "Sufficiency phase map")
    child_bounds = [[0.07, 0.45, 0.40, 0.22], [0.57, 0.45, 0.40, 0.22], [0.07, 0.10, 0.40, 0.22], [0.57, 0.10, 0.40, 0.22]]
    markers = {"mean-safe": "o", "distribution-required": "s", "finite-library-unresolved": "x"}
    for panel_index, (context, bound) in enumerate(zip(CONTEXTS, child_bounds)):
        child = _child(b, bound)
        rows = phase[phase["context"] == context]
        for classification in ["mean-safe", "distribution-required", "finite-library-unresolved"]:
            sub = rows[rows["classification"] == classification]
            child.scatter(sub["heterogeneity_sd_c"], sub["service_fraction"], s=9, color=CLASS_COLOURS[classification], marker=markers[classification], alpha=0.90 if classification == "finite-library-unresolved" else 0.85, linewidths=0.5)
        child.set_xlim(-0.08, 4.35)
        child.set_ylim(0.46, 1.04)
        child.set_title(CONTEXT_LABELS[context], loc="left", fontsize=11, fontweight="normal", pad=0)
        child.set_xticks([0, 2, 4])
        child.set_yticks([0.5, 0.75, 1.0])
        if panel_index < 2:
            child.set_xticklabels([])
        if panel_index % 2 == 1:
            child.set_yticklabels([])
    counts = phase["classification"].value_counts()
    b.legend(
        handles=[
            Line2D([], [], color=CLASS_COLOURS[key], marker=markers[key], linestyle="", markersize=5)
            for key in ["mean-safe", "distribution-required", "finite-library-unresolved"]
        ],
        labels=[
            f"Safe {counts['mean-safe']}",
            f"Distribution {counts['distribution-required']}",
            f"Unresolved {counts['finite-library-unresolved']}",
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        handlelength=0.8,
        handletextpad=0.2,
        columnspacing=0.45,
    )

    # c: all cells and resolved subset.
    summary = pd.read_csv(panel_directory / "Fig6c.csv").set_index("classification")
    _panel(c, "c", "Outcome shares", grid_axis="x")
    orders = ["mean-safe", "distribution-required", "finite-library-unresolved"]
    labels = {"mean-safe": "Mean-safe", "distribution-required": "Distribution", "finite-library-unresolved": "Unresolved"}
    left = 0.0
    for key in orders:
        count = int(summary.loc[key, "count"])
        width = 100 * count / 1008
        c.barh([1], [width], left=[left], color=CLASS_COLOURS[key], hatch="///" if key == "finite-library-unresolved" else None, edgecolor="white", label=labels[key])
        c.text(left + width / 2, 1, f"{count}\n{width:.1f}%", ha="center", va="center")
        left += width
    left = 0.0
    for key in ["mean-safe", "distribution-required"]:
        count = int(summary.loc[key, "count"])
        width = 100 * count / 606
        c.barh([0], [width], left=[left], color=CLASS_COLOURS[key], edgecolor="white")
        c.text(left + width / 2, 0, f"{count}\n{width:.1f}%", ha="center", va="center")
        left += width
    _inset_panel_axis(c, left=0.025, bottom=0.025)
    c.set_yticks([1, 0], ["All", "Resolved"])
    c.set_xlim(0, 100)
    c.set_xlabel("Design units (%)")

    # d: threshold errors retain direction and context denominator.
    threshold = pd.read_csv(panel_directory / "Fig6d.csv")
    contexts = [x for x in CONTEXTS if x in set(threshold["context"])] + ["overall"]
    y = np.arange(len(contexts))[::-1]
    false_safe = []
    false_required = []
    denominators = []
    for context in contexts:
        row = threshold[threshold["context"] == context].iloc[0]
        false_safe.append(100 * row["false_safe"] / row["denominator"])
        false_required.append(100 * row["false_required"] / row["denominator"])
        denominators.append(int(row["denominator"]))
    d.barh(y, false_safe, color=PALETTE["vermillion"], label="False-safe")
    d.barh(y, false_required, left=false_safe, color=PALETTE["sky"], label="False-required")
    overall = float(threshold.loc[threshold["context"] == "overall", "error_rate"].iloc[0] * 100)
    d.axvline(overall, color="#333333", linewidth=0.8, linestyle="--")
    d.set_yticks(y, ["MIA", "NDL", "PTA", "SEA", "Overall"])
    d.set_xlim(0, 24)
    d.set_xlabel("Classification error (%)")
    d.set_ylabel("")
    for yy, fs, fr, den in zip(y, false_safe, false_required, denominators):
        d.text(fs + fr + 0.4, yy, f"n={den}", va="center")
    d.set_ylim(-0.6, 5.2)
    d.text(
        0.98,
        0.94,
        "in-sample scan 1.059 °C | all 15.2% | max 20.4%",
        transform=d.transAxes,
        ha="right",
        va="top",
    )
    _panel(d, "d", "Threshold residual error", grid_axis="x")
    _inset_panel_axis(d, left=0.045, bottom=0.025)

    # e: approximation quality in two aligned metrics.
    approx = pd.read_csv(panel_directory / "Fig6e.csv")
    _parent_for_children(e, "e", "Approximation quality")
    reps = ["mean_only", "quantile3", "quantile5"]
    rep_labels = ["Mean", "q3", "q5"]
    rep_colours = [REPRESENTATION_COLORS[key] for key in ["mean_only", "quantile3", "quantile5"]]
    for metric, label, bound, scale in [
        ("candidate_tail_mae_c", "Tail-service MAE (°C)", [0.05, 0.13, 0.42, 0.72], 1.0),
        ("eligibility_disagreement", "Eligibility disagreement (%)", [0.56, 0.13, 0.42, 0.72], 100.0),
    ]:
        child = _child(e, bound)
        for index, (rep, color) in enumerate(zip(reps, rep_colours)):
            values = approx.loc[approx["representation"] == rep, metric].to_numpy(float) * scale
            q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            child.plot([index, index], [q1, q3], color=color, linewidth=2)
            child.scatter([index], [med], color=color, s=26, zorder=3)
            child.text(index, q3, f"{med:.2f}", ha="center", va="bottom")
        child.set_xticks(range(3), rep_labels)
        child.set_xlim(-0.45, 2.45)
        child.set_ylim(0, child.get_ylim()[1] * 1.22)
    e.text(0.24, 0.93, "Tail MAE (°C)", ha="center", va="top")
    e.text(0.76, 0.93, "Eligibility error (%)", ha="center", va="top")

    # f: protected certification trade-off, split facets.
    gate = pd.read_csv(panel_directory / "Fig6f.csv")
    _parent_for_children(f, "f", "Protected-quantile trade-off")
    split_bounds = [[0.09, 0.14, 0.38, 0.71], [0.58, 0.14, 0.40, 0.71]]
    split_labels = {
        "leave_one_context_seed_out": "Context-seed",
        "leave_one_population_out": "Population",
    }
    rep_styles = {
        "mean_only": (REPRESENTATION_COLORS["mean_only"], "s"),
        "quantile3": (REPRESENTATION_COLORS["quantile3"], "o"),
        "quantile5": (REPRESENTATION_COLORS["quantile5"], "^"),
    }
    for split, bound in zip(split_labels, split_bounds):
        child = _child(f, bound)
        rows = gate[gate["split_scheme"] == split]
        for rep, (color, marker) in rep_styles.items():
            sub = rows[rows["representation"] == rep]
            sizes = 15 + 65 * sub["decision_coverage"].to_numpy(float)
            yvals = np.log10(sub["false_accept_candidates"].to_numpy(float) + 1)
            child.scatter(100 * sub["true_action_retention"], yvals, s=sizes, color=color, marker=marker, alpha=0.72, edgecolors="white", linewidths=0.4)
        highlight = rows[(rows["representation"] == "quantile5") & (rows["margin_rule"] == "pooled_q95")].iloc[0]
        hx = 100 * highlight["true_action_retention"]
        hy = np.log10(highlight["false_accept_candidates"] + 1)
        child.scatter([hx], [hy], s=105, facecolors="none", edgecolors="#111111", linewidths=1.0)
        child.annotate(f"q5+q95\n{int(highlight['true_positive_candidates']):,}/{int(highlight['full_eligible_candidates']):,}\n{int(highlight['full_available_and_certified_decisions'])}/{int(highlight['full_available_decisions'])}", (hx, hy), xytext=(-3, 7), textcoords="offset points", ha="right", va="bottom")
        child.set_xlim(-4, 104)
        child.set_xticks([0, 50, 100])
        child.set_ylim(-0.15, np.log10(gate["false_accept_candidates"].max() + 1) + 0.15)
        child.set_yticks([0, 1, 2, 3, 4], ["0", "9", "99", "999", "9,999"])
        child.set_xlabel("")
        child.set_ylabel("")
        child.text(0.03, 0.97, split_labels[split], transform=child.transAxes, ha="left", va="top")
    f.text(0.50, 0.94, "x retention (%) | y false accepts (log) | size coverage", ha="center", va="top")

    # g: local anonymous update learning curves.
    learning = pd.read_csv(panel_directory / "Fig6g.csv")
    _parent_for_children(g, "g", "Local feedback")
    feedback_deltas = []
    for metric, label, bound in [("CRPS", "Normalized RPS", [0.10, 0.32, 0.36, 0.35]), ("tail_MAE", "Tail-share MAE", [0.61, 0.32, 0.36, 0.35])]:
        child = _child(g, bound)
        for split, color, marker in [("city", PALETTE["blue"], "o"), ("source", PALETTE["orange"], "s")]:
            rows = learning[(learning["metric"] == metric) & (learning["transfer_split"] == split)].sort_values("feedback_n")
            child.errorbar(rows["feedback_n"], rows["estimate"], yerr=[rows["estimate"] - rows["ci_low"], rows["ci_high"] - rows["estimate"]], color=color, marker=marker, capsize=2, label=split.capitalize())
            row0 = rows[rows["feedback_n"] == 0].iloc[0]
            row40 = rows[rows["feedback_n"] == 40].iloc[0]
            if split == "city":
                feedback_deltas.append(float(row40["estimate"] - row0["estimate"]))
        child.set_xticks([0, 5, 10, 20, 40])
        child.set_xlabel("")
        child.set_ylabel("")
    g.text(0.28, 0.73, "Normalized RPS", ha="center", va="center")
    g.text(0.79, 0.73, "Tail-share MAE", ha="center", va="center")
    g.legend(
        handles=[
            Line2D([], [], color=PALETTE["blue"], marker="o", label="City"),
            Line2D([], [], color=PALETTE["orange"], marker="s", label="Source"),
        ],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        handlelength=1.2,
        columnspacing=0.8,
    )
    _critical(g.text(0.50, 0.035, f"City Δ40: RPS {feedback_deltas[0]:+.3f}; tail {feedback_deltas[1]:+.3f}", ha="center", va="bottom"), "6g-feedback-footer")

    # h: constructive anonymous-summary workflow without a formal privacy guarantee.
    workflow = json.loads((panel_directory / "Fig6h_meta.json").read_text(encoding="utf-8"))["steps"]
    _blank(h, "h", "Qualified reporting workflow")
    positions = [(0.03, 0.61), (0.37, 0.61), (0.71, 0.61), (0.12, 0.29), (0.62, 0.29)]
    short = ["Archive prior\nanonymous", "Local update\nroom/building", "Five quantiles\nanonymous", "Protected rule\nq95 margin", "Report outcomes\n+ recovery"]
    colours = ["#F2F2F2", "#E8F2F8", "#DDF2EB", "#DDF2EB", "#FDE9DF"]
    for (x, y), label, color, _step in zip(positions, short, colours, workflow):
        _box(h, (x, y), 0.26, 0.20, label, facecolor=color)
    _arrow(h, (0.29, 0.71), (0.37, 0.71))
    _arrow(h, (0.63, 0.71), (0.71, 0.71))
    _arrow(h, (0.84, 0.60), (0.37, 0.50))
    _arrow(h, (0.38, 0.39), (0.62, 0.39))
    h.text(0.50, 0.10, "Report representation + mean/tail service\naction range + recovery window", ha="center", va="center")
    return figure


CAPTIONS = {
    4: (
        "Figure 4 | Demand spread contracts service-qualified annual action space. "
        "a, The same fixed action is audited under a compressed mean and the complete demand distribution. "
        "b, Fixed-centre spread perturbations for the pre-specified Miami multinational composition; horizontal segments show P10-P90. "
        "c, Full-demand and mean-based annual selections among 144 design cells at each spread multiplier. "
        "d, Mean-selected actions that fail full-demand review (denominator 128). "
        "e, Full-service eligible-action counts across all 144 cells per multiplier, retaining zero-action cells. "
        "f, Median and interquartile annual-energy change for bilateral full-service selections; the right-hand table gives matched/unmatched cells for λ=0, 0.5 and 1.5. "
        "g, Carbon intervals and median signed carrier changes for the same bilateral comparisons; colours and markers retain the λ encoding from panel f. "
        "h, Median non-additive eligibility interaction across contexts on one shared diverging action-count scale; selected-energy interactions are resolved in 585 of 1,728 cells."
    ),
    5: (
        "Figure 5 | Demand heterogeneity contracts service-qualified flexibility under event and recovery constraints. "
        "a, Pre-specified 2-h event, 48-h recovery and 50-h energy-accounting contract. "
        "b, Baseline and perturbed HVAC power for the same pre-specified action; the panel does not compare mean- and full-selected actions. "
        "c, Zone temperature envelope, occupancy-weighted temperature and full-distribution mean and fixed-27 tail-service trajectories for that case; horizontal lines are the fixed mean- and tail-service budgets. "
        "d, July cells with a service-qualified positive response under full and mean demand. "
        "e, Mean-positive selections that fail full-demand review (denominator 122). "
        "f, Selected response for 12 common full-service cells at spread multipliers 1 and 1.5. "
        "g, Response, recovery peak, recovery electricity and 50-h net-saving changes for those 12 cells; each header reports decrease, tie and increase counts. "
        "h, Positive full-service selections across demand location and spread; the February central reference (0/144) is distinct from the complete February grid (210/2,880)."
    ),
    6: (
        "Figure 6 | Protected anonymous demand summaries recover service-qualified decisions beyond a single mean. "
        "a, Schematic conditions governing whether an average is sufficient; the vertical axis is the service-budget multiplier, for which larger values permit looser service-error allowances, and the drawn regions do not define an empirical monotonic boundary. "
        "b, Exact sampled design cells classified across four contexts without spatial interpolation. "
        "c, Counts among all 1,008 design units and the 606 resolvable units. "
        "d, False-safe and false-required errors from the lowest-error 1.059 °C in-sample scanned diagnostic threshold; it did not meet the prespecified total-error, worst-context-error, sensitivity and specificity standards, and denominators are context-specific. "
        "e, Median and interquartile service-reconstruction errors for mean, three-quantile and five-quantile summaries. "
        "f, Held-out certification trade-off; marker shape identifies the anonymous summary and marker size encodes decision coverage. The highlighted q5 plus training-only pooled-q95 rule has zero observed false acceptance, retains 14,709/18,766 true-qualified candidates and covers 542/606 resolvable decisions in the context-seed split. "
        "g, Local-update learning curves; points and error bars show updated scores and their hierarchical-bootstrap 95% intervals. In the paired improvement analysis at 40 observations, both city-held-out improvement intervals are above zero, whereas both source-held-out improvement intervals cross zero. "
        "h, Anonymous-summary workflow from archive prior to qualified energy, carbon and flexibility reporting; anonymity of the summaries is not a formal privacy guarantee."
    ),
}


ALT_TEXT = {
    4: (
        "Eight-panel figure comparing one fixed annual action library under mean and full thermal-demand service checks. "
        "At fixed mean demand, increasing spread λ contracts full-qualified selections from 128/144 to 21/144 while mean-based selections stay 128/144; false acceptance rises from 0/128 to 128/128. "
        "Eligible action counts collapse, and matched annual-energy and carbon effects differ by context and carrier. The matched/unmatched comparison contains 169/263 cells. "
        "Centre×spread eligibility interactions are non-zero in 1,261/1,728 cells; energy interactions are resolved for 585/1,728 cells, with 1,143 unresolved."
    ),
    5: (
        "Eight-panel figure comparing service-qualified demand-response flexibility with a pre-specified event, recovery and energy-accounting contract. "
        "Panels b and c show baseline and perturbed physical trajectories for the same pre-specified action, with 28.3 kW response, 13.1 kW rebound and 30.3 kWh net saving; they do not compare mean- and full-selected actions. "
        "Across July design cells, full-qualified positive selections fall from 122/144 to 12/144 as λ rises, while mean-based selections remain 122/144 and false acceptance reaches 115/122. "
        "Only 12 cells are common at λ=1 and 1.5; five lose response. February has 0/144 positive selections at the central reference but 210/2,880 across the complete grid."
    ),
    6: (
        "Eight-panel figure testing anonymous demand summaries and a protected qualification rule. "
        "Panel a is a schematic separation of mean-safe, distribution-required and unresolved regimes by demand heterogeneity and service-budget multiplier; larger multipliers permit looser service-error allowances, and the drawn regions are not an empirical monotonic boundary. "
        "Panel b maps the three regimes for the exact sampled cells in Miami, New Delhi, Port Angeles and Seattle. "
        "Panel c shows that, of 1,008 design units, 275 are mean-safe, 331 require the demand distribution and 402 are unresolved; the resolved denominator is 606. "
        "Panel d shows that the lowest-error 1.059 °C in-sample scanned diagnostic threshold makes 92/606 errors and does not meet the prespecified performance standards. "
        "Panel e shows lower median tail-service and eligibility errors for three- and five-quantile summaries than for the mean alone. "
        "Panel f shows that, in the context-seed holdout, q5 plus a training-only pooled-q95 margin yields zero observed false accepts, retains 14,709/18,766 true-qualified candidates and covers 542/606 resolvable decisions; this is held-out evidence, not a universal guarantee. "
        "Panel g shows local-feedback learning curves for dimensionless normalized ordinal RPS and tail-share MAE. Their points and error bars are updated scores and their hierarchical-bootstrap 95% intervals. In the paired improvement analysis at 40 city-held-out observations, the RPS and tail-share error improvement intervals are above zero, whereas at 40 source-held-out observations both corresponding improvement intervals cross zero, including the tail-share error interval. "
        "Panel h traces an anonymous-summary qualified-reporting workflow from an archive prior through local updating, five anonymous quantiles and a protected q95 qualification rule to energy, carbon, flexibility and recovery reporting; it makes no formal privacy guarantee."
    ),
}


BUILDERS: dict[int, Callable[[Path | str], plt.Figure]] = {
    4: build_figure_4,
    5: build_figure_5,
    6: build_figure_6,
}


def render_main_figure(
    figure_number: int,
    panel_directory: Path | str,
    output_directory: Path | str,
) -> ExportBundle:
    """Build, export and caption one approved main figure."""

    if figure_number not in BUILDERS:
        raise ValueError("figure number must be 4, 5 or 6")
    output_directory = Path(output_directory)
    figure = BUILDERS[figure_number](panel_directory)
    try:
        bundle = save_figure_bundle(figure, output_directory, f"figure_{figure_number:02d}")
        (output_directory / f"figure_{figure_number:02d}_caption.txt").write_text(
            CAPTIONS[figure_number] + "\n", encoding="utf-8"
        )
        (output_directory / f"figure_{figure_number:02d}_alt_text.txt").write_text(
            ALT_TEXT[figure_number] + "\n", encoding="utf-8"
        )
        return bundle
    finally:
        plt.close(figure)
