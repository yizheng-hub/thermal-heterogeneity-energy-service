#!/usr/bin/env python3
"""Render a non-scientific eight-panel smoke figure for the figure contract."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from thermal_heterogeneity.figures import (
    CATEGORY_STYLES,
    PALETTE,
    create_panel_figure,
    data_safe_legend,
    direct_label,
    draw_world_land,
    save_figure_bundle,
    style_panel,
)


def build_smoke_figure():
    """Build varied placeholder content without reading any scientific data."""

    figure, axes = create_panel_figure()
    flat = axes.ravel()
    x = np.arange(5)

    draw_world_land(flat[0])
    flat[0].scatter([-80, 0, 78], [26, 52, 28], s=[28, 48, 68], color=PALETTE["blue"])
    style_panel(flat[0], "a", "Observed locations", grid_axis=None)

    for name in ("cooler", "no_change", "warmer"):
        style = CATEGORY_STYLES[name]
        flat[1].plot(x, x * (0.35 + 0.1 * len(name)), label=name.replace("_", " "), **style)
    flat[1].set_xticks(x)
    flat[1].set_yticks([0, 2, 4, 6])
    style_panel(flat[1], "b", "Demand categories")
    data_safe_legend(flat[1], location="top", ncol=3)

    flat[2].bar(x, [2, 5, 3, 6, 4], color=PALETTE["uniform"])
    flat[2].set_xticks(x)
    flat[2].set_yticks([0, 2, 4, 6])
    style_panel(flat[2], "c", "Shared-room counts")
    flat[2].set_xlabel("Group")
    flat[2].set_ylabel("Count")

    values = np.array([-2.0, -0.8, 0.0, 0.9, 2.1])
    flat[3].axhline(0, color="#333333", linewidth=0.65)
    flat[3].scatter(x, values, color=[PALETTE["blue"]] * 2 + [PALETTE["grey"]] + [PALETTE["vermillion"]] * 2)
    flat[3].set_xticks(x)
    flat[3].set_yticks([-2, -1, 0, 1, 2])
    style_panel(flat[3], "d", "Signed outcomes")
    flat[3].set_ylabel("Change")

    for offset, name in enumerate(("uniform", "personal_centre", "ltpref")):
        style = CATEGORY_STYLES[name]
        flat[4].plot(x, 2 + offset + np.sin(x), label=name.replace("_", " "), **style)
    flat[4].set_xticks(x)
    flat[4].set_yticks([2, 4, 6])
    style_panel(flat[4], "e", "Control trajectories")
    data_safe_legend(flat[4], location="top", ncol=3)

    matrix = np.arange(25).reshape(5, 5)
    flat[5].imshow(matrix, cmap="cividis", origin="lower", aspect="auto")
    flat[5].set_xticks(x)
    flat[5].set_yticks(x)
    style_panel(flat[5], "f", "Decision surface", grid_axis=None)
    flat[5].set_xlabel("Demand location")
    flat[5].set_ylabel("Demand spread")

    flat[6].plot(x, [1, 1.7, 2.1, 2.45, 2.7], color=PALETTE["green"])
    flat[6].set_xticks(x)
    flat[6].set_yticks([1, 1.5, 2, 2.5])
    direct_label(
        flat[6], 4, 2.7, "Protected quantile", color=PALETTE["green"],
        dx_points=-4, ha="right"
    )
    style_panel(flat[6], "g", "Protected decisions")

    flat[7].barh([0, 1, 2], [0.42, 0.78, 0.89], color=[PALETTE["vermillion"], PALETTE["green"], PALETTE["blue"]])
    flat[7].set_yticks([0, 1, 2], ["False", "Retain", "Cover"])
    flat[7].set_xlim(0, 1)
    flat[7].set_xticks([0, 0.5, 1])
    flat[7].set_xlabel("Fraction")
    style_panel(flat[7], "h", "Qualification summary", grid_axis="x")
    return figure


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/figure_smoke"))
    args = parser.parse_args()
    figure = build_smoke_figure()
    try:
        bundle = save_figure_bundle(figure, args.output, "figure_render_smoke")
        print(bundle.qa)
    finally:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
