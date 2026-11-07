"""Full-page supporting evidence, redrawn from the contained numerical results.

The original panel routines supply the data and encodings. This module changes
their grouping, physical layout and short headings, without replaying images.
"""

from pathlib import Path
import hashlib
import json
import string

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.font_manager import FontProperties

import supplementary_figures as si
from strengthened_figures import export, style, title

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {1: (1, 3), 2: (5, 6, 7), 3: (8, 9), 4: (10, 11, 12)}
REMAINING_SI = {1: 2, 2: 4, 3: 13, 4: 14, 5: 15, 6: 16}


def draw(number, output):
    fonts = style()
    fig = plt.figure(figsize=(179 / 25.4, 210 / 25.4))
    source = si.Sources(ROOT / "Data_for_Figures/output/from_raw", ROOT / "Data_for_Figures/output/strengthening")
    panels, legends, colorbars, origins = [], [], [], []
    original_canvas = si.canvas

    def canvas(count=4, legend_gap=False):
        axes = [fig.add_axes([0.1, 0.1, 0.3, 0.2]) for _ in range(count)]
        panels.extend(axes)
        return fig, axes

    try:
        si.canvas = canvas
        for old_number in GROUPS[number]:
            before_axes, before_legends, before_panels = len(fig.axes), len(fig.legends), len(panels)
            si.FUNCTIONS[old_number](source)
            new_panels = panels[before_panels:]
            origins.extend((old_number, letter) for letter in string.ascii_lowercase[:len(new_panels)])
            legends.extend((old_number, legend) for legend in fig.legends[before_legends:])
            colorbars.extend((old_number, ax) for ax in fig.axes[before_axes:] if ax not in new_panels)
    finally:
        si.canvas = original_canvas

    # Main comparisons occupy the wide first row; diagnostics use three columns.
    if number in (1, 3):
        order = ([8, 9, 4, 5, 6, 7, 0, 1, 2, 3] if number == 1 else list(range(8)))
        top = [[0.145, 0.765, 0.34, 0.18], [0.645, 0.765, 0.34, 0.18]]
        slots = top + [[x, y, 0.19, 0.145] for y in [0.525, 0.275] for x in [0.145, 0.465, 0.785]] + [[0.145, 0.055, 0.34, 0.145], [0.645, 0.055, 0.34, 0.145]]
        if number == 3:
            top = [[0.13, 0.765, 0.32, 0.18], [0.63, 0.765, 0.32, 0.18]]
            slots = top + [[x, 0.445, 0.19, 0.20] for x in [0.145, 0.465, 0.785]] + [[x, 0.115, 0.19, 0.23] for x in [0.145, 0.465, 0.785]]
    else:
        order = list(range(len(panels)))
        slots = [[x, y, 0.19, 0.13] for y in [0.80, 0.565, 0.33, 0.095] for x in [0.145, 0.465, 0.785]]
        if number == 4:
            slots = [[x, y, 0.19, 0.13] for y in [0.80, 0.55, 0.32, 0.09] for x in [0.145, 0.465, 0.785]]
    panel_map = {}
    for letter, index, bounds in zip(string.ascii_lowercase, order, slots):
        ax = panels[index]
        ax.set_position(bounds)
        heading = ax.get_title(loc="left").split(" ", 1)[1]
        heading = heading.replace("\n", ": ")
        short = {
            "Later requests: Repeated calibration": "Later requests, repeated",
            "Acceptance: Repeated calibration": "Acceptance, repeated",
            "Later requests: Single-record calibration": "Later requests, single",
            "Acceptance: Single-record calibration": "Acceptance, single",
            "Individual minus building": "Individual information",
            "Center prediction error": "Later-center error",
            "Magnitude of exceedance": "Service exceedance",
            "Feasible settings out of 54": "Service frontier",
            "Annual operation support": "Operation support",
            "End-use relationship": "Heating and cooling",
            "Mean choices exceeding individual service limits": "Service exceedances",
            "Where averaging matters": "Service exceedances",
            "Hourly electricity residual": "Electricity residual",
            "Repeated-record support": "Repeated records",
        }.get(heading, heading)
        # A one-line heading is preferred whenever its measured width fits.
        renderer = fig.canvas.get_renderer()
        heading_width = renderer.get_text_width_height_descent(
            f"({letter}) {short}", FontProperties(family="Times New Roman", size=10, weight="bold"), False
        )[0]
        row_neighbors = [slot[0] for slot in slots if abs(slot[1] - bounds[1]) < 1e-10 and slot[0] > bounds[0]]
        heading_right = min(row_neighbors) - 0.012 if row_neighbors else 0.993
        available = (heading_right - bounds[0]) * fig.bbox.width
        if heading_width > available:
            words = short.split()
            middle = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
            short = " ".join(words[:middle]) + "\n" + " ".join(words[middle:])
        title(ax, letter, short)
        panel_map[letter] = {"source_figure": origins[index][0], "source_panel": origins[index][1]}
        if ax.get_xlabel() == "Choice-disagreement frequency":
            ax.set_xlabel("Choice-disagreement\nfrequency")
        if ax.get_xlabel() == "Reassignment allowance (%)":
            ax.set_xlabel("Reassignment\nallowance (%)")
        ax.tick_params(axis="both", pad=2)
        if ax.get_xscale() == "linear" and len(ax.get_xticks()) > 6:
            ax.xaxis.set_major_locator(MaxNLocator(3))
        if origins[index][0] == 3 and origins[index][1] in "abcd":
            ax.set_yticks(ax.get_yticks(), [t.get_text().replace(" + ", "\n+ ") for t in ax.get_yticklabels()])
        for coordinate, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            ticks = coordinate.get_majorticklocs()
            inside = (ticks >= min(limits) - 1e-10) & (ticks <= max(limits) + 1e-10)
            if not inside.all():
                coordinate.set_ticks(ticks[inside])
    for old_number, axis in colorbars:
        parent = panels[origins.index((old_number, "c" if old_number == 5 else "a"))]
        bounds = parent.get_position()
        if old_number == 5:
            parent.set_position([bounds.x0, bounds.y0, bounds.width - 0.035, bounds.height])
            bounds = parent.get_position()
        axis.set_position([bounds.x1 + 0.012, bounds.y0, 0.012, bounds.height])
    for old_number, legend in legends:
        if number == 2 and old_number == 6:
            panels[6].legend(legend.legend_handles, [t.get_text() for t in legend.get_texts()], loc="upper left", handlelength=0.8)
            legend.remove()
            continue
        legend.set_loc("center")
        legend.set_ncols(2 if len(legend.get_texts()) == 2 else 3)
        legend.set_bbox_to_anchor({1: (0.55, 0.018), 2: (0.55, 0.502 if old_number == 6 else 0.027), 3: (0.55, 0.695 if old_number == 8 else 0.035), 4: (0.55, 0.73)}[number], transform=fig.transFigure)
    if number == 1:
        panels[0].set_xlim(10, 55)
        legend = panels[0].get_legend()
        panels[0].legend(legend.legend_handles, [t.get_text() for t in legend.get_texts()],
                         loc="upper right", handlelength=0.75, handletextpad=0.35,
                         borderaxespad=0.25)
    if number == 4:
        panels[5].set_position([0.785, 0.535, 0.19, 0.145])
        panel = panels[6]
        panel.set_ylim(0, 1.6)
        panel.get_legend().set_loc("upper left")
        panel.tick_params(axis="x", pad=5)
        for line in panel.lines:
            if len(line.get_xdata()) == 2 and all(x == 0 for x in line.get_xdata()):
                line.set_ydata([0, 1 / 1.6])
        colorbars[0][1].set_yticks([0, 200])
    if number == 3:
        panels[7].set_xlim(0.94, 1.06)
        panels[7].set_ylim(0, 1.5)
        panels[7].get_legend().set_loc("upper right")
    if number == 2:
        panels[0].set_ylim(0, 100)
        lo, hi = panels[6].get_ylim()
        panels[6].set_ylim(lo, hi + 0.50 * (hi - lo))
    # Display actual cell counts where a matrix encodes discrete settings/events.
    if number in (2, 4):
        matrix_axis = panels[2 if number == 2 else 4]
        values = matrix_axis.images[0].get_array()
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = int(values[row, column])
                if value:
                    matrix_axis.text(column, row, str(value), ha="center", va="center", color="white" if value > values.max() * 0.6 else "black")
    output = Path(output)
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("Publication output must remain inside the project")
    try:
        export(fig, output, 200 + number, source.paths, string.ascii_lowercase[:len(panels)], fonts)
    finally:
        for extension in ["png", "pdf", "svg", "json"]:
            old = output / f"Figure_{200 + number}.{extension}"
            if old.exists():
                old.replace(output / f"Figure_E{number:02}.{extension}")
    receipt_path = output / f"Figure_E{number:02}.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update({"panel_sources": panel_map, "extended_plot_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "figure_width_in": 179 / 25.4, "figure_height_in": 210 / 25.4})
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return output / f"Figure_E{number:02}.png"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, choices=GROUPS)
    parser.add_argument("--output", type=Path, default=ROOT / "output/publication/extended")
    args = parser.parse_args()
    draw(args.number, args.output)
