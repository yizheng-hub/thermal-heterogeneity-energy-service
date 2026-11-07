"""Shared rendering contract for the six frozen main figures.

This module owns layout, typography, colour, export and map-basemap rules. It
does not read scientific panel data or compute scientific results.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.legend import Legend
from PIL import Image


FIGURE_WIDTH_MM = 180
FIGURE_HEIGHT_MM = 230
FIGURE_SIZE_IN = (FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4)
PANEL_ROWS = 4
PANEL_COLUMNS = 2
BASE_FONT_PT = 11
PANEL_TITLE_PT = 12
FONT_FAMILY = "Times New Roman"
PNG_600_SIZE = (4252, 5433)
PNG_WORD_SIZE = (2126, 2717)

PALETTE = {
    "blue": "#6F9EBC",
    "sky": "#ACCAD8",
    "orange": "#C8A36A",
    "vermillion": "#C78470",
    "green": "#7FA995",
    "purple": "#AF94AD",
    "grey": "#A5A5A5",
    "light_grey": "#D2D2D2",
    "dark_grey": "#7F7F7F",
    "uniform": "#7F7F7F",
}

# Shared semantic colours for comparable main and supplementary panels.
CONTEXT_COLORS = {
    "Miami": PALETTE["vermillion"],
    "NewDelhi": PALETTE["purple"],
    "PortAngeles_small2019": PALETTE["blue"],
    "Seattle_small2019": PALETTE["green"],
}
CONTROL_COLORS = {
    "uniform": PALETTE["dark_grey"],
    "personal_center": PALETTE["orange"],
    "ltpref_interval": PALETTE["green"],
}
REPRESENTATION_COLORS = {
    "mean_only": PALETTE["dark_grey"],
    "quantile3": PALETTE["orange"],
    "quantile5": PALETTE["green"],
}
GRIFFITHS_COLORS = {0.33: PALETTE["blue"], 0.50: PALETTE["orange"], 0.67: PALETTE["green"]}
VARIANCE_COLORS = [PALETTE["blue"], PALETTE["sky"], PALETTE["orange"], PALETTE["vermillion"], PALETTE["grey"]]
DEMAND_CMAP = LinearSegmentedColormap.from_list(
    "demand_orange_yellow_green", ["#C98043", "#E5B76A", "#FFF0B4", "#B9D6AA", "#6F9F72"]
)
COUNT_CMAP = LinearSegmentedColormap.from_list(
    "record_or_qualified_count", ["#FFFFFF", "#C7DCE6", PALETTE["blue"]]
)
SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "muted_signed", [PALETTE["blue"], "#F7F6F3", PALETTE["vermillion"]]
)

CATEGORY_STYLES = {
    "cooler": {"color": PALETTE["blue"], "marker": "v"},
    "no_change": {"color": PALETTE["light_grey"], "marker": "o"},
    "warmer": {"color": PALETTE["vermillion"], "marker": "^"},
    "uniform": {"color": CONTROL_COLORS["uniform"], "marker": "o", "linestyle": "-"},
    "personal_centre": {"color": PALETTE["orange"], "marker": "s", "linestyle": "--"},
    "ltpref": {"color": PALETTE["green"], "marker": "^", "linestyle": ":"},
    "full_demand": {"color": PALETTE["blue"], "marker": "o", "linestyle": "-"},
    "mean": {"color": PALETTE["dark_grey"], "marker": "s", "linestyle": "--"},
    "protected_q5": {"color": PALETTE["green"], "marker": "^", "linestyle": ":"},
}

_PANEL_TITLE_GID = "thermal-heterogeneity-panel-title"
_ASSET_PACKAGE = "thermal_heterogeneity.assets"
_ASSET_NAME = "ne_110m_land.geojson"
_PROVENANCE_NAME = "natural_earth_110m_land.provenance.json"
_COLLISION_EDGE_TOLERANCE_PX = 0.5


def apply_data_axis_style(figure: Figure, map_axes: Sequence[Axes] = ()) -> int:
    """Use one L-shaped frame for Cartesian data axes, excluding map/key axes."""
    excluded = set(map_axes)
    checked = 0
    for axis in figure.findobj(Axes):
        if axis in excluded or not axis.axison or axis.get_label() == "<colorbar>":
            continue
        if not (axis.lines or axis.collections or axis.images or axis.patches or axis.containers):
            continue
        for side, spine in axis.spines.items():
            spine.set_visible(side in {"left", "bottom"})
            spine.set_linewidth(0.65)
            spine.set_edgecolor("#8E8E8E")
        axis.tick_params(top=False, right=False, colors="black")
        checked += 1
    return checked


def layout_snapshot(figure: Figure) -> dict[str, dict[str, Any]]:
    """Record data-axis domains separately from their physical page geometry."""
    result = {}

    def record(axis: Axes, key: str) -> None:
        result[key] = {
            "xlim": [float(value) for value in axis.get_xlim()],
            "ylim": [float(value) for value in axis.get_ylim()],
            "xscale": axis.get_xscale(),
            "yscale": axis.get_yscale(),
            "bounds": [float(value) for value in axis.get_position().bounds],
            "children": len(axis.child_axes),
        }
        for index, child in enumerate(axis.child_axes):
            record(child, f"{key}/child{index}")

    for index, axis in enumerate(figure.axes):
        record(axis, f"axis{index}")
    return result


class FigureContractError(RuntimeError):
    """Raised when a rendered figure violates the locked visual contract."""


class FigureFontError(FigureContractError):
    """Raised when exact Times New Roman would be replaced by another font."""


@dataclass(frozen=True)
class ExportBundle:
    """Paths and audit metadata for one complete figure export."""

    pdf: Path
    svg: Path
    png_600: Path
    png_word: Path
    qa: Path
    audit: dict[str, Any]


@dataclass(eq=False, frozen=True)
class _SamePanelOverlapException:
    reason: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_times_new_roman() -> dict[str, str]:
    """Resolve the exact normal and bold fonts, failing instead of falling back."""

    local_fonts = Path(__file__).resolve().parents[2] / "tools/fonts"
    if local_fonts.is_dir():
        from matplotlib import font_manager

        font_manager.fontManager.ttflist = [
            font for font in font_manager.fontManager.ttflist if font.name != FONT_FAMILY
        ]
        for font in sorted(local_fonts.glob("times*.ttf")):
            font_manager.fontManager.addfont(str(font))
        font_manager.fontManager._findfont_cached.cache_clear()
    resolved: dict[str, str] = {}
    for weight in ("normal", "bold"):
        properties = FontProperties(family=[FONT_FAMILY], weight=weight)
        try:
            font_path = Path(findfont(properties, fallback_to_default=False))
        except ValueError as exc:
            raise FigureFontError(
                f"{FONT_FAMILY} ({weight}) is required; install it before rendering"
            ) from exc
        actual_name = FontProperties(fname=str(font_path)).get_name()
        if actual_name != FONT_FAMILY:
            raise FigureFontError(
                f"font fallback prohibited: requested {FONT_FAMILY}, resolved {actual_name}"
            )
        resolved[weight] = str(font_path.resolve())
    return resolved


def apply_figure_style() -> dict[str, str]:
    """Apply the locked global style and return the exact font files in use."""

    fonts = require_times_new_roman()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [FONT_FAMILY],
            "font.size": BASE_FONT_PT,
            "font.weight": "normal",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.titlesize": PANEL_TITLE_PT,
            "axes.titleweight": "bold",
            "axes.labelsize": BASE_FONT_PT,
            "axes.labelweight": "normal",
            "axes.labelpad": -3.0,
            "xtick.labelsize": BASE_FONT_PT,
            "ytick.labelsize": BASE_FONT_PT,
            "xtick.major.pad": 1.0,
            "ytick.major.pad": 1.0,
            "legend.fontsize": BASE_FONT_PT,
            "legend.title_fontsize": BASE_FONT_PT,
            "legend.frameon": False,
            "legend.facecolor": "none",
            "legend.edgecolor": "none",
            "legend.framealpha": 0.0,
            "axes.linewidth": 0.65,
            "axes.edgecolor": "#333333",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.35,
            "grid.alpha": 1.0,
            "lines.linewidth": 1.3,
            "lines.markersize": 5.0,
            "boxplot.medianprops.color": PALETTE["dark_grey"],
            "boxplot.whiskerprops.color": PALETTE["dark_grey"],
            "boxplot.capprops.color": PALETTE["dark_grey"],
            "hatch.color": PALETTE["dark_grey"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "thermal-heterogeneity-figures-v1",
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
        }
    )
    return fonts


def create_panel_figure() -> tuple[Figure, np.ndarray]:
    """Create the locked 180 × 230 mm, two-column by four-row canvas."""

    apply_figure_style()
    figure = Figure(figsize=FIGURE_SIZE_IN, constrained_layout=False)
    FigureCanvasAgg(figure)
    axes = figure.subplots(PANEL_ROWS, PANEL_COLUMNS, squeeze=False)
    axes_width_mm = (FIGURE_WIDTH_MM - 12 - 6 - 10) / PANEL_COLUMNS
    axes_height_mm = (FIGURE_HEIGHT_MM - 6 - 10 - 3 * 12) / PANEL_ROWS
    figure.subplots_adjust(
        left=12 / FIGURE_WIDTH_MM,
        right=1 - 6 / FIGURE_WIDTH_MM,
        top=1 - 6 / FIGURE_HEIGHT_MM,
        bottom=10 / FIGURE_HEIGHT_MM,
        wspace=10 / axes_width_mm,
        hspace=12 / axes_height_mm,
    )
    return figure, axes


def style_panel(
    axis: Axes,
    letter: str,
    title: str,
    *,
    grid_axis: str | None = "y",
) -> mpl.text.Text:
    """Apply shared axes styling and add the complete panel title."""

    if letter not in "abcdefgh":
        raise FigureContractError(f"panel letter must be a-h, received {letter!r}")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_linewidth(0.65)
        axis.spines[side].set_color("#333333")
    axis.tick_params(width=0.65, length=3.0, color="#333333")
    if grid_axis is not None:
        if grid_axis not in {"x", "y", "both"}:
            raise FigureContractError("grid_axis must be x, y, both or None")
        axis.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.35, zorder=0)
    panel_title = axis.set_title(
        f"({letter}) {title}",
        loc="left",
        fontsize=PANEL_TITLE_PT,
        fontweight="bold",
        pad=2,
    )
    panel_title.set_gid(_PANEL_TITLE_GID)
    return panel_title


def data_safe_legend(
    axis: Axes,
    *,
    location: str = "top",
    ncol: int = 2,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
) -> Legend:
    """Place a legend outside the data rectangle and never hide observations."""

    if ncol < 1 or ncol > 5:
        raise FigureContractError("legends may contain at most five columns")
    if location != "top":
        raise FigureContractError("data-safe legends use the reserved top band")
    lower, upper = axis.get_ylim()
    if lower >= upper:
        raise FigureContractError("data-safe legend requires a non-inverted y axis")
    original_upper = upper
    axis.set_ylim(lower, upper + (upper - lower) * 0.30)
    kwargs: dict[str, Any] = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 0.99),
        "ncol": ncol,
        "frameon": False,
        "borderaxespad": 0,
        "handlelength": 1.8,
        "columnspacing": 1.0,
        "handletextpad": 0.45,
    }
    if handles is not None:
        kwargs["handles"] = handles
    if labels is not None:
        kwargs["labels"] = labels
    legend = axis.legend(**kwargs)
    if len(legend.get_texts()) > 5:
        legend.remove()
        axis.set_ylim(lower, original_upper)
        raise FigureContractError("legends may contain at most five entries")
    legend.set_gid("thermal-heterogeneity-data-safe-legend")
    legend._thermal_original_ymax = original_upper
    return legend


def direct_label(
    axis: Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = "#333333",
    dx_points: float = 4,
    dy_points: float = 0,
    **kwargs: Any,
) -> mpl.text.Annotation:
    """Add a compact normal-weight label next to a plotted datum."""

    if kwargs.get("fontweight", "normal") != "normal":
        raise FigureContractError("only panel titles may use bold text")
    return axis.annotate(
        text,
        xy=(x, y),
        xytext=(dx_points, dy_points),
        textcoords="offset points",
        color=color,
        fontsize=BASE_FONT_PT,
        fontweight="normal",
        va="center",
        clip_on=False,
        **kwargs,
    )


def _asset_path(name: str) -> Path:
    return Path(str(files(_ASSET_PACKAGE).joinpath(name)))


def natural_earth_provenance() -> dict[str, Any]:
    """Load and verify the pinned Natural Earth provenance record."""

    provenance = json.loads(_asset_path(_PROVENANCE_NAME).read_text(encoding="utf-8"))
    asset = _asset_path(_ASSET_NAME)
    if asset.stat().st_size != provenance["size_bytes"]:
        raise FigureContractError("Natural Earth asset size differs from provenance")
    if _sha256(asset) != provenance["sha256"]:
        raise FigureContractError("Natural Earth asset checksum differs from provenance")
    return provenance


def load_world_land() -> tuple[np.ndarray, ...]:
    """Return exterior land rings from the pinned Natural Earth GeoJSON."""

    natural_earth_provenance()
    payload = json.loads(_asset_path(_ASSET_NAME).read_text(encoding="utf-8"))
    polygons: list[np.ndarray] = []
    for feature in payload["features"]:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            groups = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            groups = geometry["coordinates"]
        else:
            raise FigureContractError(f"unexpected Natural Earth geometry: {geometry['type']}")
        for rings in groups:
            exterior = np.asarray(rings[0], dtype=float)
            if exterior.ndim != 2 or exterior.shape[1] != 2:
                raise FigureContractError("invalid Natural Earth polygon coordinates")
            polygons.append(exterior)
    if not polygons:
        raise FigureContractError("Natural Earth asset contains no land polygons")
    return tuple(polygons)


def draw_world_land(
    axis: Axes,
    *,
    facecolor: str = "#F2F2F2",
    edgecolor: str = "#8C8C8C",
    linewidth: float = 0.35,
) -> PolyCollection:
    """Draw the pinned public-domain land outline without a GIS dependency."""

    collection = PolyCollection(
        load_world_land(),
        closed=True,
        facecolors=facecolor,
        edgecolors=edgecolor,
        linewidths=linewidth,
        zorder=0,
    )
    axis.add_collection(collection)
    axis.set_xlim(-180, 180)
    axis.set_ylim(-60, 90)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([-120, -60, 0, 60, 120])
    axis.set_yticks([-30, 0, 30, 60])
    axis.set_xlabel("Longitude (°)")
    axis.set_ylabel("Latitude (°)")
    return collection


def _iter_nonempty_text(figure: Figure) -> Iterable[mpl.text.Text]:
    for artist in figure.findobj(mpl.text.Text):
        if artist.get_text().strip():
            yield artist


def allow_same_panel_text_overlap(
    *artists: mpl.text.Text,
    reason: str,
) -> None:
    """Allow one documented overlap group inside a single top-level panel.

    The exception never suppresses overlap or safe-region failures between two
    different top-level panels. It is deliberately attached to exact artist
    objects rather than a global threshold or text pattern.
    """

    if len(artists) != 2:
        raise FigureContractError("a same-panel overlap exception must name exactly two artists")
    if not reason.strip():
        raise FigureContractError("a same-panel overlap exception needs a specific reason")
    exception = _SamePanelOverlapException(reason.strip())
    for artist in artists:
        existing = list(getattr(artist, "_thermal_same_panel_overlap_exceptions", ()))
        existing.append(exception)
        artist._thermal_same_panel_overlap_exceptions = tuple(existing)


def _shared_same_panel_exception(
    first: mpl.text.Text,
    second: mpl.text.Text,
) -> _SamePanelOverlapException | None:
    first_groups = getattr(first, "_thermal_same_panel_overlap_exceptions", ())
    second_groups = getattr(second, "_thermal_same_panel_overlap_exceptions", ())
    for group in first_groups:
        if any(group is candidate for candidate in second_groups):
            return group
    return None


def _display_bbox(values: mpl.transforms.Bbox) -> list[float]:
    return [round(float(value), 3) for value in values.bounds]


def _panel_roots(
    figure: Figure,
    required_letters: str = "abcdefgh",
) -> dict[str, Axes]:
    roots: dict[str, Axes] = {}
    for axis in figure.axes:
        for artist in axis.findobj(mpl.text.Text):
            if artist.get_gid() != _PANEL_TITLE_GID:
                continue
            match = re.match(r"^\(([a-h])\)\s", artist.get_text())
            if not match:
                raise FigureContractError(
                    f"panel title does not begin with (a)-(h): {artist.get_text()!r}"
                )
            letter = match.group(1)
            if letter in roots and roots[letter] is not axis:
                raise FigureContractError(f"multiple top-level axes claim panel {letter}")
            roots[letter] = axis
    if set(roots) != set(required_letters):
        raise FigureContractError(
            "cross-axis audit requires exactly one title-labelled root for panels "
            f"{required_letters}"
        )
    return dict(sorted(roots.items()))


def _all_axes(figure: Figure) -> list[Axes]:
    axes: list[Axes] = []

    def visit(axis: Axes) -> None:
        axes.append(axis)
        for child in axis.child_axes:
            visit(child)

    for axis in figure.axes:
        visit(axis)
    return axes


def _subplot_slot(axis: Axes, figure: Figure) -> mpl.transforms.Bbox:
    try:
        normalized = axis.get_subplotspec().get_position(figure)
    except AttributeError:
        normalized = axis.get_position(original=True)
    return figure.transFigure.transform_bbox(normalized)


def _panel_safe_regions(
    figure: Figure,
    roots: dict[str, Axes],
    panel_letters: str = "abcdefgh",
) -> dict[str, mpl.transforms.Bbox]:
    """Return each panel's allowed text envelope in display coordinates.

    Adjacent panels may both use their shared gutter.  The neighbour's axes
    rectangle is the protected boundary; text-on-text overlaps in the gutter
    are checked separately.  This avoids treating a normal ylabel or xlabel
    as a collision merely because it crosses the geometric gutter midpoint.
    """

    slots = {letter: _subplot_slot(axis, figure) for letter, axis in roots.items()}
    canvas_width, canvas_height = figure.canvas.get_width_height()
    regions: dict[str, mpl.transforms.Bbox] = {}
    panel_rows = int(np.ceil(len(panel_letters) / PANEL_COLUMNS))
    for index, letter in enumerate(panel_letters):
        row, column = divmod(index, PANEL_COLUMNS)
        if column == 0:
            x0 = 0.0
            right_index = index + 1
            x1 = (
                slots[panel_letters[right_index]].x0
                if right_index < len(panel_letters)
                else float(canvas_width)
            )
        else:
            x0 = slots[panel_letters[index - 1]].x1
            x1 = float(canvas_width)
        if row == 0:
            y1 = float(canvas_height)
        else:
            upper = panel_letters[index - PANEL_COLUMNS]
            y1 = slots[upper].y0
        lower_index = index + PANEL_COLUMNS
        if row == panel_rows - 1 or lower_index >= len(panel_letters):
            y0 = 0.0
        else:
            lower = panel_letters[lower_index]
            y0 = slots[lower].y1
        regions[letter] = mpl.transforms.Bbox.from_extents(x0, y0, x1, y1)
    return regions


def _axis_panel_assignments(
    figure: Figure,
    all_axes: Sequence[Axes],
    roots: dict[str, Axes],
    safe_regions: dict[str, mpl.transforms.Bbox],
) -> dict[Axes, str]:
    assignments: dict[Axes, str] = {axis: letter for letter, axis in roots.items()}

    def assign_children(axis: Axes, letter: str) -> None:
        for child in axis.child_axes:
            assignments[child] = letter
            assign_children(child, letter)

    for letter, root in roots.items():
        assign_children(root, letter)

    renderer = figure.canvas.get_renderer()
    for axis in all_axes:
        if axis in assignments:
            continue
        if axis.get_gid() == "thermal-heterogeneity-global-legend-axis":
            assignments[axis] = "__legend__"
            continue
        bounds = axis.get_window_extent(renderer)
        center = ((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2)
        candidates = [
            letter for letter, region in safe_regions.items() if region.contains(*center)
        ]
        if len(candidates) != 1:
            raise FigureContractError(
                f"child axes cannot be assigned to one top-level panel: center={center}"
            )
        assignments[axis] = candidates[0]
    return assignments


def _visible_text_records(
    figure: Figure,
    all_axes: Sequence[Axes],
    axis_panels: dict[Axes, str],
    safe_regions: dict[str, mpl.transforms.Bbox],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    renderer = figure.canvas.get_renderer()
    text_axes: dict[mpl.text.Text, Axes] = {}
    for axis in reversed(all_axes):
        for artist in axis.findobj(mpl.text.Text):
            text_axes.setdefault(artist, axis)
    records: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []
    for sequence, artist in enumerate(_iter_nonempty_text(figure)):
        if not artist.get_visible():
            continue
        if artist.get_alpha() == 0:
            continue
        try:
            if mpl.colors.to_rgba(artist.get_color())[-1] == 0:
                continue
        except (TypeError, ValueError):
            pass
        owner = text_axes.get(artist, artist.axes)
        if owner is not None and not owner.get_visible():
            continue
        if owner is not None and owner.get_gid() == "thermal-heterogeneity-global-legend-axis":
            continue
        bounds = artist.get_window_extent(renderer)
        if bounds.width <= 0 or bounds.height <= 0:
            continue
        panel = axis_panels.get(owner) if owner is not None else None
        if panel is None:
            center = ((bounds.x0 + bounds.x1) / 2, (bounds.y0 + bounds.y1) / 2)
            candidates = [
                letter for letter, region in safe_regions.items() if region.contains(*center)
            ]
            if len(candidates) == 1:
                panel = candidates[0]
            else:
                unassigned.append(
                    {
                        "text": artist.get_text(),
                        "bbox_px": _display_bbox(bounds),
                        "candidate_panels": candidates,
                    }
                )
                continue
        records.append(
            {
                "sequence": sequence,
                "artist": artist,
                "text": artist.get_text(),
                "panel": panel,
                "axis": owner,
                "axis_id": all_axes.index(owner) if owner in all_axes else "figure",
                "bbox": bounds,
                "role": "panel_title" if artist.get_gid() == _PANEL_TITLE_GID else "text",
            }
        )
    return records, unassigned


def _bbox_overlap(
    first: mpl.transforms.Bbox,
    second: mpl.transforms.Bbox,
) -> mpl.transforms.Bbox | None:
    intersection = mpl.transforms.Bbox.intersection(first, second)
    if intersection is None:
        return None
    if (
        intersection.width <= _COLLISION_EDGE_TOLERANCE_PX
        or intersection.height <= _COLLISION_EDGE_TOLERANCE_PX
    ):
        return None
    return intersection


def audit_cross_axis_text_collisions(
    figure: Figure,
    *,
    raise_on_error: bool = True,
    panel_letters: str = "abcdefgh",
) -> dict[str, Any]:
    """Audit rendered text collisions across axes and top-level panel cells.

    Text from child axes is assigned to the enclosing title-labelled panel.
    Cross-axis overlaps inside a panel block unless the exact artists share a
    documented same-panel exception. Cross-panel overlaps and safe-cell
    intrusions always block and cannot be ignored.
    """

    figure.canvas.draw()
    roots = _panel_roots(figure, panel_letters)
    all_axes = _all_axes(figure)
    safe_regions = _panel_safe_regions(figure, roots, panel_letters)
    axis_panels = _axis_panel_assignments(figure, all_axes, roots, safe_regions)
    records, unassigned = _visible_text_records(
        figure, all_axes, axis_panels, safe_regions
    )
    collisions: list[dict[str, Any]] = []
    ignored: list[dict[str, Any]] = []

    pairs_checked = 0
    for index, first in enumerate(records):
        for second in records[index + 1 :]:
            if first["axis"] is second["axis"]:
                continue
            pairs_checked += 1
            intersection = _bbox_overlap(first["bbox"], second["bbox"])
            if intersection is None:
                continue
            exception = _shared_same_panel_exception(first["artist"], second["artist"])
            same_panel = first["panel"] == second["panel"]
            item = {
                "type": "cross_axis_text_overlap",
                "panel_relation": "same_panel" if same_panel else "different_panels",
                "panels": [first["panel"], second["panel"]],
                "axis_ids": [first["axis_id"], second["axis_id"]],
                "texts": [first["text"], second["text"]],
                "bboxes_px": [
                    _display_bbox(first["bbox"]),
                    _display_bbox(second["bbox"]),
                ],
                "intersection_bbox_px": _display_bbox(intersection),
            }
            if same_panel and exception is not None:
                item["reason"] = exception.reason
                ignored.append(item)
            else:
                if not same_panel and exception is not None:
                    item["invalid_ignore_reason"] = exception.reason
                collisions.append(item)

    canvas_width, canvas_height = figure.canvas.get_width_height()
    tolerance = _COLLISION_EDGE_TOLERANCE_PX
    for record in records:
        panel = record["panel"]
        region = safe_regions[panel]
        bounds = record["bbox"]
        directions = []
        if region.x0 > 0 and bounds.x0 < region.x0 - tolerance:
            directions.append("left")
        if region.x1 < canvas_width and bounds.x1 > region.x1 + tolerance:
            directions.append("right")
        if region.y0 > 0 and bounds.y0 < region.y0 - tolerance:
            directions.append("down")
        if region.y1 < canvas_height and bounds.y1 > region.y1 + tolerance:
            directions.append("up")
        if directions:
            collisions.append(
                {
                    "type": "panel_safe_region_intrusion",
                    "panel": panel,
                    "axis_id": record["axis_id"],
                    "text": record["text"],
                    "directions": directions,
                    "bbox_px": _display_bbox(bounds),
                    "safe_region_bbox_px": _display_bbox(region),
                }
            )

    for item in unassigned:
        collisions.append({"type": "unassigned_visible_text", **item})

    report = {
        "status": "PASS" if not collisions else "FAIL",
        "top_level_panels": len(panel_letters),
        "axes_checked": len(all_axes),
        "visible_text_artists": len(records) + len(unassigned),
        "cross_axis_pairs_checked": pairs_checked,
        "blocking_collision_count": len(collisions),
        "ignored_same_panel_overlap_count": len(ignored),
        "blocking_collisions": collisions,
        "ignored_same_panel_overlaps": ignored,
        "panel_safe_regions_px": {
            letter: _display_bbox(region) for letter, region in safe_regions.items()
        },
        "edge_tolerance_px": _COLLISION_EDGE_TOLERANCE_PX,
    }
    if collisions and raise_on_error:
        preview = "; ".join(
            (
                f"{item['type']}:{item.get('panels', item.get('panel'))}:"
                f"{item.get('texts', item.get('text'))}"
            )
            for item in collisions[:5]
        )
        raise FigureContractError(
            f"cross-axis text collision audit failed ({len(collisions)} blocking): {preview}"
        )
    return report


def audit_text_contract(figure: Figure) -> dict[str, Any]:
    """Verify exact font, size and bold-title rules for every text artist."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    counts = {"text_artists": 0, "panel_titles": 0}
    errors: list[str] = []
    for artist in _iter_nonempty_text(figure):
        counts["text_artists"] += 1
        properties = artist.get_fontproperties()
        try:
            resolved = findfont(properties, fallback_to_default=False)
            actual_name = FontProperties(fname=resolved).get_name()
        except ValueError:
            actual_name = "unresolved"
        if actual_name != FONT_FAMILY:
            errors.append(f"font:{artist.get_text()!r}:{actual_name}")
        is_title = artist.get_gid() == _PANEL_TITLE_GID
        expected_size = PANEL_TITLE_PT if is_title else BASE_FONT_PT
        if abs(artist.get_fontsize() - expected_size) > 1e-9:
            errors.append(
                f"size:{artist.get_text()!r}:{artist.get_fontsize()}!={expected_size}"
            )
        weight = str(properties.get_weight()).lower()
        if is_title:
            counts["panel_titles"] += 1
            if weight not in {"bold", "700"}:
                errors.append(f"panel_title_not_bold:{artist.get_text()!r}:{weight}")
        elif weight not in {"normal", "regular", "400"}:
            errors.append(f"non_title_bold:{artist.get_text()!r}:{weight}")
        bounds = artist.get_window_extent(renderer)
        canvas_width, canvas_height = figure.canvas.get_width_height()
        if (
            bounds.x0 < -0.5
            or bounds.y0 < -0.5
            or bounds.x1 > canvas_width + 0.5
            or bounds.y1 > canvas_height + 0.5
        ):
            errors.append(f"clipped_text:{artist.get_text()!r}:{list(bounds.bounds)}")
    if len(figure.axes) != 8:
        errors.append(f"axes:{len(figure.axes)}!=8")
    if counts["panel_titles"] != 8:
        errors.append(f"panel_titles:{counts['panel_titles']}!=8")
    width, height = figure.get_size_inches()
    if abs(width - FIGURE_SIZE_IN[0]) > 1e-9 or abs(height - FIGURE_SIZE_IN[1]) > 1e-9:
        errors.append(f"canvas_inches:{width}x{height}")
    if errors:
        raise FigureContractError("text/layout contract failed: " + "; ".join(errors))
    return counts


def audit_data_safe_legends(figure: Figure) -> dict[str, int]:
    """Verify all legends are transparent and reserved-band legends avoid data."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legends_checked = 0
    data_safe_checked = 0
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend is None:
            continue
        legends_checked += 1
        if legend.get_frame().get_visible():
            raise FigureContractError("legend frame/background must be disabled")
        if legend.get_gid() != "thermal-heterogeneity-data-safe-legend":
            continue
        data_safe_checked += 1
        original_ymax = getattr(legend, "_thermal_original_ymax", None)
        if original_ymax is None:
            raise FigureContractError("data-safe legend lacks its reserved-band boundary")
        data_ceiling = axis.transData.transform((0, original_ymax))[1]
        if legend.get_window_extent(renderer).y0 < data_ceiling - 0.5:
            raise FigureContractError("data-safe legend extends below its reserved top band")
    for legend in figure.findobj(Legend):
        if legend.axes is not None:
            continue
        legends_checked += 1
        if legend.get_frame().get_visible():
            raise FigureContractError("legend frame/background must be disabled")
    return {
        "legends_checked": legends_checked,
        "data_safe_legends_checked": data_safe_checked,
    }


def _save_atomic(figure: Figure, target: Path, **kwargs: Any) -> None:
    temporary = target.with_name(target.name + ".tmp" + target.suffix)
    figure.savefig(temporary, **kwargs)
    temporary.replace(target)


def _save_raster(figure: Figure, target: Path, dpi: int, pixels: tuple[int, int]) -> None:
    original_size = tuple(figure.get_size_inches())
    figure.set_size_inches(pixels[0] / dpi, pixels[1] / dpi, forward=True)
    rendered = target.with_name(target.name + ".rendered.png")
    converted = target.with_name(target.name + ".rgb.png")
    try:
        figure.savefig(rendered, dpi=dpi, format="png")
        with Image.open(rendered) as image:
            image.convert("RGB").save(
                converted,
                format="PNG",
                dpi=(dpi, dpi),
                compress_level=6,
            )
        converted.replace(target)
    finally:
        figure.set_size_inches(original_size, forward=True)
        rendered.unlink(missing_ok=True)
        converted.unlink(missing_ok=True)


def _audit_pdf(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    errors = []
    for forbidden in (b"DejaVu", b"/Type3"):
        if forbidden in payload:
            errors.append(f"forbidden PDF token {forbidden.decode('ascii')}")
    if b"TimesNewRoman" not in payload or b"/FontFile2" not in payload:
        errors.append("embedded Times New Roman TrueType font not found")
    media = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]", payload)
    if not media:
        errors.append("PDF MediaBox not found")
        size_points = None
    else:
        size_points = [float(media.group(1)), float(media.group(2))]
        expected = [value / 25.4 * 72 for value in (FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM)]
        if any(abs(observed - target) > 0.02 for observed, target in zip(size_points, expected)):
            errors.append(f"PDF MediaBox {size_points} differs from {expected}")
    pdffonts_rows: list[str] = []
    executable = shutil.which("pdffonts")
    if executable:
        completed = subprocess.run(
            [executable, str(path)], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            errors.append(f"pdffonts failed: {completed.stderr.strip()}")
        else:
            pdffonts_rows = completed.stdout.splitlines()[2:]
            for row in pdffonts_rows:
                if row.strip() and ("TimesNewRoman" not in row or " yes " not in f" {row} "):
                    errors.append(f"unexpected or unembedded PDF font: {row.strip()}")
    if errors:
        raise FigureContractError("PDF audit failed: " + "; ".join(errors))
    return {"media_box_points": size_points, "pdffonts": pdffonts_rows}


def _audit_svg(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    errors = []
    if "DejaVu" in text:
        errors.append("DejaVu fallback found")
    text_elements = len(re.findall(r"<text\b", text))
    if text_elements == 0:
        errors.append("no live SVG text elements")
    if FONT_FAMILY not in text:
        errors.append("Times New Roman family missing")
    width = re.search(r'<svg[^>]*width="([0-9.]+)pt"', text)
    height = re.search(r'<svg[^>]*height="([0-9.]+)pt"', text)
    expected = [value / 25.4 * 72 for value in (FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM)]
    dimensions = None
    if width and height:
        dimensions = [float(width.group(1)), float(height.group(1))]
        if any(abs(observed - target) > 0.02 for observed, target in zip(dimensions, expected)):
            errors.append(f"SVG canvas {dimensions} differs from {expected}")
    else:
        errors.append("SVG point dimensions missing")
    if errors:
        raise FigureContractError("SVG audit failed: " + "; ".join(errors))
    return {"canvas_points": dimensions, "live_text_elements": text_elements}


def _audit_png(path: Path, expected_size: tuple[int, int]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.load()
        size = image.size
        mode = image.mode
        corner = image.convert("RGB").getpixel((0, 0))
        dpi = image.info.get("dpi")
    errors = []
    if size != expected_size:
        errors.append(f"pixels:{size}!={expected_size}")
    if mode != "RGB":
        errors.append(f"mode:{mode}")
    if corner != (255, 255, 255):
        errors.append(f"background_corner:{corner}")
    if errors:
        raise FigureContractError("PNG audit failed: " + "; ".join(errors))
    return {"pixels": list(size), "mode": mode, "corner_rgb": list(corner), "dpi": dpi}


def save_figure_bundle(figure: Figure, output_directory: Path | str, stem: str) -> ExportBundle:
    """Export and audit PDF, SVG, 600-dpi PNG and 300-dpi Word PNG."""

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stem):
        raise FigureContractError("figure stem contains unsupported characters")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    fonts = require_times_new_roman()
    text_audit = audit_text_contract(figure)
    collision_audit = audit_cross_axis_text_collisions(figure)
    legend_audit = audit_data_safe_legends(figure)
    pdf = output / f"{stem}.pdf"
    svg = output / f"{stem}.svg"
    png_600 = output / f"{stem}_600dpi.png"
    png_word = output / f"{stem}_word.png"
    qa = output / f"{stem}_qa.json"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _save_atomic(
            figure,
            pdf,
            format="pdf",
            metadata={
                "Title": "Human thermal heterogeneity figure",
                "Author": "Study authors",
                "Subject": "Reproducible figure export",
                "Creator": "thermal_heterogeneity.figures",
                "Producer": "Matplotlib",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        _save_atomic(
            figure,
            svg,
            format="svg",
            metadata={"Title": "Human thermal heterogeneity figure", "Date": None},
        )
        _save_raster(figure, png_600, 600, PNG_600_SIZE)
        _save_raster(figure, png_word, 300, PNG_WORD_SIZE)
    fallback_warnings = [
        str(item.message)
        for item in caught
        if "font" in str(item.message).lower() or "glyph" in str(item.message).lower()
    ]
    if fallback_warnings:
        raise FigureFontError("font/glyph warning during export: " + "; ".join(fallback_warnings))
    audit = {
        "status": "PASS",
        "contract": {
            "canvas_mm": [FIGURE_WIDTH_MM, FIGURE_HEIGHT_MM],
            "grid": [PANEL_ROWS, PANEL_COLUMNS],
            "base_font_pt": BASE_FONT_PT,
            "panel_title_pt": PANEL_TITLE_PT,
            "font_family": FONT_FAMILY,
        },
        "fonts": fonts,
        "text": text_audit,
        "cross_axis_text": collision_audit,
        "legends": legend_audit,
        "pdf": _audit_pdf(pdf),
        "svg": _audit_svg(svg),
        "png_600": _audit_png(png_600, PNG_600_SIZE),
        "png_word": _audit_png(png_word, PNG_WORD_SIZE),
        "files": {},
        "natural_earth": natural_earth_provenance(),
    }
    for path in (pdf, svg, png_600, png_word):
        audit["files"][path.name] = {
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    qa.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ExportBundle(pdf, svg, png_600, png_word, qa, audit)


def clarify_control_caption(text, *, uses_three_controls=False):
    """Keep each control figure understandable without the main text."""
    text = text.replace("P10–P90/P25–P75", "10th–90th and 25th–75th percentile ranges")
    text = text.replace("P10–P90", "10th–90th percentiles").replace("P25–P75", "25th–75th percentiles")
    text = text.replace("percentiles ranges", "percentile ranges").replace("percentiles intervals", "percentile intervals")
    text = text.replace("LTPref interval", "Personal range").replace("LTPref control", "Personal range control")
    text = text.replace("LTPref conditions", "personal range conditions")
    text = text.replace("native-zone", "zone-level").replace("building-bound", "building-wide")
    text = text.replace("admitted", "engineering-checked").replace("Admitted", "Engineering-checked")
    if uses_three_controls or "Personal range" in text or "personal range" in text:
        text = text.rstrip() + (
            " Personal ranges were estimated from recorded environmental conditions and "
            "thermal responses using the published elicitation method. "
            "All controls choose shared targets within the permitted spaces."
        )
    if uses_three_controls and re.search(r"fixed[- ](?:annual[- ])?tail", text) and "worst served" not in text:
        text += " The fixed tail tracks the profiles worst served by common 25 °C operation."
    definitions = {
        "HVAC": "HVAC denotes heating, ventilation and air conditioning.",
        "HHV": "HHV denotes higher heating value.",
        "INR": "INR denotes Indian rupees.",
    }
    for short, definition in definitions.items():
        meaning = definition.split(" denotes ", 1)[1].rstrip(".")
        if re.search(r"\b" + short + r"\b", text) and meaning.casefold() not in text.casefold():
            text += " " + definition
    return text
