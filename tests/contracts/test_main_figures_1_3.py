"""Contracts for the retained legacy eight-panel renderers, not Plot.ipynb."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from thermal_heterogeneity.figures import (
    audit_cross_axis_text_collisions,
    audit_text_contract,
    require_times_new_roman,
)
from thermal_heterogeneity.main_figures_1_3 import (
    ALT_TEXT,
    CAPTIONS,
    FIGURE_BUILDERS,
    close_figure,
)


def _panel_root() -> Path | None:
    explicit = os.environ.get("THERMAL_HETEROGENEITY_PANEL_DATA")
    if explicit:
        return Path(explicit)
    repository = Path(__file__).resolve().parents[2]
    candidate = repository / "results" / "panel_data"
    if candidate.is_dir():
        return candidate
    return None


@pytest.mark.parametrize("number", [1, 2, 3])
def test_main_figure_has_exact_eight_panel_contract(number):
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    try:
        require_times_new_roman()
    except Exception:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = FIGURE_BUILDERS[number](root)
    try:
        audit = audit_text_contract(figure)
        assert len(figure.axes) == 8
        assert audit["panel_titles"] == 8
    finally:
        close_figure(figure)


@pytest.mark.parametrize("number", [1, 2, 3])
def test_main_figure_has_no_cross_panel_text_collision(number):
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[number](root)
    try:
        report = audit_cross_axis_text_collisions(figure, raise_on_error=False)
        assert report["blocking_collision_count"] == 0, report["blocking_collisions"]
    finally:
        close_figure(figure)


def _artists_with_gid(figure, gid):
    return [artist for artist in figure.findobj() if artist.get_gid() == gid]


def _overlap(first, second, tolerance=0.5):
    from matplotlib.transforms import Bbox

    intersection = Bbox.intersection(first, second)
    return (
        intersection is not None
        and intersection.width > tolerance
        and intersection.height > tolerance
    )


@pytest.mark.parametrize(
    ("number", "gid"),
    [
        (1, "fig1g-value-label"),
        (2, "fig2b-summary"),
        (2, "fig2g-end-label"),
        (3, "fig3d-n-table"),
        (3, "fig3e-n-table"),
        (3, "fig3f-match-note"),
    ],
)
def test_key_annotations_stay_inside_their_own_axes(number, gid):
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[number](root)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        artists = _artists_with_gid(figure, gid)
        assert artists, gid
        for artist in artists:
            bounds = artist.get_window_extent(renderer)
            axis_bounds = artist.axes.get_window_extent(renderer)
            assert bounds.x0 >= axis_bounds.x0 - 0.5
            assert bounds.y0 >= axis_bounds.y0 - 0.5
            assert bounds.x1 <= axis_bounds.x1 + 0.5
            assert bounds.y1 <= axis_bounds.y1 + 0.5
    finally:
        close_figure(figure)


@pytest.mark.parametrize(
    ("number", "gid"),
    [(2, "fig2g-end-label"), (3, "fig3d-n-table"), (3, "fig3e-n-table")],
)
def test_aligned_annotation_groups_do_not_overlap(number, gid):
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[number](root)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        artists = _artists_with_gid(figure, gid)
        bounds = [artist.get_window_extent(renderer) for artist in artists]
        for index, first in enumerate(bounds):
            for second in bounds[index + 1 :]:
                assert not _overlap(first, second), gid
    finally:
        close_figure(figure)


def test_fig2b_summary_does_not_touch_title_or_legend():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[2](root)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        summary = _artists_with_gid(figure, "fig2b-summary")[0].get_window_extent(renderer)
        title = figure.axes[1]._left_title.get_window_extent(renderer)
        legend = figure.axes[1].get_legend().get_window_extent(renderer)
        assert not _overlap(summary, title)
        assert not _overlap(summary, legend)
    finally:
        close_figure(figure)


def test_fig2b_legend_does_not_occlude_any_window_bar_patch():
    from matplotlib.transforms import Bbox

    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[2](root)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bounds = figure.axes[1].get_legend().get_window_extent(renderer)
        bar_patches = _artists_with_gid(figure, "fig2b-window-bar")
        assert len(bar_patches) == 63 * 3
        overlaps = []
        for index, patch in enumerate(bar_patches):
            intersection = Bbox.intersection(
                legend_bounds,
                patch.get_window_extent(renderer),
            )
            if intersection is not None and intersection.width > 0 and intersection.height > 0:
                overlaps.append(index)
        assert overlaps == []
    finally:
        close_figure(figure)


def test_figure_2_uses_room_window_not_instantaneous_opposition_language():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    for text in (CAPTIONS[2], ALT_TEXT[2]):
        assert "simultaneous" not in text.lower()
        assert "within the same 30-min room-window" in text
        assert "within-window opposing requests" in text

    figure = FIGURE_BUILDERS[2](root)
    try:
        visible = "\n".join(
            artist.get_text() for artist in figure.findobj() if hasattr(artist, "get_text")
        )
        assert "Within-window opposing requests" in visible
        assert "simultaneous" not in visible.lower()
    finally:
        close_figure(figure)


def test_figure_3_separates_controls_from_qualification_representations():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    for text in (CAPTIONS[3], ALT_TEXT[3]):
        lower = text.lower()
        assert "uniform average" in lower
        assert "personal centre" in lower
        assert "personal range" in lower
        assert "control families" in lower
        assert "mean and full-distribution" in lower
        assert "qualification representations" in lower
        assert "demand representations" not in lower
        assert "frozen" not in lower

    figure = FIGURE_BUILDERS[3](root)
    try:
        visible = "\n".join(
            artist.get_text() for artist in figure.findobj() if hasattr(artist, "get_text")
        )
        assert "Three control families" in visible
        assert "Same building and operating permissions" in visible
        assert "Three demand representations" not in visible
        assert "Same frozen building" not in visible
        assert "frozen" not in visible.lower()
    finally:
        close_figure(figure)


def test_fig3f_match_note_does_not_touch_bar_counts_or_legend():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    figure = FIGURE_BUILDERS[3](root)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        note = _artists_with_gid(figure, "fig3f-match-note")[0].get_window_extent(renderer)
        neighbours = _artists_with_gid(figure, "fig3f-bar-n")
        neighbours.append(figure.axes[5].get_legend())
        for artist in neighbours:
            assert not _overlap(note, artist.get_window_extent(renderer))
    finally:
        close_figure(figure)


def test_fig3_unmatched_carrier_rows_are_not_plotted_as_signed_changes():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    import pandas as pd

    frame = pd.read_csv(root / "Fig3f.csv")
    signed = [column for column in frame if column.endswith("_change_mwh_signed")]
    unmatched = frame.loc[~frame["matched"].astype(str).str.lower().eq("true")]
    assert unmatched[signed].isna().all().all()
    assert len(frame) == 144
    assert len(frame) - len(unmatched) == 59


def test_fig1_frozen_denominators_and_observed_support():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    import pandas as pd

    archive = pd.read_csv(root / "Fig1a.csv").set_index("population")
    assert int(archive.loc["complete", "records"]) == 21_418
    assert int(archive.loc["executable", "records"]) == 21_399
    assert int(archive.loc["complete", "analysis_units"]) == 12_130
    assert int(archive.loc["executable", "analysis_units"]) == 12_113
    assert pd.read_csv(root / "Fig1b.csv")["supported"].sum() == 31
    assert pd.read_csv(root / "Fig1c.csv")["supported"].sum() == 26
    assert pd.read_csv(root / "Fig1e.csv")["share"].sum() == pytest.approx(1.0)


def test_fig1g_direction_is_demand_centre_minus_campaign_target():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    import pandas as pd

    panel_path = root / "Fig1g.csv"
    assert hashlib.sha256(panel_path.read_bytes()).hexdigest() == (
        "4cf6dd618fad79495e5bdbe521671f4e8b23eeacfd9a7118887746423eb15b96"
    )
    medians = pd.read_csv(panel_path).set_index("policy")["median_delta_c"]
    assert medians.loc["China summer rule"] == pytest.approx(0.4)
    assert medians.loc["Japan Cool Biz"] == pytest.approx(-2.05)
    assert medians.loc["Thailand EGAT campaign"] == pytest.approx(-2.32056033698)

    figure = FIGURE_BUILDERS[1](root)
    try:
        label = figure.axes[6].get_xlabel()
        assert label == "Demand centre − campaign target (°C)"
    finally:
        close_figure(figure)

    caption = CAPTIONS[1].lower()
    alt_text = ALT_TEXT[1].lower()
    for text in (caption, alt_text):
        assert "demand centre minus campaign target" in text
        assert "positive values" in text and "warmer" in text
        assert "negative values" in text and "cooler" in text
        assert "campaign target minus demand centre" not in text


def test_fig2_window_denominators_and_external_replication():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    import pandas as pd

    windows = pd.read_csv(root / "Fig2b.csv")
    present = (windows[["n_cooler", "n_no_change", "n_warmer"]] > 0).sum(axis=1)
    assert len(windows) == 63
    assert windows["n_votes"].sum() == 225
    assert (present > 1).sum() == 49
    assert ((windows["n_cooler"] > 0) & (windows["n_warmer"] > 0)).sum() == 6
    assert len(pd.read_csv(root / "Fig2e.csv")) == 4
    thresholds = pd.read_csv(root / "Fig2f.csv")
    assert thresholds["false_accept_count"].tolist() == [4, 4, 5, 2]
    assert thresholds["false_reject_count"].sum() == 0
    replication = pd.read_csv(root / "Fig2h.csv").drop_duplicates("dataset").set_index("dataset")
    assert int(replication.loc["NUS_HAAS", "disagreement_count"]) == 49
    assert int(replication.loc["ENTH", "disagreement_count"]) == 22


def test_fig3_direction_counts_and_context_medians():
    root = _panel_root()
    if root is None:
        pytest.skip("panel-data v2 is materialized by the reproducibility workflow")
    import pandas as pd

    pairs = pd.read_csv(root / "Fig3g.csv")
    assert pairs["energy_direction"].value_counts().to_dict() == {
        "same_direction_increase": 54,
        "all_tied": 36,
        "match_lost": 30,
        "direction_not_stable": 7,
        "same_direction_decrease": 3,
    }
    energy = pd.read_csv(root / "Fig3d.csv")
    matched = energy.loc[energy["matched"] & energy["delta_energy_mwh"].notna()]
    medians = matched.loc[matched["delta_c"] == 2].groupby("context")["delta_energy_mwh"].median()
    assert medians.loc["Miami"] == pytest.approx(-5.430748, abs=1e-6)
    assert medians.loc["NewDelhi"] == pytest.approx(-12.714606, abs=1e-6)
    assert medians.loc["PortAngeles_small2019"] == pytest.approx(0.832167, abs=1e-6)
    assert medians.loc["Seattle_small2019"] == pytest.approx(0.389189, abs=1e-6)
