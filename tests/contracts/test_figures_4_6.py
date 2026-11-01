"""Contracts for the retained legacy eight-panel renderers, not Plot.ipynb."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from thermal_heterogeneity import figures
from thermal_heterogeneity.figures_4_6 import (
    ALT_TEXT,
    BUILDERS,
    CAPTIONS,
    verify_panel_inputs,
)


def _panel_data() -> Path:
    explicit = os.environ.get("THERMAL_HETEROGENEITY_PANEL_DATA")
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_dir():
            pytest.fail("Explicit panel-data directory is missing")
        return candidate
    candidate = Path(__file__).resolve().parents[2] / "results" / "panel_data"
    if candidate.is_dir():
        return candidate
    pytest.skip("retained legacy eight-panel inputs are not included in this checkout")


@pytest.mark.parametrize("number", [4, 5, 6])
def test_frozen_panel_inputs_and_denominators(number):
    hashes = verify_panel_inputs(_panel_data(), number)
    assert sorted(hashes) == [f"Fig{number}{letter}" for letter in "abcdefgh"]
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize("number", [4, 5, 6])
def test_each_builder_has_exactly_eight_contract_panels(number):
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[number](_panel_data())
    try:
        audit = figures.audit_text_contract(figure)
        assert len(figure.axes) == 8
        assert audit["panel_titles"] == 8
    finally:
        plt.close(figure)


def test_captions_cover_every_panel_and_key_boundaries():
    for number, caption in CAPTIONS.items():
        for letter in "abcdefgh":
            assert f"{letter}," in caption
    assert "same pre-specified action" in CAPTIONS[5]
    assert "0/144" in CAPTIONS[5]
    assert "210/2,880" in CAPTIONS[5]
    assert "14,709/18,766" in CAPTIONS[6]
    assert "542/606" in CAPTIONS[6]


def test_figure_6_caption_locks_corrected_semantics():
    caption = CAPTIONS[6]
    assert "Schematic" in caption
    assert "service-budget multiplier" in caption
    assert "looser service-error allowances" in caption
    assert "do not define an empirical monotonic boundary" in caption
    assert "in-sample scanned diagnostic threshold" in caption
    assert "did not meet the prespecified" in caption
    assert "points and error bars show updated scores" in caption
    assert "paired improvement analysis" in caption
    assert "city-held-out improvement intervals are above zero" in caption
    assert "source-held-out improvement intervals cross zero" in caption
    assert "Anonymous-summary workflow" in caption
    assert "not a formal privacy guarantee" in caption
    assert "locked 1.059" not in caption
    assert "Privacy-preserving" not in caption


def test_alt_text_covers_comparison_direction_denominators_and_limits():
    assert set(ALT_TEXT) == {4, 5, 6}
    for number, text in ALT_TEXT.items():
        assert len(text) >= 300
        assert "/" in text
        assert "panel" in text.lower()
    assert "169/263" in ALT_TEXT[4] and "1,143 unresolved" in ALT_TEXT[4]
    assert "same pre-specified action" in ALT_TEXT[5] and "0/144" in ALT_TEXT[5]
    assert "14,709/18,766" in ALT_TEXT[6] and "not a universal guarantee" in ALT_TEXT[6]
    for letter in "abcdefgh":
        assert f"Panel {letter}" in ALT_TEXT[6]
    assert "local-feedback learning curves" in ALT_TEXT[6]
    assert "qualified-reporting workflow" in ALT_TEXT[6]


def test_figures_4_to_6_submission_text_uses_scientific_boundary_terms():
    for number in (4, 5, 6):
        for text in (CAPTIONS[number], ALT_TEXT[number]):
            assert "frozen" not in text.lower()
    assert "same fixed action" in CAPTIONS[4]
    assert "fixed annual action library" in ALT_TEXT[4]
    assert "Pre-specified 2-h event" in CAPTIONS[5]
    assert "fixed mean- and tail-service budgets" in CAPTIONS[5]
    assert "hierarchical-bootstrap 95% intervals" in CAPTIONS[6]
    assert "hierarchical-bootstrap 95% intervals" in ALT_TEXT[6]


@pytest.mark.parametrize("number", [4, 5, 6])
def test_figures_4_to_6_visible_artwork_has_no_internal_frozen_wording(number):
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[number](_panel_data())
    try:
        visible = "\n".join(text.get_text() for text in figure.findobj(Text))
        assert "frozen" not in visible.lower()
        if number == 4:
            assert "Fixed action" in visible
        if number == 5:
            assert "Fixed-action power trajectory" in visible
    finally:
        plt.close(figure)


@pytest.mark.parametrize("number", [4, 6])
def test_submission_facing_figure_text_has_no_gate_wording(number):
    for text in (CAPTIONS[number], ALT_TEXT[number]):
        assert re.search(r"\bgates?\b", text, flags=re.IGNORECASE) is None

    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[number](_panel_data())
    try:
        visible = "\n".join(text.get_text() for text in figure.findobj(Text))
        assert re.search(r"\bgates?\b", visible, flags=re.IGNORECASE) is None
    finally:
        plt.close(figure)


def test_figure_6_alt_text_locks_corrected_semantics():
    alt_text = ALT_TEXT[6]
    assert "Panel a is a schematic" in alt_text
    assert "service-budget multiplier" in alt_text
    assert "not an empirical monotonic boundary" in alt_text
    assert "in-sample scanned diagnostic threshold" in alt_text
    assert "does not meet the prespecified performance standards" in alt_text
    assert "city-held-out observations" in alt_text
    assert "points and error bars are updated scores" in alt_text
    assert "paired improvement analysis" in alt_text
    assert "improvement intervals are above zero" in alt_text
    assert (
        "source-held-out observations both corresponding improvement intervals cross zero"
        in alt_text
    )
    assert "including the tail-share error interval" in alt_text
    assert "dimensionless normalized ordinal RPS and tail-share MAE" in alt_text
    assert "anonymous-summary qualified-reporting workflow" in alt_text
    assert "no formal privacy guarantee" in alt_text
    assert "locked 1.059" not in alt_text
    assert "required service increase" not in alt_text
    assert "privacy-preserving" not in alt_text.lower()


def test_figure_6g_panel_data_preserves_displayed_learning_curve_levels():
    learning = pd.read_csv(_panel_data() / "Fig6g.csv")
    assert set(learning["transfer_split"]) == {"city", "source"}
    assert set(learning["metric"]) == {"CRPS", "tail_MAE"}
    assert set(learning["feedback_n"].astype(int)) == {0, 5, 10, 20, 40}
    for split in ("city", "source"):
        for metric in ("CRPS", "tail_MAE"):
            rows = learning[
                (learning["transfer_split"] == split) & (learning["metric"] == metric)
            ].set_index("feedback_n")
            assert rows.loc[40, "estimate"] < rows.loc[0, "estimate"]


def test_figure_6_display_labels_do_not_overstate_schematic_or_threshold():
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[6](_panel_data())
    try:
        visible = "\n".join(text.get_text() for text in figure.findobj(Text))
        assert "Service-budget multiplier" in visible
        assert "looser allowance" in visible
        assert "in-sample scan 1.059 °C" in visible
        assert "Required service fraction" not in visible
        assert "rule 1.059 °C" not in visible
    finally:
        plt.close(figure)


@pytest.mark.parametrize("number", [4, 5, 6])
def test_layout_critical_text_stays_in_its_panel_without_collisions(number):
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[number](_panel_data())
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        critical = [
            artist
            for artist in figure.findobj(Text)
            if (artist.get_gid() or "").startswith("figures-4-6-layout-critical:")
        ]
        assert critical
        for artist in critical:
            bbox = artist.get_window_extent(renderer)
            owner = artist.axes.get_window_extent(renderer)
            assert bbox.x0 >= owner.x0 - 1
            assert bbox.y0 >= owner.y0 - 1
            assert bbox.x1 <= owner.x1 + 1
            assert bbox.y1 <= owner.y1 + 1
            owner_panel = artist.axes
            if owner_panel not in figure.axes:
                owner_panel = next(
                    panel for panel in figure.axes if artist.axes in panel.child_axes
                )
            for other_axis in figure.axes:
                if other_axis is owner_panel:
                    continue
                assert not bbox.overlaps(other_axis.get_window_extent(renderer))
        for index, artist in enumerate(critical):
            first = artist.get_window_extent(renderer)
            for other in critical[index + 1 :]:
                if artist.axes is other.axes:
                    assert not first.overlaps(other.get_window_extent(renderer))
    finally:
        plt.close(figure)


@pytest.mark.parametrize("number", [4, 5, 6])
def test_panel_titles_fit_within_their_own_columns(number):
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[number](_panel_data())
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        titles = [
            artist
            for artist in figure.findobj(Text)
            if artist.get_gid() == "thermal-heterogeneity-panel-title"
        ]
        assert len(titles) == 8
        for title in titles:
            bbox = title.get_window_extent(renderer)
            owner = title.axes.get_window_extent(renderer)
            assert bbox.x0 >= owner.x0 - 1
            assert bbox.x1 <= owner.x1 + 1
    finally:
        plt.close(figure)


def test_case_lock_prohibits_two_action_claim():
    panel_data = _panel_data()
    case = json.loads((panel_data / "Fig5bc_case_lock.json").read_text(encoding="utf-8"))
    assert case["mean_full_same_action"] is True
    assert case["mean_selected_action_id"] == case["full_selected_action_id"]
    assert "does not compare mean- and full-selected actions" in CAPTIONS[5]


def test_figure_4f_denominator_annotations_do_not_overlap_data_or_legend():
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[4](_panel_data())
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel = figure.axes[5]
        denominator_text = [
            artist
            for artist in panel.findobj(Text)
            if "figures-4-6-layout-critical:4f-denominator" in (artist.get_gid() or "")
        ]
        assert len(denominator_text) == 5
        divider_x = panel.transData.transform((50, 0))[0]
        denominator_boxes = [artist.get_window_extent(renderer) for artist in denominator_text]
        assert all(box.x0 > divider_x + 1 for box in denominator_boxes)

        marker_boxes = []
        for marker in panel.findobj(PathCollection):
            if not (marker.get_gid() or "").startswith("figure-4f-data-marker:"):
                continue
            diameter = np.sqrt(float(marker.get_sizes()[0])) * figure.dpi / 72 + 2
            for x, y in marker.get_offset_transform().transform(marker.get_offsets()):
                marker_boxes.append(
                    Bbox.from_bounds(x - diameter / 2, y - diameter / 2, diameter, diameter)
                )
        interval_boxes = [
            artist.get_window_extent(renderer).padded(2)
            for artist in panel.findobj(Line2D)
            if (artist.get_gid() or "").startswith("figure-4f-data-interval:")
        ]
        assert marker_boxes and interval_boxes
        legend = next(
            artist
            for artist in panel.findobj(Legend)
            if artist.get_gid() == "figure-4f-legend"
        )
        legend_box = legend.get_window_extent(renderer)
        for annotation_box in denominator_boxes:
            assert not annotation_box.overlaps(legend_box)
            assert all(not annotation_box.overlaps(box) for box in marker_boxes)
            assert all(not annotation_box.overlaps(box) for box in interval_boxes)
    finally:
        plt.close(figure)


def test_figure_5d_last_annotation_clears_marker_legend_and_lambda_label():
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure = BUILDERS[5](_panel_data())
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        panel = figure.axes[3]
        labels = [
            artist
            for artist in panel.findobj(Text)
            if (artist.get_gid() or "").startswith("figures-4-6-layout-critical:5d-")
        ]
        target = next(
            artist
            for artist in labels
            if artist.get_gid() == "figures-4-6-layout-critical:5d-full-1.5"
        )
        target_box = target.get_window_extent(renderer)
        marker_x, marker_y = panel.transData.transform((1.5, 12))
        marker_box = Bbox.from_bounds(marker_x - 5, marker_y - 5, 10, 10)
        legend = next(
            artist
            for artist in panel.findobj(Legend)
            if artist.get_gid() == "figure-5d-legend"
        )
        legend_box = legend.get_window_extent(renderer)
        assert not target_box.overlaps(marker_box)
        assert all(not label.get_window_extent(renderer).overlaps(legend_box) for label in labels)
        assert not target_box.overlaps(panel.xaxis.label.get_window_extent(renderer))
        assert all(
            not target_box.overlaps(tick.get_window_extent(renderer))
            for tick in panel.get_xticklabels()
        )
    finally:
        plt.close(figure)
