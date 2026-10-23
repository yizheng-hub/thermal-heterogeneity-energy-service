from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from thermal_heterogeneity import figures


def _geometry_only_panel_figure():
    figure = Figure(figsize=figures.FIGURE_SIZE_IN, constrained_layout=False)
    FigureCanvasAgg(figure)
    axes = figure.subplots(4, 2, squeeze=False)
    axes_width_mm = (figures.FIGURE_WIDTH_MM - 12 - 6 - 10) / 2
    axes_height_mm = (figures.FIGURE_HEIGHT_MM - 6 - 10 - 3 * 12) / 4
    figure.subplots_adjust(
        left=12 / figures.FIGURE_WIDTH_MM,
        right=1 - 6 / figures.FIGURE_WIDTH_MM,
        top=1 - 6 / figures.FIGURE_HEIGHT_MM,
        bottom=10 / figures.FIGURE_HEIGHT_MM,
        wspace=10 / axes_width_mm,
        hspace=12 / axes_height_mm,
    )
    for letter, axis in zip("abcdefgh", axes.ravel()):
        axis.set_xticks([])
        axis.set_yticks([])
        figures.style_panel(axis, letter, "Collision contract", grid_axis=None)
    return figure, axes


def test_natural_earth_asset_matches_pinned_provenance():
    provenance = figures.natural_earth_provenance()
    assert provenance["licence"] == "Public domain"
    assert provenance["repository_commit"] == "ca96624a56bd078437bca8184e78163e5039ad19"
    assert len(figures.load_world_land()) > 100


def test_font_resolution_fails_closed(monkeypatch):
    def missing_font(*args, **kwargs):
        raise ValueError("font missing")

    monkeypatch.setattr(figures, "findfont", missing_font)
    with pytest.raises(figures.FigureFontError, match="required"):
        figures.require_times_new_roman()


def test_locked_canvas_titles_and_data_safe_legend():
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure, axes = figures.create_panel_figure()
    try:
        for letter, axis in zip("abcdefgh", axes.ravel()):
            axis.plot([0, 1], [0, 1], label="Series")
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            figures.style_panel(axis, letter, "Contract test")
        figures.data_safe_legend(axes[0, 0], location="top")
        assert figures.audit_text_contract(figure)["panel_titles"] == 8
        assert figures.audit_data_safe_legends(figure)["data_safe_legends_checked"] == 1
        assert figures.audit_cross_axis_text_collisions(figure)["status"] == "PASS"
    finally:
        plt.close(figure)


def test_complete_export_bundle(tmp_path):
    try:
        figures.require_times_new_roman()
    except figures.FigureFontError:
        pytest.skip("Times New Roman is not installed on this platform")
    figure, axes = figures.create_panel_figure()
    try:
        for letter, axis in zip("abcdefgh", axes.ravel()):
            axis.plot([0, 1], [0, 1])
            axis.set_xticks([0, 1])
            axis.set_yticks([0, 1])
            figures.style_panel(axis, letter, "Export test")
        bundle = figures.save_figure_bundle(figure, tmp_path, "contract_test")
        audit = json.loads(bundle.qa.read_text(encoding="utf-8"))
        assert audit["status"] == "PASS"
        assert audit["png_600"]["pixels"] == [4252, 5433]
        assert audit["png_word"]["pixels"] == [2126, 2717]
        assert audit["svg"]["live_text_elements"] > 0
        assert audit["cross_axis_text"]["blocking_collision_count"] == 0
    finally:
        plt.close(figure)


def test_cross_panel_collision_and_safe_region_intrusion_are_blocking():
    figure, axes = _geometry_only_panel_figure()
    try:
        left = axes[0, 0].text(
            1.0,
            0.5,
            "COLLISION",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="center",
        )
        right = axes[0, 1].text(
            0.0,
            0.5,
            "COLLISION",
            transform=axes[0, 1].transAxes,
            ha="right",
            va="center",
        )
        axes[0, 0].text(
            1.2,
            0.2,
            "INTRUDE",
            transform=axes[0, 0].transAxes,
            ha="left",
            va="center",
        )
        figures.allow_same_panel_text_overlap(
            left,
            right,
            reason="deliberate invalid cross-panel exception used by the contract test",
        )
        report = figures.audit_cross_axis_text_collisions(figure, raise_on_error=False)
        assert report["status"] == "FAIL"
        assert any(
            item["type"] == "cross_axis_text_overlap"
            and item["panel_relation"] == "different_panels"
            and item.get("invalid_ignore_reason")
            for item in report["blocking_collisions"]
        )
        assert sum(
            item["type"] == "panel_safe_region_intrusion"
            for item in report["blocking_collisions"]
        ) >= 1
        with pytest.raises(figures.FigureContractError, match="cross-axis text collision"):
            figures.audit_cross_axis_text_collisions(figure)
    finally:
        plt.close(figure)


def test_exact_same_panel_cross_axis_overlap_can_be_documented():
    figure, axes = _geometry_only_panel_figure()
    try:
        first_axis = axes[0, 0].inset_axes([0.2, 0.2, 0.5, 0.5])
        second_axis = axes[0, 0].inset_axes([0.2, 0.2, 0.5, 0.5])
        for axis in (first_axis, second_axis):
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
        first = first_axis.text(0.5, 0.5, "intentional", ha="center", va="center")
        second = second_axis.text(0.5, 0.5, "intentional", ha="center", va="center")
        failing = figures.audit_cross_axis_text_collisions(
            figure, raise_on_error=False
        )
        assert failing["status"] == "FAIL"
        assert any(
            item["type"] == "cross_axis_text_overlap"
            and item["panel_relation"] == "same_panel"
            and item["texts"] == ["intentional", "intentional"]
            for item in failing["blocking_collisions"]
        )
        figures.allow_same_panel_text_overlap(
            first,
            second,
            reason="paired foreground and backing text in one panel",
        )
        report = figures.audit_cross_axis_text_collisions(figure)
        assert report["status"] == "PASS"
        assert report["ignored_same_panel_overlap_count"] == 1
    finally:
        plt.close(figure)
