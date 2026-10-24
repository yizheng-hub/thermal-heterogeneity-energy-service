"""Report rendered text geometry for explicit visual review, never a visual PASS."""

from __future__ import annotations
from matplotlib.text import Text


def audit_figure(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = fig.bbox
    entries = []
    outside = []
    collisions = []
    hidden_ticks = set()
    for ax in fig.findobj(__import__("matplotlib").axes.Axes):
        for ticker, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            low, high = sorted(limits)
            for tick in ticker.get_major_ticks() + ticker.get_minor_ticks():
                if tick.get_loc() < low - 1e-9 or tick.get_loc() > high + 1e-9:
                    hidden_ticks.update([id(tick.label1), id(tick.label2)])
    for artist in fig.findobj(Text):
        if not artist.get_visible() or not artist.get_text().strip() or id(artist) in hidden_ticks:
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        entry = {"text": artist.get_text(), "bounds_px": [round(x, 2) for x in box.bounds]}
        entries.append((artist, box, entry))
        if (
            box.x0 < frame.x0 - 0.5
            or box.y0 < frame.y0 - 0.5
            or box.x1 > frame.x1 + 0.5
            or box.y1 > frame.y1 + 0.5
        ):
            outside.append(entry)
    for i, (left, lb, le) in enumerate(entries):
        for right, rb, re in entries[i + 1 :]:
            if left is right:
                continue
            dx = min(lb.x1, rb.x1) - max(lb.x0, rb.x0)
            dy = min(lb.y1, rb.y1) - max(lb.y0, rb.y0)
            if dx > 1 and dy > 1:
                collisions.append(
                    {"left": le, "right": re, "intersection_px": [round(dx, 2), round(dy, 2)]}
                )
    return {
        "status": "REQUIRES_VISUAL_REVIEW",
        "visible_texts": len(entries),
        "clipped_texts": outside,
        "text_overlap_candidates": collisions,
        "note": "Bounding boxes include glyph whitespace. Inspect all full-size figures and every candidate; this report does not certify readability or absence of text-data intersections.",
    }
