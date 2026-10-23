"""Static interval compatibility within each existing well-mixed thermal zone.

This is a temperature-overlap bound under the model-derived intervals. It is
not a measured comfort rate and does not prove an HVAC system can attain the
temperature. A common command across zones does not imply equal temperatures,
so the calculation must not merge zones into floors or whole buildings.
"""

import numpy as np


def room_interval_feasibility(profile_zone, lower_c, upper_c):
    zone = np.asarray(profile_zone)
    lower = np.asarray(lower_c, dtype=float)
    upper = np.asarray(upper_c, dtype=float)
    if not (zone.ndim == lower.ndim == upper.ndim == 1):
        raise ValueError("Profile zone and interval arrays must be one-dimensional")
    if not (zone.size == lower.size == upper.size) or zone.size == 0:
        raise ValueError("Equal nonzero profile-array lengths are required")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (lower > upper).any():
        raise ValueError("Each profile must have a finite ordered interval")
    rows = []
    for value in np.unique(zone):
        selected = zone == value
        lo, hi = lower[selected], upper[selected]
        coordinates = np.unique(np.r_[lo, hi])
        simultaneous = ((coordinates[:, None] >= lo) & (coordinates[:, None] <= hi)).sum(axis=1)
        best = int(simultaneous.argmax())
        rows.append({
            "zone_id": value.item() if hasattr(value, "item") else value,
            "assigned_profile_count": int(lo.size),
            "max_simultaneously_covered_profiles": int(simultaneous[best]),
            "max_simultaneous_modeled_interval_fraction": float(simultaneous[best] / lo.size),
            "example_max_overlap_temperature_c": float(coordinates[best]),
            "common_interval_lower_c": float(lo.max()),
            "common_interval_upper_c": float(hi.min()),
            "all_profile_intersection_exists": bool(lo.max() <= hi.min()),
        })
    return rows
