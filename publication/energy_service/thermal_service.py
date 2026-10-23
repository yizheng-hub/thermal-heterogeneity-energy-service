"""Shared service endpoints for the frozen-building control comparison.

Centres and intervals are model-derived references, not satisfaction labels.
Native zone occupancy is divided equally among assigned profiles in each zone.
"""

from __future__ import annotations

import numpy as np


def score_service(
    temperature_c,
    native_people,
    profile_zone,
    centre_c,
    lower_c,
    upper_c,
    fixed_tail,
    timestep_hours=1.0,
):
    """Return aggregate, hourly and per-profile temperature-reference metrics.

    Arrays of temperatures and occupancy have shape (time, existing_zone).
    Profile arrays have one entry per demand profile. ``fixed_tail`` contains
    profile indices fixed from a separate reference run, never the tested plan.
    Occupancy-zero hours do not enter either service or strict coverage.
    """
    temperature = np.asarray(temperature_c, dtype=float)
    people = np.asarray(native_people, dtype=float)
    zone = np.asarray(profile_zone, dtype=int)
    centre = np.asarray(centre_c, dtype=float)
    lower = np.asarray(lower_c, dtype=float)
    upper = np.asarray(upper_c, dtype=float)
    tail = np.asarray(fixed_tail, dtype=int)
    if temperature.ndim != 2 or temperature.shape != people.shape:
        raise ValueError("Temperature and native occupancy must share (time, zone) shape")
    if not (zone.ndim == centre.ndim == lower.ndim == upper.ndim == 1):
        raise ValueError("Profile inputs must be one-dimensional")
    if not (zone.size == centre.size == lower.size == upper.size):
        raise ValueError("Profile inputs must have equal lengths")
    if not all(np.isfinite(a).all() for a in (temperature, people, centre, lower, upper)):
        raise ValueError("Service inputs must be finite")
    if (people < 0).any() or (lower > upper).any() or timestep_hours <= 0:
        raise ValueError("Occupancy, intervals or timestep are invalid")
    if zone.size == 0 or zone.min() < 0 or zone.max() >= temperature.shape[1]:
        raise ValueError("Every profile must reference an existing thermal zone")
    if tail.size == 0 or np.unique(tail).size != tail.size:
        raise ValueError("A nonempty fixed reference-tail set is required")
    if tail.min() < 0 or tail.max() >= zone.size:
        raise ValueError("Fixed-tail index is outside the assigned profile set")

    counts = np.bincount(zone, minlength=temperature.shape[1])
    if ((counts == 0) & (people.sum(axis=0) > 0)).any():
        raise ValueError("An occupied zone has no assigned demand profile")
    weight = people[:, zone] / counts[zone]
    person_hours = weight.sum(axis=0) * timestep_hours
    total_person_hours = float(person_hours.sum())
    if total_person_hours <= 0 or (person_hours <= 0).any():
        raise ValueError("The evaluated window must include occupancy for every profile")
    profile_temperature = temperature[:, zone]
    centre_error = np.abs(profile_temperature - centre)
    interval_error = np.maximum(lower - profile_temperature, 0) + np.maximum(
        profile_temperature - upper, 0
    )
    covered = interval_error <= 1e-9
    profile_centre = (centre_error * weight).sum(axis=0) * timestep_hours / person_hours
    profile_excess = (interval_error * weight).sum(axis=0) * timestep_hours / person_hours
    profile_coverage = (covered * weight).sum(axis=0) * timestep_hours / person_hours
    occupied = weight > 0
    strict_profile = np.all(covered | ~occupied, axis=0)
    hourly_weight = weight.sum(axis=1)
    # The locked tail endpoint averages profile-level distances equally, after
    # each profile's own occupied-time average; native weights apply to the
    # population endpoint, not a second time to this fixed-profile endpoint.
    hourly_tail_weight = occupied[:, tail].sum(axis=1)

    def hourly_mean(values, weights, denominator):
        numerator = (values * weights).sum(axis=1)
        return np.divide(
            numerator, denominator, out=np.full_like(numerator, np.nan),
            where=denominator > 0,
        )

    degree_hours = float((interval_error * weight).sum() * timestep_hours)
    summary = {
        "occupied_native_person_hours": total_person_hours,
        "mean_reference_mismatch_c": float(np.dot(profile_centre, person_hours) / total_person_hours),
        "fixed_tail_reference_mismatch_c": float(profile_centre[tail].mean()),
        "modeled_interval_person_hour_coverage": float(np.dot(profile_coverage, person_hours) / total_person_hours),
        "modeled_interval_mean_excess_c": degree_hours / total_person_hours,
        "modeled_interval_excess_native_person_c_hours": degree_hours,
        "profile_coverage_p10": float(np.quantile(profile_coverage, 0.1)),
        "profile_coverage_median": float(np.median(profile_coverage)),
        "profile_coverage_minimum": float(profile_coverage.min()),
        "profiles_with_no_occupied_hour_interval_violation": int(strict_profile.sum()),
        "assigned_profile_count": int(zone.size),
        "all_profiles_without_interval_violation": bool(strict_profile.all()),
    }
    profiles = {
        "reference_mismatch_c": profile_centre,
        "modeled_interval_mean_excess_c": profile_excess,
        "modeled_interval_coverage": profile_coverage,
        "native_person_hour_weight": person_hours,
        "no_occupied_hour_interval_violation": strict_profile,
    }
    hourly = {
        "native_people": hourly_weight,
        "mean_reference_mismatch_c": hourly_mean(centre_error, weight, hourly_weight),
        "fixed_tail_reference_mismatch_c": hourly_mean(
            centre_error[:, tail], occupied[:, tail], hourly_tail_weight
        ),
        "modeled_interval_coverage": hourly_mean(covered, weight, hourly_weight),
        "modeled_interval_mean_excess_c": hourly_mean(interval_error, weight, hourly_weight),
    }
    return {"summary": summary, "profiles": profiles, "hourly": hourly}
