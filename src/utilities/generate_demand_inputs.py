"""Replay the locked upstream numerical demand construction from external inputs.

This optional entry does not alter the fixed publication-analysis inputs. It
requires the original checksum-matched ASHRAE files and either the released
numeric-only scene specification or the author-retained original scene pool.
Question text, dialogue records and original record/subject identifiers are
never exported. Fresh replay labels do not recreate prior release identifiers.

The PMV routine is retained from the study's archived numerical fallback in
run_full_archive_virtual_ltpref.py, which implements the Fanger heat balance.
The earlier study workflow used pythermalcomfort for a separate compatibility
comparison. That package is neither imported nor bundled by this entry point.
See THIRD_PARTY_NOTICES.md for the recorded provenance boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOCKED_INPUT_SHA256 = {
    "measurements": "23711ee1cfd988501d1ee77fc2207884e3edcb126cf96c0cd30a5380f39bd5e0",
    "metadata": "af6770a2b469f079cd172bd6632d1ae726a3a663515126816ff6d2d8f51bfc10",
    "scene_pool": "8d40dbd3a996ef0eac5b1f6cd4c9976e5154cb222f54f84e5e77f3edf2803076",
    "numeric_scenes": "7397a15d4b9492b6fb0cb94e8e4f248f5c4da69b7ba40d7836f3b106396a5744",
}
PMV_VALUES = np.arange(-3.0, 3.0 + 0.1, 0.1)
UNIFORM = np.ones_like(PMV_VALUES) / len(PMV_VALUES)
MAX_ROUNDS = 19


def verify_input(path: Path, kind: str) -> str:
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    if observed != LOCKED_INPUT_SHA256[kind]:
        raise ValueError(f"Locked {kind} SHA-256 differs")
    return observed


SCENE_LOGIC = {
    "cinema_strong_cooling_summer": "COMPLAIN_COLD",
    "shopping_mall_cooling_summer": "COMPLAIN_COLD",
    "office_transition_cool": "COMPLAIN_COLD",
    "office_avoid_cold_draft": "COMPLAIN_COLD",
    "office_avoid_draft": "COMPLAIN_COLD",
    "summer_adjust_ac_frequently": "COMPLAIN_HEAT",
    "office_warm_tolerable_summer": "TOLERATE_HEAT",
    "summer_no_ac_energy_saving": "TOLERATE_HEAT",
    "short_stay_no_ac": "TOLERATE_HEAT",
    "metro_strong_cooling_summer": "PREFER_COLD",
    "office_ac_strong_cooling_summer": "PREFER_COLD",
    "metro_weak_cooling_summer": "PREFER_WARM",
    "office_slightly_warm_preference": "PREFER_WARM",
    "sunlit_room_preference": "PREFER_WARM",
    "meeting_room_low_cooling": "PREFER_WARM",
    "summer_bbq_outdoor_choice": "PREFER_WARM",
    "summer_bbq_indoor_ac_choice": "PREFER_COLD",
    "office_initial_cold_adaptation": "PREFER_NEUTRAL",
    "neutral_indoor_unnoticed_ac": "PREFER_NEUTRAL",
}


def pmv_fanger(
    tdb: float,
    tr: float,
    vr: float,
    rh: float,
    met: float,
    clo: float,
    wme: float = 0.0,
) -> float:
    """Return Fanger PMV using the ISO 7730 numerical implementation.

    ASHRAE PMV is identical to ISO PMV when the cooling-effect adjustment is
    zero.  The historical simulator used 0.1 m/s for all centre/bound mapping,
    so that is the relevant branch here.
    """

    values = (tdb, tr, vr, rh, met, clo, wme)
    if not all(np.isfinite(values)) or met <= 0 or clo < 0 or vr < 0:
        return float("nan")

    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (tdb + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(vr)
    taa = tdb + 273.0
    tra = tr + 273.0
    tcla = taa + (35.5 - tdb) / (3.5 * icl + 0.1)
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = tcla / 50.0

    for _ in range(151):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = max(hcf, hcn)
        xn = (p5 + p4 * hc - p2 * xn**4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= 0.00015:
            break
    else:
        return float("nan")

    tcl = 100.0 * xn - 273.0
    hl1 = 3.05e-3 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7e-5 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - tdb)
    hl5 = 3.96 * fcl * (xn**4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - tdb)
    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    return ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)


def calculate_scene_pmv(scene: dict[str, Any], clo: float, met: float) -> float:
    physical = scene["physical_parameters"]
    value = pmv_fanger(
        float(physical["air_temp"]),
        float(physical["mrt"]),
        float(physical["air_velocity"]),
        float(physical["relative_humidity"]),
        met,
        clo,
    )
    if np.isfinite(value):
        return value
    return float(np.mean(scene["pmv_range"]))


def construct_likelihood(
    scene: dict[str, Any],
    logic_type: str,
    response_yes: bool,
    dynamic_center: float | None,
) -> np.ndarray:
    profile = scene.get("evidence_influence_profile", {}) or {}
    secondary_weight = float(profile.get("secondary_weight", 0.3))
    configured_tau = profile.get("decay_tau")
    p_min, p_max = map(float, scene["pmv_range"])
    p_mean = (
        float(dynamic_center)
        if dynamic_center is not None
        else (p_min + p_max) / 2.0
    )
    p_width = max(p_max - p_min, 0.5)
    dynamic_sigma = p_width * 2.0
    dynamic_tau = (
        float(configured_tau)
        if configured_tau is not None
        else max(p_width * 1.5, 0.8)
    )
    comfort = np.exp(-0.5 * ((PMV_VALUES - p_mean) / (p_width * 2.0)) ** 2)

    if logic_type == "COMPLAIN_COLD":
        yes = np.exp(-0.5 * ((PMV_VALUES - (p_mean + 0.3)) / (dynamic_sigma * 1.2)) ** 2)
        no = np.maximum(1.0 - yes, comfort)
    elif logic_type == "COMPLAIN_HEAT":
        yes = np.exp(-0.5 * ((PMV_VALUES - (p_mean - 0.3)) / (dynamic_sigma * 1.2)) ** 2)
        no = np.maximum(1.0 - yes, comfort)
    elif logic_type == "TOLERATE_HEAT":
        yes = np.exp(-0.5 * ((PMV_VALUES - (p_mean + 0.2)) / (dynamic_sigma * 1.5)) ** 2)
        no = 1.0 / (1.0 + np.exp((PMV_VALUES - p_mean) / dynamic_tau))
    elif logic_type == "PREFER_WARM":
        yes = np.exp(-0.5 * ((PMV_VALUES - (p_mean + 0.5)) / dynamic_sigma) ** 2)
        no = 1.0 - yes
    elif logic_type == "PREFER_COLD":
        yes = np.exp(-0.5 * ((PMV_VALUES - (p_mean - 0.5)) / dynamic_sigma) ** 2)
        no = 1.0 - yes
    else:
        yes = comfort
        no = 1.0 - yes

    selected = yes if response_yes else no
    result = (1.0 - secondary_weight) * selected + secondary_weight / len(PMV_VALUES)
    return np.clip(result, 1e-6, 1.0)


def update_posterior(
    prior: np.ndarray,
    scene: dict[str, Any],
    logic_type: str,
    response_yes: bool,
    dynamic_center: float,
) -> np.ndarray:
    likelihood = construct_likelihood(scene, logic_type, response_yes, dynamic_center)
    strength = float(scene.get("evidence_strength", 1.0))
    if logic_type.startswith("COMPLAIN"):
        strength *= 0.6
    posterior = prior * likelihood**strength
    total = posterior.sum()
    if total < 1e-9:
        posterior = prior * 0.5 + UNIFORM * 0.5
        total = posterior.sum()
    posterior = posterior / total
    peak_index = int(np.argmax(posterior))
    edge_factor = abs(peak_index / (len(posterior) - 1) - 0.5)
    uniform_weight = max(0.03, 0.12 * (1.0 - edge_factor))
    return (1.0 - uniform_weight) * posterior + uniform_weight * UNIFORM


def entropy(probabilities: np.ndarray) -> float:
    safe = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def inferred_range(probabilities: np.ndarray, confidence_level: float) -> tuple[float, float]:
    peak = int(np.argmax(probabilities))
    probability = float(probabilities[peak])
    left = right = peak
    while probability < confidence_level:
        can_left = left > 0
        can_right = right < len(probabilities) - 1
        if not can_left and not can_right:
            break
        p_left = float(probabilities[left - 1]) if can_left else -1.0
        p_right = float(probabilities[right + 1]) if can_right else -1.0
        if p_left >= p_right:
            left -= 1
            probability += p_left
        else:
            right += 1
            probability += p_right
    return float(PMV_VALUES[left]), float(PMV_VALUES[right])


def information_gain(scene: dict[str, Any], posterior: np.ndarray) -> float:
    """Reproduce the historical candidate-ranking objective.

    The legacy implementation ranked every candidate with its neutral-response
    likelihood centred on the scene's declared PMV range, then asked the
    conversational strategist to choose among the top three.  The offline
    implementation deterministically chooses rank one.
    """

    yes = construct_likelihood(scene, "PREFER_NEUTRAL", True, None)
    no = construct_likelihood(scene, "PREFER_NEUTRAL", False, None)
    p_yes = float(np.sum(posterior * yes))
    post_yes = posterior * yes
    post_no = posterior * no
    post_yes /= post_yes.sum()
    post_no /= post_no.sum()
    return entropy(posterior) - (p_yes * entropy(post_yes) + (1.0 - p_yes) * entropy(post_no))


def intended_answer(scene: dict[str, Any], scene_pmv: float, true_ideal_pmv: float) -> bool:
    scene_id = scene["scene_id"]
    if scene_id == "summer_bbq_indoor_ac_choice":
        return abs(true_ideal_pmv - 0.5) < abs(true_ideal_pmv - 1.5)
    if scene_id == "summer_bbq_outdoor_choice":
        return abs(true_ideal_pmv - 1.5) < abs(true_ideal_pmv - 0.5)
    if scene_id == "metro_strong_cooling_summer":
        return abs(true_ideal_pmv + 1.1) < abs(true_ideal_pmv + 0.4)
    if scene_id == "metro_weak_cooling_summer":
        return abs(true_ideal_pmv + 0.4) < abs(true_ideal_pmv + 1.1)

    logic_type = SCENE_LOGIC.get(scene_id, "PREFER_NEUTRAL")
    delta = scene_pmv - true_ideal_pmv
    if logic_type == "COMPLAIN_COLD":
        return delta < -0.3
    if logic_type == "COMPLAIN_HEAT":
        return delta > 0.3
    if logic_type == "PREFER_COLD":
        return true_ideal_pmv < -0.3
    if logic_type == "PREFER_WARM":
        return true_ideal_pmv > 0.3
    if logic_type == "PREFER_NEUTRAL":
        return abs(delta) <= 0.5
    if logic_type == "TOLERATE_HEAT":
        return true_ideal_pmv >= -0.3
    raise ValueError(f"Unknown scene logic for {scene_id}")


def temperature_for_pmv_unbounded(target_pmv: float, clo: float, met: float) -> tuple[float, bool]:
    lower = 5.0
    upper = 40.0
    lower_pmv = pmv_fanger(lower, lower, 0.1, 50.0, met, clo)
    upper_pmv = pmv_fanger(upper, upper, 0.1, 50.0, met, clo)
    if target_pmv <= lower_pmv:
        return lower, True
    if target_pmv >= upper_pmv:
        return upper, True
    for _ in range(40):
        middle = (lower + upper) / 2.0
        middle_pmv = pmv_fanger(middle, middle, 0.1, 50.0, met, clo)
        if middle_pmv < target_pmv:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0, False

def build_cohort(measurements: pd.DataFrame, metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep original grouping keys in memory only until anonymous output is built."""
    if measurements.record_id.duplicated().any() or metadata.building_id.duplicated().any():
        raise ValueError("Raw measurement and metadata keys must be unique")
    frame = measurements.merge(metadata, on="building_id", how="left", validate="many_to_one")
    def clean(column):
        return frame[column].astype("string").str.strip().str.lower()
    met = pd.to_numeric(frame.met, errors="coerce")
    sensation = pd.to_numeric(frame.thermal_sensation, errors="coerce")
    operative = pd.to_numeric(frame.top, errors="coerce")
    air = pd.to_numeric(frame.ta, errors="coerce")
    keep = (clean("building_type").eq("office")
            & clean("cooling_type").eq("air conditioned")
            & met.between(1.0, 1.3) & sensation.notna() & (operative.notna() | air.notna()))
    selected = frame.loc[keep].copy()
    selected["met"] = met.loc[keep]
    selected["thermal_sensation"] = sensation.loc[keep]
    selected["observed_temperature_c"] = operative.loc[keep].fillna(air.loc[keep])
    selected["temperature_source"] = np.where(operative.loc[keep].notna(), "operative", "air_fallback")
    publication = selected.publication.astype("string").str.strip()
    contributor = selected.contributor.astype("string").str.strip()
    selected["source_group"] = publication.where(publication.notna() & publication.ne(""),
                                                   "contributor::" + contributor)
    known = selected.subject_id.notna()
    selected["unit_group"] = np.where(
        known, "building::" + selected.building_id.astype(str) + "::subject::" + selected.subject_id.astype(str),
        "record::" + selected.record_id.astype(str))
    for label, g in (("033", .33), ("050", .50), ("067", .67)):
        selected[f"reference_g{label}_c"] = selected.observed_temperature_c - selected.thermal_sensation / g
    counts = selected.source_group.value_counts(dropna=False)
    source_weight = selected.source_group.map(counts).rdiv(1.0).to_numpy(dtype=float)
    selected["record_weight"] = 1.0
    selected["unit_weight"] = 1.0 / selected.groupby("unit_group").record_id.transform("count")
    selected["building_weight"] = 1.0 / selected.groupby("building_id").record_id.transform("count")
    selected["source_weight"] = source_weight / source_weight.mean()
    selected = selected.sort_values("record_id").reset_index(drop=True)
    selected["clo_analysis"] = pd.to_numeric(selected.clo, errors="coerce")
    selected["clothing_imputation"] = np.where(selected.clo_analysis.notna(), "observed", "missing")
    for columns, label in ((["source_group", "season"], "source_season_median"),
                           (["source_group"], "source_median"),
                           (["climate", "season"], "climate_season_median"),
                           (["season"], "season_median")):
        medians = selected.groupby(columns, dropna=False).clo.transform("median")
        usable = selected.clo_analysis.isna() & medians.notna()
        selected.loc[usable, "clo_analysis"] = medians.loc[usable]
        selected.loc[usable, "clothing_imputation"] = label
    missing = selected.clo_analysis.isna()
    selected.loc[missing, "clo_analysis"] = float(selected.clo.median())
    selected.loc[missing, "clothing_imputation"] = "global_median"
    scope = {"source_measurements": len(measurements), "records": len(selected),
             "analysis_units": selected.unit_group.nunique(), "buildings": selected.building_id.nunique(),
             "cities": selected.city.nunique(), "countries": selected.country.nunique(),
             "observed_clothing": int(selected.clothing_imputation.eq("observed").sum()),
             "imputed_clothing": int(selected.clothing_imputation.ne("observed").sum())}
    return selected, scope


def anonymous_inputs(cohort: pd.DataFrame) -> pd.DataFrame:
    columns = ["observed_temperature_c", "temperature_source", "thermal_sensation", "met", "clo_analysis",
               "clothing_imputation", "reference_g033_c", "reference_g050_c", "reference_g067_c",
               "record_weight", "unit_weight", "building_weight", "source_weight"]
    result = cohort[columns].copy()
    result.insert(0, "replay_row", np.arange(len(result)))
    for original, name in (("unit_group", "replay_unit"), ("building_id", "replay_building"),
                            ("source_group", "replay_source")):
        result[name] = pd.factorize(cohort[original], sort=True)[0]
    return result


def numeric_scenes(path: Path, *, numeric_only: bool = False) -> list[dict]:
    verify_input(path, "numeric_scenes" if numeric_only else "scene_pool")
    original = json.loads(path.read_text(encoding="utf-8"))
    if len(original) != MAX_ROUNDS or {s["scene_id"] for s in original} != set(SCENE_LOGIC):
        raise ValueError("Expected the locked 19-scene set")
    scenes = []
    for scene in original:
        influence = scene.get("evidence_influence_profile", {}) or {}
        scenes.append({"scene_id": scene["scene_id"], "pmv_range": scene["pmv_range"],
                       "physical_parameters": {k: scene["physical_parameters"][k]
                           for k in ("air_temp", "mrt", "air_velocity", "relative_humidity")},
                       "evidence_strength": scene.get("evidence_strength", 1.0),
                       "evidence_influence_profile": {k: influence[k]
                           for k in ("secondary_weight", "decay_tau") if k in influence}})
    return scenes


def virtual_interval(reference: float, clo: float, met: float, scenes: list[dict]) -> dict:
    """Replay all 19 numerical updates and the original unconstrained inversion."""
    true_pmv = pmv_fanger(reference, reference, .1, 50., met, clo)
    if not np.isfinite(true_pmv):
        raise ValueError("Invalid numerical persona PMV")
    true_pmv = float(np.clip(true_pmv, PMV_VALUES[0], PMV_VALUES[-1]))
    posterior = UNIFORM.copy()
    remaining = list(scenes)
    for round_number in range(1, MAX_ROUNDS + 1):
        ranked = sorted(((information_gain(s, posterior), s["scene_id"], s) for s in remaining),
                        key=lambda item: (-item[0], item[1]))
        _, scene_id, scene = ranked[0]
        remaining = [s for s in remaining if s["scene_id"] != scene_id]
        scene_pmv = calculate_scene_pmv(scene, clo, met)
        answer = intended_answer(scene, scene_pmv, true_pmv)
        posterior = update_posterior(posterior, scene, SCENE_LOGIC[scene_id], answer, scene_pmv)
        low, high = inferred_range(posterior, .7 if round_number > 4 else .8)
    centre_pmv = float(np.sum(posterior * PMV_VALUES))
    low_c, _ = temperature_for_pmv_unbounded(low, clo, met)
    high_c, _ = temperature_for_pmv_unbounded(high, clo, met)
    half = abs(high_c - low_c) / 2.0
    lower = float(np.clip(reference - half, 16., 30.))
    upper = float(np.clip(reference + half, 16., 30.))
    return {"posterior_low_pmv": low, "posterior_high_pmv": high,
            "posterior_mean_pmv": centre_pmv, "interval_half_width_c": half,
            "interval_lower_c": lower, "interval_centre_c": reference,
            "interval_upper_c": upper, "main_control_eligible": bool(upper > lower),
            "virtual_model_derived": True, "rounds": MAX_ROUNDS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    scenes = parser.add_mutually_exclusive_group(required=True)
    scenes.add_argument("--scene-pool", type=Path, help="Author-retained original configuration")
    scenes.add_argument("--numeric-scenes", type=Path, help="Released numeric-only 19-scene specification")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError("Choose a new upstream replay output directory")
    scene_path = args.numeric_scenes or args.scene_pool
    scene_kind = "numeric_scenes" if args.numeric_scenes else "scene_pool"
    hashes = {"measurements": verify_input(args.measurements, "measurements"),
              "metadata": verify_input(args.metadata, "metadata"),
              scene_kind: verify_input(scene_path, scene_kind)}
    scenes = numeric_scenes(scene_path, numeric_only=bool(args.numeric_scenes))
    cohort, scope = build_cohort(pd.read_csv(args.measurements, low_memory=False),
                                  pd.read_csv(args.metadata, low_memory=False))
    if scope["records"] != 21418 or scope["analysis_units"] != 12130:
        raise ValueError("Raw cohort differs from the locked study cohort")
    profiles = anonymous_inputs(cohort)
    rows = []
    for index, row in profiles.iterrows():
        result = virtual_interval(float(row.reference_g050_c), float(row.clo_analysis),
                                  float(row.met), scenes)
        rows.append({"replay_row": int(row.replay_row), **result})
        if (index + 1) % 1000 == 0:
            print(f"Numerical demand replay {index + 1}/{len(profiles)}", flush=True)
    intervals = pd.DataFrame(rows)
    if int(intervals.main_control_eligible.sum()) != 21399:
        raise ValueError("Executable interval count differs from the locked 21399")
    args.output_root.mkdir(parents=True)
    profiles.to_csv(args.output_root / "anonymous_numeric_observations.csv", index=False,
                    float_format="%.17g")
    intervals.to_csv(args.output_root / "virtual_intervals_numeric.csv", index=False,
                     float_format="%.17g")
    report = {"status": "PASS", "scope": "Optional upstream numerical replay, separate from fixed analysis inputs",
              "input_sha256": hashes, "cohort": scope, "virtual_intervals": len(intervals),
              "executable_intervals": int(intervals.main_control_eligible.sum()), "rounds_per_record": MAX_ROUNDS,
              "original_identifiers_exported": False, "question_text_or_dialogues_exported": False,
              "replay_labels": "Fresh deterministic numerical labels; not prior release_profile_id labels",
              "external_scene_pool_redistributed": False, "fixed_publication_inputs_changed": False,
              "scene_input_mode": "released_numeric_only" if args.numeric_scenes else "original_external_pool",
              "energyplus_simulations": 0,
              "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output_root.glob("*.csv"))}}
    (args.output_root / "UPSTREAM_NUMERIC_REPLAY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
