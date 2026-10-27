"""Test annual conclusions against common 24, 25 and 26 C references.

Each common temperature is an external service anchor only.  It fixes the
least-served 10% of profiles and the service limits used to review the same
completed annual candidate library; it is never inserted as a controller
candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.global_extension.analyse_global_results import (
    summarise_information_diagnostic,
)
from analysis.global_extension.tail_fraction_sensitivity import (
    analyse_tail_fraction_sensitivity,
)


DEFAULT_REFERENCE_TARGETS_C = (24.0, 25.0, 26.0)
TAIL_FRACTION = 0.10
DETAIL_OUTPUTS = (
    "TAIL_FRACTION_RUN_METRICS.csv",
    "TAIL_FRACTION_ANCHORS.csv",
    "TAIL_FRACTION_MEMBERSHIPS.csv",
    "TAIL_FRACTION_DIRECT_COMPARISONS.csv",
    "TAIL_FRACTION_COMMON_SERVICE_SELECTIONS.csv",
    "TAIL_FRACTION_COMMON_SERVICE_COMPARISONS.csv",
    "TAIL_FRACTION_INFORMATION_DIAGNOSTIC.csv",
    "TAIL_FRACTION_SUMMARY.csv",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference_slug(reference_target_c: float) -> str:
    """Return a stable directory label for a finite reference temperature."""
    if not np.isfinite(reference_target_c):
        raise ValueError("Reference target must be finite")
    text = f"{float(reference_target_c):g}".replace("-", "m").replace(".", "p")
    return f"reference_{text}C"


def headline_row(reference_target_c: float, result: dict) -> dict:
    """Flatten the predeclared headline endpoints for one reference."""
    information = result["information"]["by_service_fraction"]["1"][
        "full_distribution_resolvable"
    ]
    common = result["common_service"]
    row: dict[str, object] = {
        "reference_target_c": float(reference_target_c),
        "full_resolvable_scenarios_at_fraction_1": information["scenarios"],
        "eligibility_disagreement_at_fraction_1": information[
            "eligibility_disagreement"
        ],
        "selection_disagreement_at_fraction_1": information[
            "selection_disagreement"
        ],
        "zone_mean_selected_plan_fails_full_review_at_fraction_1": information[
            "zone_mean_selected_plan_fails_full_review"
        ],
        "uniform_available_at_fraction_0p75": common["0.75"][
            "uniform_service_available"
        ],
        "uniform_available_at_fraction_0p50": common["0.5"][
            "uniform_service_available"
        ],
    }
    for control in ("personal_center", "personal_range"):
        prefix = control
        direct = result["direct"][control]
        direction = direct["facility_energy_direction"]
        row |= {
            f"{prefix}_direct_mean_mismatch_lower": direct[
                "population_mean_mismatch_lower"
            ],
            f"{prefix}_direct_tail_mismatch_lower": direct[
                "fixed_tail_mismatch_lower"
            ],
            f"{prefix}_direct_energy_increase": direction["increase"],
            f"{prefix}_direct_energy_decrease": direction["decrease"],
            f"{prefix}_available_at_fraction_0p75": common["0.75"][
                "alternatives"
            ][control]["control_service_available"],
            f"{prefix}_available_at_fraction_0p50": common["0.5"][
                "alternatives"
            ][control]["control_service_available"],
        }
    return row


def selection_failures_by_climate(
    reference_target_c: float, information: pd.DataFrame
) -> pd.DataFrame:
    """Count fraction-1 information failures in each climate."""
    required = {
        "climate",
        "service_fraction",
        "full_unresolved",
        "eligibility_disagreement",
        "selection_disagreement",
        "zone_mean_selected_plan_fails_full_review",
    }
    missing = required - set(information)
    if missing:
        raise ValueError(f"Information diagnostic lacks {sorted(missing)}")

    def boolean(values: pd.Series) -> pd.Series:
        if values.dtype == bool:
            return values
        result = values.astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )
        if result.isna().any():
            raise ValueError("Information diagnostic contains an invalid boolean")
        return result.astype(bool)

    scoped = information.loc[
        np.isclose(
            pd.to_numeric(information.service_fraction, errors="raise"), 1.0
        )
        & ~boolean(information.full_unresolved)
    ].copy()
    for name in (
        "eligibility_disagreement",
        "selection_disagreement",
        "zone_mean_selected_plan_fails_full_review",
    ):
        scoped[name] = boolean(scoped[name])
    rows = []
    for climate, group in scoped.groupby("climate", sort=True):
        rows.append(
            {
                "reference_target_c": float(reference_target_c),
                "climate": str(climate),
                "full_resolvable_scenarios_at_fraction_1": int(len(group)),
                "eligibility_disagreement_at_fraction_1": int(
                    group.eligibility_disagreement.sum()
                ),
                "selection_disagreement_at_fraction_1": int(
                    group.selection_disagreement.sum()
                ),
                "zone_mean_selected_plan_fails_full_review_at_fraction_1": int(
                    group.zone_mean_selected_plan_fails_full_review.sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _refresh_existing_audit(target_root: Path, reference_target_c: float) -> dict:
    """Refresh descriptive metadata without rescoring stored trajectories."""
    audit_path = target_root / "TAIL_FRACTION_SENSITIVITY.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing existing sensitivity audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    information = pd.read_csv(
        target_root / "TAIL_FRACTION_INFORMATION_DIAGNOSTIC.csv"
    )
    audit["reference_target_c"] = float(reference_target_c)
    audit["interpretation"] = (
        "Tail shares are predeclared sensitivity endpoints for the fixed "
        f"least-served profiles under the common {reference_target_c:g} C "
        "reference. They are not clinical, physiological or policy "
        "vulnerability standards."
    )
    audit["results"][f"{TAIL_FRACTION:g}"]["information"] = (
        summarise_information_diagnostic(
            information, reference_target_c=reference_target_c
        )
    )
    audit["output_sha256"] = {
        name: sha256(target_root / name) for name in DETAIL_OUTPUTS
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def analyse_reference_anchor_sensitivity(
    metrics_path: Path,
    run_manifest_path: Path,
    profiles_path: Path,
    anchor_manifest_path: Path,
    output_root: Path,
    *,
    reference_targets_c: tuple[float, ...] = DEFAULT_REFERENCE_TARGETS_C,
    reuse_existing: bool = False,
) -> dict:
    targets = tuple(float(value) for value in reference_targets_c)
    if not targets or any(not np.isfinite(value) for value in targets):
        raise ValueError("Reference targets must be finite")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("Reference targets must be unique and increasing")

    output_root.mkdir(parents=True, exist_ok=True)
    audits: dict[str, dict] = {}
    headline_rows: list[dict] = []
    climate_failure_rows: list[pd.DataFrame] = []
    combined: dict[str, list[pd.DataFrame]] = {name: [] for name in DETAIL_OUTPUTS}
    subaudit_hashes: dict[str, str] = {}

    for target in targets:
        slug = reference_slug(target)
        target_root = output_root / slug
        audit = (
            _refresh_existing_audit(target_root, target)
            if reuse_existing
            else analyse_tail_fraction_sensitivity(
                metrics_path,
                run_manifest_path,
                profiles_path,
                anchor_manifest_path,
                target_root,
                tail_fractions=(TAIL_FRACTION,),
                reference_target_c=target,
            )
        )
        result = audit["results"][f"{TAIL_FRACTION:g}"]
        audits[f"{target:g}"] = result
        headline_rows.append(headline_row(target, result))
        subaudit = target_root / "TAIL_FRACTION_SENSITIVITY.json"
        subaudit_hashes[slug] = sha256(subaudit)
        climate_failure_rows.append(
            selection_failures_by_climate(
                target,
                pd.read_csv(target_root / "TAIL_FRACTION_INFORMATION_DIAGNOSTIC.csv"),
            )
        )
        for name in DETAIL_OUTPUTS:
            frame = pd.read_csv(target_root / name)
            frame.insert(0, "reference_target_c", target)
            combined[name].append(frame)

    outputs: dict[str, pd.DataFrame] = {
        name.replace("TAIL_FRACTION", "REFERENCE_ANCHOR"): pd.concat(
            frames, ignore_index=True
        )
        for name, frames in combined.items()
    }
    outputs["REFERENCE_ANCHOR_HEADLINE.csv"] = pd.DataFrame(headline_rows)
    outputs["REFERENCE_ANCHOR_SELECTION_FAILURES_BY_CLIMATE.csv"] = pd.concat(
        climate_failure_rows, ignore_index=True
    )
    for name, frame in outputs.items():
        frame.to_csv(output_root / name, index=False)

    audit = {
        "status": "PASS",
        "definition": (
            "Common 24, 25 and 26 C trajectories are external references used "
            "to fix the least-served 10% of profiles and service limits. They "
            "are not controller candidates."
        ),
        "reference_targets_c": list(targets),
        "tail_fraction": TAIL_FRACTION,
        "results": audits,
        "selection_failures_by_climate_fraction_1": {
            f"{target:g}": {
                row.climate: int(
                    row.zone_mean_selected_plan_fails_full_review_at_fraction_1
                )
                for row in outputs[
                    "REFERENCE_ANCHOR_SELECTION_FAILURES_BY_CLIMATE.csv"
                ]
                .loc[lambda frame: np.isclose(frame.reference_target_c, target)]
                .itertuples(index=False)
            }
            for target in targets
        },
        "input_sha256": {
            "metrics": sha256(metrics_path),
            "run_manifest": sha256(run_manifest_path),
            "profiles": sha256(profiles_path),
            "anchor_manifest": sha256(anchor_manifest_path),
        },
        "subaudit_sha256": subaudit_hashes,
        "output_sha256": {name: sha256(output_root / name) for name in outputs},
    }
    (output_root / "REFERENCE_ANCHOR_SENSITIVITY.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--anchor-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--reference-targets-c",
        type=float,
        nargs="+",
        default=DEFAULT_REFERENCE_TARGETS_C,
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    audit = analyse_reference_anchor_sensitivity(
        args.metrics.resolve(),
        args.run_manifest.resolve(),
        args.profiles.resolve(),
        args.anchor_manifest.resolve(),
        args.output_root.resolve(),
        reference_targets_c=tuple(args.reference_targets_c),
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
