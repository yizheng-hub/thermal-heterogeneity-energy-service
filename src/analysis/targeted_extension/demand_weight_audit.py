"""Verify that demand support points preserve native physical occupancy weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_KEYS = ["context", "composition_id", "layout"]


def audit_profiles(
    profiles: pd.DataFrame, contexts: dict, controller_root: Path, source: str
) -> pd.DataFrame:
    rows = []
    for key, group in profiles.groupby(GROUP_KEYS, sort=True):
        context, composition_id, layout = key
        config = contexts[context]
        with np.load(controller_root / config["probes"], allow_pickle=False) as stored:
            people = stored["native_people"].astype(float)
            zones = stored["zone_ids"].astype(str).tolist()
        counts = group.zone_id.value_counts().reindex(zones, fill_value=0).to_numpy(int)
        if (counts <= 0).any():
            raise ValueError(f"{source}/{key}: an existing zone has no demand support point")
        assigned = np.repeat(np.arange(len(zones)), counts)
        # Every physical zone occupancy is divided among its demand support
        # points. Summing those weights must exactly recover native occupancy.
        profile_weights = people[:, assigned] / counts[assigned]
        recovered = np.column_stack(
            [profile_weights[:, assigned == zone].sum(axis=1) for zone in range(len(zones))]
        )
        difference = recovered - people
        rows.append(
            {
                "data_source": source,
                "context": context,
                "composition_id": composition_id,
                "layout": layout,
                "demand_support_points": len(group),
                "native_thermal_zones": len(zones),
                "minimum_support_points_per_zone": int(counts.min()),
                "maximum_support_points_per_zone": int(counts.max()),
                "native_person_hours": float(people.sum()),
                "recovered_person_hours": float(recovered.sum()),
                "maximum_hour_zone_weight_error": float(np.abs(difference).max()),
                "person_hour_error": float(difference.sum()),
                "weight_conservation_pass": bool(np.allclose(recovered, people, atol=1e-12)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--profiles", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contexts = json.loads(args.contexts.read_text(encoding="utf-8"))
    tables = []
    for specification in args.profiles:
        if "=" not in specification:
            parser.error("--profiles must use NAME=PATH")
        source, path = specification.split("=", 1)
        tables.append(
            audit_profiles(
                pd.read_csv(Path(path).resolve()),
                contexts,
                args.controller_root.resolve(),
                source,
            )
        )
    audit = pd.concat(tables, ignore_index=True)
    if not audit.weight_conservation_pass.all():
        raise ValueError("At least one demand assignment changes native physical occupancy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    print(audit.groupby("data_source").agg(
        groups=("weight_conservation_pass", "size"),
        groups_passing=("weight_conservation_pass", "sum"),
        maximum_error=("maximum_hour_zone_weight_error", "max"),
    ).to_string())


if __name__ == "__main__":
    main()
