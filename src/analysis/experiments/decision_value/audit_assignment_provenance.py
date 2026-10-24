"""Audit sampled composition provenance without interpreting it as field placement."""

import argparse
import json
from pathlib import Path

import pandas as pd

project = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=project / "clean_data")
parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
human = pd.read_csv(args.data_root / "human/human_profiles_minimized.csv.gz")
assign = pd.read_csv(args.data_root / "population/population_assignments_minimized.csv.gz")
joined = assign.merge(
    human[
        ["release_profile_id", "release_unit_id", "release_building_group_id", "country", "city"]
    ],
    left_on="release_archive_profile_id",
    right_on="release_profile_id",
    validate="many_to_one",
    suffixes=("", "_archive"),
)
assert len(joined) == len(assign)
rows = []
for (scenario, seed), group in joined.groupby(["scenario_id", "composition_seed"], sort=True):
    local = group.drop_duplicates("release_unit_id")
    rows.append(
        dict(
            scenario_id=scenario,
            composition_seed=int(seed),
            units=len(local),
            observed_source_buildings=local.release_building_group_id.nunique(),
            observed_source_cities=local.city.nunique(),
            largest_source_building_count=int(
                local.groupby("release_building_group_id").size().max()
            ),
            contexts=group.context.nunique(),
        )
    )
pd.DataFrame(rows).to_csv(args.output_root / "ASSIGNMENT_PROVENANCE.csv", index=False)
overlap = []
for heldout in sorted(joined.scenario_id.unique()):
    test = set(joined.loc[joined.scenario_id.eq(heldout), "release_unit_id"])
    train = set(joined.loc[~joined.scenario_id.eq(heldout), "release_unit_id"])
    overlap.append(
        dict(
            heldout_population=heldout,
            heldout_unique_units=len(test),
            training_unique_units=len(train),
            units_shared_between_train_and_test=len(test & train),
        )
    )
pd.DataFrame(overlap).to_csv(args.output_root / "HUMAN_HOLDOUT_OVERLAP.csv", index=False)
result = dict(
    assignment_rows=len(joined),
    unique_sampled_archive_units=joined.release_unit_id.nunique(),
    all_compositions_have_267_units=all(row["units"] == 267 for row in rows),
    source_buildings_min=min(row["observed_source_buildings"] for row in rows),
    source_buildings_max=max(row["observed_source_buildings"] for row in rows),
    compositions_are_observed_building_preserved=False,
    context_seed_folds_are_person_disjoint=False,
    population_holdout=overlap,
)
(args.output_root / "ASSIGNMENT_AUDIT.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
print(json.dumps(result, indent=2))
