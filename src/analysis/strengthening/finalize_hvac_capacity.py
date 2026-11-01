from common import *
from hvac_runs import freeze_hardware
import pandas as pd
import time


def main():
    log = ROOT / "logs/hvac_design_v2.log"
    while not log.exists() or log.read_text(errors="replace").count("HVAC_DESIGN") < 6:
        time.sleep(5)
    records = json.loads((ROOT / "results/HVAC_DESIGN_RESULTS.json").read_text())
    for row in records:
        if not row.get("completed"):
            continue
        text, values, missing = freeze_hardware(row["idf"], row.get("capacity_sql", row["sql"]))
        path = Path(row["fixed_capacity_model"])
        path.write_text(text)
        pd.DataFrame(values).to_csv(path.parent / "FROZEN_CAPACITIES.csv", index=False)
        write_json(path.parent / "UNRESOLVED_CAPACITIES.json", missing)
        row.update(
            {
                "fixed_hardware_fields": len(values),
                "unresolved_hardware_fields": len(missing),
                "capacity_freeze_pass": not missing,
                "fixed_capacity_sha256": digest(path),
            }
        )
    write_json(ROOT / "results/HVAC_DESIGN_RESULTS.json", records)
    record_status(
        "hvac_design",
        "completed" if all(x.get("capacity_freeze_pass") for x in records) else "partial",
        records,
    )
    print(
        [(x.get("system"), x.get("climate"), x.get("unresolved_hardware_fields")) for x in records],
        flush=True,
    )


if __name__ == "__main__":
    main()
