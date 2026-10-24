"""Preserve locally held frozen models and weather without a redistribution grant."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=False)
    models = json.loads((args.stage_root / "engineering/MODEL_CONTEXTS.json").read_text(encoding="utf-8"))
    jobs = json.loads((args.stage_root / "engineering/UNIQUE_ANNUAL_JOBS.json").read_text(encoding="utf-8"))
    runtimes = {row["context"]: row["runtime_sha256"] for row in jobs}
    exported = {}
    for context, cfg in models.items():
        folder = out / context
        folder.mkdir()
        for field, name in (("baseline_idf", "baseline.idf"), ("weather", "weather.epw")):
            source = Path(cfg[field])
            expected = cfg["baseline_idf_sha256" if field == "baseline_idf" else "weather_sha256"]
            if sha256(source) != expected:
                raise ValueError(f"Frozen {field} checksum changed")
            shutil.copyfile(source, folder / name)
        exported[context] = {"baseline_idf": f"{context}/baseline.idf",
                             "baseline_idf_sha256": cfg["baseline_idf_sha256"],
                             "weather": f"{context}/weather.epw", "weather_sha256": cfg["weather_sha256"],
                             "half_deadband_c": cfg["half_deadband_c"],
                             "runtime_sha256": runtimes[context]}
    (out / "MODEL_CONTEXTS.json").write_text(json.dumps(exported, indent=2) + "\n", encoding="utf-8")
    (out / "LOCAL_SOURCE_TERMS.md").write_text(
        "# Locally supplied complete model inputs\n\n"
        "These exact model and weather files are a local preservation of previously supplied "
        "materials. This export does not assert redistribution permission or add an upstream "
        "licence. They remain outside the public code allow-list. No EnergyPlus executable is "
        "copied into this directory. Use the recorded runtime identity and original source terms.\n",
        encoding="utf-8")
    print(json.dumps({"status": "LOCAL_INPUTS_PRESERVED", "complete_models": len(exported),
                      "weather_files": len(exported), "redistribution_permission_asserted": False}))


if __name__ == "__main__":
    main()
