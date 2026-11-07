"""Sequential simulation stages with recorded quality gates and resumable caches."""

from common import *
import subprocess
import time


def command(script, *args):
    label = "_".join([Path(script).stem, *args])
    with (ROOT / "logs" / (label + ".log")).open("a", encoding="utf-8") as log:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parent / script), *args],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write_json(
        ROOT / "audit" / ("process_" + label + ".json"),
        {"returncode": result.returncode, "log": str(ROOT / "logs" / (label + ".log"))},
    )
    print(label, result.returncode, flush=True)
    return result.returncode == 0
