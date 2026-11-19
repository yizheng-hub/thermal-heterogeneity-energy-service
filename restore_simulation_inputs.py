"""Restore the study's 28 scientific inputs after extracting Simulation_Inputs.zip."""
import argparse
import csv
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parent
DOE_ARCHIVE = "ASHRAE901_OfficeMedium_STD2019.zip"
DOE_SHA256 = "8431a33c3c27c0c34b1efc31be5f558b2b17e29a8850accc1bbce5f4f0dc5f7c"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def source_bytes(url, filename, source_dir):
    local = source_dir / filename if source_dir else None
    if local and local.is_file():
        return local.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Study input restoration"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(
            f"Could not download {filename} from {url}. Download it from the "
            f"listed provider, save it as {filename}, and rerun with --source-dir DIRECTORY."
        ) from error


def write_verified(path, content, expected):
    if digest(content) != expected:
        raise ValueError(f"Source bytes do not match the study input: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        stream.write(content)
    try:
        if path.exists():
            raise FileExistsError(path)
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def restore(source_dir):
    data = ROOT / "data"
    with (data / "SIMULATION_INPUT_SOURCES.csv").open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    archive = None
    for row in sorted(records, key=lambda r: r["delivery"] == "generate"):
        path = (ROOT / row["file"]).resolve()
        if not path.is_relative_to(data.resolve()):
            raise ValueError(f"Invalid input path: {row['file']}")
        expected = row["sha256"]
        if path.is_file():
            if digest(path.read_bytes()) != expected:
                raise ValueError(f"Existing input has a different checksum: {path}")
            continue
        method = row["delivery"]
        if method == "included":
            raise FileNotFoundError(f"Extract Simulation_Inputs.zip at the repository root: {path}")
        if method == "generate":
            subprocess.run(
                [sys.executable, str(ROOT / "src/utilities/prepare_reference_model.py"),
                 "--source-idf", str(data / "upstream/global_climate_matrix/prototypes_22_1/Miami.idf"),
                 "--output", str(path)], check=True,
            )
            if digest(path.read_bytes()) != expected:
                raise ValueError("Generated reference checksum differs from the study reference")
            continue
        if method == "archive member":
            if archive is None:
                content = source_bytes(row["source_url"], DOE_ARCHIVE, source_dir)
                if digest(content) != DOE_SHA256:
                    raise ValueError("Obtain the 22.1 DOE OfficeMedium STD2019 archive specified in the manifest.")
                archive = zipfile.ZipFile(io.BytesIO(content))
            content = archive.read(row["member"])
        elif method == "download":
            content = source_bytes(row["source_url"], path.name, source_dir)
        else:
            raise ValueError(f"Unknown delivery method: {method}")
        write_verified(path, content, expected)
    if archive is not None:
        archive.close()
    print(f"Verified {len(records)} scientific inputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Folder containing previously downloaded provider files.")
    restore(parser.parse_args().source_dir)
