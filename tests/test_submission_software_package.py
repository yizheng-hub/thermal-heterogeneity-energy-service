"""Submission software archives freeze current source bytes and omit local assets."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tomllib
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "submission_packages", ROOT / "publication/build_submission_packages.py"
)
package = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package)


def source_tree(root):
    files = {name: (name + "\n").encode() for name in package.REQUIRED_FILES}
    files.update({
        "README.md": b"# Study software\nuv sync --frozen --extra notebook\nuv run --frozen python reproduce_publication.py\n",
        "REPRODUCTION_FROM_RAW.md": b"# Inputs\nPrepare the model with `src/utilities/prepare_reference_model.py`.\n",
        "pyproject.toml": b'[project]\nreadme = "README.md"\n',
        "src/method.py": b"VALUE = 2\n",
        "src/runtime_config.json": b'{"version":"22.1.0"}\n',
        **{name: b"# package source or test\n" for name in package.PACKAGER_FILES},
        "tools/author_helper.py": b"# required author utility\n",
        "audit/review.md": b"local review\n",
        "tmp/development.log": b"local log\n",
        "src/__pycache__/method.pyc": b"cache",
        "tools/fonts/font.ttf": b"font",
        "tools/EnergyPlus/energyplus.exe": b"runtime",
        "data/upstream/source.csv": b"raw\n",
        "src/.env": b"LOCAL_SETTING=example\n",
        "src/credentials.json": b"{}\n",
        "AGENTS.md": b"local instructions\n",
        "publication/Main_manuscript.md": b"private authoring source\n",
        ".github/workflows/release-ci.yml": b"old release checks\n",
        "src/utilities/release_scan.py": b"# old release checks\n",
        "src/provenance/SCRIPT_MIGRATION_MAP.json": b"{}\n",
        "src/analysis/strengthening/lossless_history_storage.py": b"# scientific runtime dependency\n",
        "src/analysis/support/fixed_reference.csv": b"value\n1\n",
        "Data_for_Figures/figure.csv": b"value\n1\n",
    })
    files[package.ALLOWLIST] = ("\n".join(sorted(
        set(files) - set(package.PACKAGER_FILES)
    )) + "\n").encode()
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return files


def test_software_contains_current_source_licenses_config_and_entry_points(tmp_path):
    root = tmp_path / "source"
    source_tree(root)
    snapshot = package.collect_software_snapshot(root, "submission-1")
    included = {row["file"] for row in snapshot["files"]}
    assert {package.DOCUMENT_ALIASES.get(name, name) for name in package.REQUIRED_FILES}.issubset(included)
    assert set(package.PACKAGER_FILES).issubset(included)
    assert {"src/method.py", "src/runtime_config.json"}.issubset(included)
    assert not any(Path(name).suffix.lower() == ".md" for name in included)
    assert not any(name.startswith(("audit/", "tmp/", ".github/")) for name in included)
    assert not included.intersection({"AGENTS.md", "src/utilities/release_scan.py", "src/provenance/SCRIPT_MIGRATION_MAP.json"})
    assert "tools/author_helper.py" in included
    assert not any(name.startswith("Data_for_Figures/") for name in included)
    assert {"src/analysis/strengthening/lossless_history_storage.py", "src/analysis/support/fixed_reference.csv"}.issubset(included)
    assert not list(tmp_path.glob("*.zip"))
    destination = tmp_path / "software.zip"
    result = package.build_software_package(root, destination, "submission-1")
    assert result["source_snapshot_sha256"] == snapshot["source_snapshot_sha256"]
    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("CODE_MANIFEST.json"))
        assert manifest == snapshot
        assert archive.read("src/method.py") == b"VALUE = 2\n"
        assert set(archive.namelist()) == included | {"CODE_MANIFEST.json"}
        assert tomllib.loads(archive.read("pyproject.toml").decode())["project"]["readme"] == {
            "file": "README.txt", "content-type": "text/plain"
        }
        assert "uv run --frozen python reproduce_publication.py" in archive.read("README.txt").decode()
        allowed = archive.read(package.ALLOWLIST).decode().splitlines()
        assert set(allowed) == included
        assert not any(name.endswith(".md") for name in allowed)
        assert "excluded_paths" not in manifest
        for row in manifest["files"]:
            assert hashlib.sha256(archive.read(row["file"])).hexdigest() == row["sha256"]


def test_snapshot_and_zip_are_deterministic_and_refuse_overwrite(tmp_path):
    root = tmp_path / "source"
    source_tree(root)
    first = package.collect_software_snapshot(root, "submission-1")
    assert package.collect_software_snapshot(root, "submission-1") == first
    left, right = tmp_path / "one.zip", tmp_path / "two.zip"
    package.build_software_package(root, left, "submission-1")
    package.build_software_package(root, right, "submission-1")
    assert left.read_bytes() == right.read_bytes()
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(left) as archive:
        archive.extractall(extracted)
    assert package.collect_software_snapshot(extracted, "submission-1")["source_snapshot_sha256"] == first["source_snapshot_sha256"]
    with pytest.raises(FileExistsError):
        package.build_software_package(root, left, "submission-1")
    (root / "src/method.py").write_bytes(b"VALUE = 3\n")
    assert package.collect_software_snapshot(root, "submission-1")["source_snapshot_sha256"] != first["source_snapshot_sha256"]
    stale_destination = tmp_path / "stale.zip"
    with pytest.raises(ValueError, match="Source changed"):
        package.build_software_package(root, stale_destination, "submission-1", source_snapshot=first)
    assert not stale_destination.exists()


def test_archive_keeps_scientific_asset_tests_but_removes_old_scan_test():
    payload = (
        "@pytest.mark.parametrize('x', [1, 2])\n"
        "def test_demographic_summary_disclosure_threshold(x):\n    pass\n\n"
        "def test_source_building_input_verification():\n    assert True\n"
    ).encode()
    result = package._archive_payload("tests/contracts/test_paths.py", payload).decode()
    assert "test_demographic_summary_disclosure_threshold" not in result
    assert "pytest.mark.parametrize" not in result
    assert "test_source_building_input_verification" in result


def test_plain_documents_are_readable_and_originals_are_unchanged(tmp_path):
    root = tmp_path / "source"
    originals = source_tree(root)
    originals["THIRD_PARTY_NOTICES.md"] = b"## Sources\nSee `data/source_manifest.csv`.\n"
    (root / "THIRD_PARTY_NOTICES.md").write_bytes(originals["THIRD_PARTY_NOTICES.md"])
    documents = package.documentation_payloads(root, "0.2.1")
    assert set(documents) == set(package.DOCUMENT_ALIASES.values())
    assert all(not name.endswith(".md") for name in documents)
    readme = documents["README.txt"].decode()
    assert "uv sync --frozen --extra notebook" in readme
    assert "uv run --frozen python reproduce_publication.py" in readme
    raw = documents["REPRODUCTION_FROM_RAW.txt"].decode()
    assert raw == "Inputs\nPrepare the model with src/utilities/prepare_reference_model.py.\n"
    assert documents["THIRD_PARTY_NOTICES.txt"].decode() == "Sources\nSee data/metadata/source_manifest.csv.\n"
    for name in package.DOCUMENT_ALIASES:
        assert (root / name).read_bytes() == originals[name]


def test_source_paths_cannot_escape_the_project(tmp_path):
    root = tmp_path / "source"
    source_tree(root)
    with pytest.raises(ValueError, match="project-relative"):
        package.collect_software_snapshot(root, "submission-1", ["../outside.py"])
    with pytest.raises(ValueError, match="Missing source"):
        package.collect_software_snapshot(root, "submission-1", ["src/absent.py"])


def data_writer(payloads, *, corrupt_checksum=False, wrong_version=False):
    def write(destination, *, release_id, source_snapshot):
        assert isinstance(destination, io.BytesIO)
        manifest = {
            "release_id": "different" if wrong_version else release_id,
            "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
            "files": [{"file": name, "sha256": hashlib.sha256(payload).hexdigest()}
                      for name, payload in payloads.items()],
        }
        if corrupt_checksum:
            manifest["files"][0]["sha256"] = "0" * 64
        with zipfile.ZipFile(destination, "w") as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
            archive.writestr("SOURCE_DATA_MANIFEST.json", json.dumps(manifest))
        return manifest
    return write


def test_combined_archive_merges_direct_members_and_shared_hashes(tmp_path, monkeypatch):
    from publication import build_review_package

    root = tmp_path / "source"
    source_tree(root)
    snapshot = package.collect_software_snapshot(root, "0.2.2")
    data = {
        "Data_for_Figures/input.csv": b"value\n1\n",
        "Data_for_Figures/native.npz": b"native scientific NPZ bytes",
        "LICENSE": (root / "LICENSE").read_bytes(),
    }
    monkeypatch.setattr(build_review_package, "build", data_writer(data))
    destination = tmp_path / "Code_and_Data.zip"
    result = package.build_combined_package(root, destination, "0.2.2", source_snapshot=snapshot)
    assert result["source_snapshot_sha256"] == snapshot["source_snapshot_sha256"]
    assert result["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    with zipfile.ZipFile(destination) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert not any(name.lower().endswith((".zip", ".7z", ".tar", ".tgz")) for name in names)
        assert archive.read("Data_for_Figures/native.npz") == data["Data_for_Figures/native.npz"]
        code = json.loads(archive.read("CODE_MANIFEST.json"))
        source_data = json.loads(archive.read("SOURCE_DATA_MANIFEST.json"))
        assert code["release_id"] == source_data["release_id"] == "0.2.2"
        assert code["source_snapshot_sha256"] == source_data["source_snapshot_sha256"]
        for manifest in (code, source_data):
            for row in manifest["files"]:
                assert hashlib.sha256(archive.read(row["file"])).hexdigest() == row["sha256"]
    with pytest.raises(FileExistsError):
        package.build_combined_package(root, destination, "0.2.2")


@pytest.mark.parametrize("name", ["inner.zip", "inner.7z", "inner.tar", "inner.tgz", "inner.tar.gz"])
def test_combined_archive_rejects_nested_archives_before_writing(tmp_path, monkeypatch, name):
    from publication import build_review_package

    root = tmp_path / "source"
    source_tree(root)
    monkeypatch.setattr(build_review_package, "build", data_writer({name: b"archive"}))
    destination = tmp_path / "Code_and_Data.zip"
    with pytest.raises(ValueError, match="Nested archives"):
        package.build_combined_package(root, destination, "0.2.2")
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["duplicate", "checksum", "version"])
def test_combined_archive_rejects_conflicts_or_bad_data_manifest(tmp_path, monkeypatch, mode):
    from publication import build_review_package

    root = tmp_path / "source"
    source_tree(root)
    writer = data_writer(
        {"src/method.py": b"different from the supplied code"},
        corrupt_checksum=mode == "checksum",
        wrong_version=mode == "version",
    )
    monkeypatch.setattr(build_review_package, "build", writer)
    destination = tmp_path / "Code_and_Data.zip"
    with pytest.raises(ValueError, match={
        "duplicate": "Different payloads",
        "checksum": "checksum mismatch",
        "version": "versions differ",
    }[mode]):
        package.build_combined_package(root, destination, "0.2.2")
    assert not destination.exists()


def test_combined_cli_uses_combined_builder(tmp_path, monkeypatch, capsys):
    destination = tmp_path / "Code_and_Data.zip"
    calls = []
    monkeypatch.setattr(package, "build_combined_package", lambda *args: calls.append(args) or {"release_id": args[2]})
    monkeypatch.setattr(sys, "argv", ["packager", "combined", "--destination", str(destination), "--release-id", "0.2.2"])
    package.main()
    assert calls == [(ROOT, destination, "0.2.2")]
    assert json.loads(capsys.readouterr().out)["release_id"] == "0.2.2"
