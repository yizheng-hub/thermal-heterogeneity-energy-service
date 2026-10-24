#!/usr/bin/env python3
"""Fail-closed structural check for the public release tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


ALLOWLIST_PATH = "src/provenance/PUBLIC_FILE_ALLOWLIST.txt"
LOCAL_GENERATED_TOP_LEVEL = {"output", "results", "intermediate_files"}
LOCAL_ASSET_DIRECTORIES = {
    "data/inputs", "data/typical_days", "data/upstream", "data/engineering_qa",
    "plot/main", "plot/supplementary",
}
ALLOWED_EMPTY_TOP_LEVEL = {".github"}
LOCAL_METADATA_TOP_LEVEL = {".git"}

FORBIDDEN_DIRECTORY_NAMES = {
    ".runtime",
    "external",
    "private",
    "raw",
    "results",
    "source_snapshots",
    "drafts",
    "templates",
}

IGNORED_LOCAL_CACHE_NAMES = {
    ".cache",
    ".pytest_cache",
    ".pytest_tmp",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
}

FORBIDDEN_SUFFIXES = {
    ".7z",
    ".docx",
    ".epw",
    ".eso",
    ".idf",
    ".pdf",
    ".sql",
    ".zip",
    ".zst",
    ".png", ".svg", ".jpg", ".jpeg", ".npy", ".npz", ".gz",
}
FORBIDDEN_MANUSCRIPT_NAMES = {"MAIN_MANUSCRIPT.md", "SUPPLEMENTARY_INFORMATION.md"}


def _load_allowlist(root: Path) -> tuple[set[str], list[str]]:
    path = root / ALLOWLIST_PATH
    if not path.is_file():
        return set(), [f"public allow-list missing: {ALLOWLIST_PATH}"]
    entries: list[str] = []
    errors: list[str] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        logical = PurePosixPath(value)
        if (
            "\\" in value
            or logical.is_absolute()
            or any(part in {"", ".", ".."} for part in logical.parts)
        ):
            errors.append(f"invalid public allow-list path on line {number}: {value}")
        entries.append(value)
    if len(entries) != len(set(entries)):
        errors.append("public allow-list contains duplicate paths")
    if entries != sorted(entries, key=str.casefold):
        errors.append("public allow-list paths must be sorted case-insensitively")
    if ALLOWLIST_PATH not in entries:
        errors.append("public allow-list must include itself")
    return set(entries), errors


def _is_ignored_local(relative: Path) -> bool:
    if not relative.parts:
        return False
    for prefix in LOCAL_ASSET_DIRECTORIES:
        if relative.as_posix() == prefix or relative.as_posix().startswith(prefix + "/"):
            return relative.as_posix() != prefix + "/README.md"
    if relative.parts[0] in LOCAL_GENERATED_TOP_LEVEL:
        return relative.as_posix() not in {
            relative.parts[0] + "/README.md", relative.parts[0] + "/.gitkeep"}
    if relative.parts[0] in LOCAL_METADATA_TOP_LEVEL:
        return True
    return bool(
        set(relative.parts) & IGNORED_LOCAL_CACHE_NAMES
        or any(part.endswith(".egg-info") for part in relative.parts)
    )


def run_checks(root: Path) -> dict[str, object]:
    root = root.resolve()
    errors: list[str] = []
    warnings: list[str] = []

    if not root.is_dir():
        return {"status": "FAIL", "errors": ["release root is not a directory"]}

    allowlist, allowlist_errors = _load_allowlist(root)
    errors.extend(allowlist_errors)
    for path in sorted(allowlist):
        if _is_ignored_local(Path(path)):
            errors.append(f"local-only file must not be allow-listed: {path}")
    allowed_top = {PurePosixPath(path).parts[0] for path in allowlist}
    present_top = {p.name for p in root.iterdir()}
    unexpected = sorted(
        present_top
        - allowed_top
        - ALLOWED_EMPTY_TOP_LEVEL
        - LOCAL_METADATA_TOP_LEVEL
        - LOCAL_GENERATED_TOP_LEVEL
        - IGNORED_LOCAL_CACHE_NAMES
    )
    if unexpected:
        errors.append(f"unexpected top-level entries: {unexpected}")

    missing = sorted(path for path in allowlist if not (root / path).is_file())
    if missing:
        errors.append(f"allow-listed public files missing: {missing}")

    cache_dirs_seen: set[str] = set()
    public_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if _is_ignored_local(relative):
            cache_dirs_seen.update(set(relative.parts) & IGNORED_LOCAL_CACHE_NAMES)
            cache_dirs_seen.update(part for part in relative.parts if part.endswith(".egg-info"))
            continue
        if path.is_symlink():
            errors.append(f"symbolic link is not allow-listed: {relative.as_posix()}")
        if path.is_dir() and path.name.lower() in FORBIDDEN_DIRECTORY_NAMES:
            errors.append(f"forbidden directory: {relative.as_posix()}")
        if path.is_file():
            public_files.add(relative.as_posix())
            if path.name in FORBIDDEN_MANUSCRIPT_NAMES:
                errors.append(f"forbidden manuscript: {relative.as_posix()}")
            if path.suffix.lower() in FORBIDDEN_SUFFIXES:
                errors.append(f"forbidden file type: {relative.as_posix()}")
            if path.stat().st_size > 5 * 1024 * 1024:
                errors.append(f"public file exceeds Git size ceiling: {relative.as_posix()}")

    extra_public = sorted(public_files - allowlist)
    if extra_public:
        errors.append(f"public files are not allow-listed: {extra_public}")

    if cache_dirs_seen:
        warnings.append(
            "ignored local build/cache directories are present but denied by .gitignore: "
            + ", ".join(sorted(cache_dirs_seen))
        )

    gitignore = root / ".gitignore"
    if gitignore.is_file():
        text = gitignore.read_text(encoding="utf-8")
        first_rule = next(
            (
                line.strip()
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ),
            "",
        )
        if first_rule != "/*":
            errors.append(".gitignore must begin with the deny-by-default rule /*")
        gitignore_lines = {line.strip() for line in text.splitlines() if line.strip()}
        missing_rules = sorted(path for path in allowlist if f"!/{path}" not in gitignore_lines)
        if missing_rules:
            errors.append(f".gitignore does not unignore allow-listed files: {missing_rules}")
        for required_pattern in (
            "**/raw/**",
            "**/*.epw",
            "**/*.docx",
            "**/*.zst",
            "**/.runtime/**",
            "**/*.egg-info/**",
        ):
            if required_pattern not in text:
                errors.append(f".gitignore missing fail-closed pattern: {required_pattern}")
        for path in LOCAL_ASSET_DIRECTORIES | LOCAL_GENERATED_TOP_LEVEL:
            if f"!/{path}/**" in gitignore_lines:
                errors.append(f"local-only directory must remain ignored: {path}")

    if (root / ".git").exists():
        warnings.append("Git metadata exists; structural checks ignore its internal files.")

    return {
        "status": "PASS" if not errors else "FAIL",
        "check": "release_layout",
        "allowlisted_file_count": len(allowlist),
        "observed_public_file_count": len(public_files),
        "git_initialized": (root / ".git").exists(),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_checks(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
