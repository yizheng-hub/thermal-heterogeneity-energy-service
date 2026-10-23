# Data and software licences

## Original project material

| Material | Licence | Scope |
|---|---|---|
| Original source code | BSD-3-Clause | Files authored for this computational release unless a file-level notice says otherwise. |
| Original documentation | CC BY 4.0 | Repository documentation and protocols authored for this release. |
| Original figures and author-generated tables | CC BY 4.0 | Only products authored by the study contributors and cleared for release. |

The BSD licence is in `LICENSE`. The documentation/data notice is in `LICENSE-DOCS-DATA`.

## Third-party exclusions

The project licences do not relicense third-party human datasets, building prototypes, EnergyPlus, weather files, policies, standards, accounting sources or excerpts from those sources. They also do not authorize release of private crosswalks or non-minimized participant-linked derivatives.

The repository uses five rights-aware dispositions:

- `GITHUB_DIRECT`: author-generated material cleared for direct publication after scanning.
- `GITHUB_DERIVED_AGGREGATE_ONLY`: only a minimized aggregate or sanitized derivative may be tracked.
- `ZENODO_DERIVED_MINIMIZED`: a versioned minimized derivative may be deposited after schema and privacy validation.
- `USER_DOWNLOAD_REQUIRED`: retain only source metadata, retrieval instructions and checksums; do not mirror the source.
- `DO_NOT_RELEASE`: exclude from every public package.

`ZENODO_LARGE_AUDIT_OPTIONAL` is an additional non-Git class for a selected, separately reviewed native audit subset. It is not required for Level 1 reproduction.

Every one of the 28 asset classes has an explicit disposition in `data/source_manifest.csv`. A missing or `UNKNOWN` disposition is a hard failure. The R0-R1 scaffold contains no analysis dataset.

The bundled Natural Earth 1:110m land basemap is third-party public-domain material. Its pinned source and checksum are recorded in the package asset provenance and `THIRD_PARTY_NOTICES.md`; the project licences do not change its public-domain status.
