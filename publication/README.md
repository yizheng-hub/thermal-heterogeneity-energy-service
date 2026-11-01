# Publication reproduction

The publication entry point is `plot/Plot.ipynb`: one setup cell followed by
one complete figure per cell (Figures 1–6 and Supplementary Figures S1–S16).
All numerical and geographic inputs for these cells are in `Result/`.
`Result/FIGURE_DATA_MANIFEST.json` records their source and exported checksums.
These data are supplied in **Supplementary Data 1**, separately from Git.

Unpack Supplementary Data 1 into the code root, preserving its `Result/`
directory. Install the pinned environment with `uv sync --frozen --extra notebook`.
Times New Roman regular and bold fonts must be installed or supplied under
`tools/fonts/` under a suitable local license; fonts are not redistributed in
the code attachment. Then execute the notebook from the project root or `plot/`.
Alternatively, run:

```text
python reproduce_publication.py --plan
python reproduce_publication.py --output-root output/publication
```

This executes the notebook cells and generates six main figures, sixteen
supplementary figures and nine tables. Each figure is exported as PNG, PDF and
editable-text SVG with a source-hash and layout receipt. This command does not
run EnergyPlus. The manuscript uses the same figure and table inputs.

For an author copy containing `publication/Main_manuscript.md`,
`publication/Supplementary_information.md` and the retained Word templates,
`--documents` also builds the two Word documents. This optional step needs
python-docx 1.2.0 and lxml 6.1.1. Word rendering requires Microsoft Word on
Windows and uses `publication/render_with_word.ps1`; rendering is separate from
building the documents. Author manuscripts, templates and licensed runtimes
are retained locally, not in the source-only Git repository.

To export new figure inputs after a scientific reanalysis, first generate the
figures and tables from the scientific output tree, then run
`python publication/figure_data.py --publication-root output/publication`.
Export rejects changed source hashes and does not change scientific values.
It exports only anonymous plotted demand quantities and factual floor vertices, excluding the full third-party model. It makes illustrated hourly-trajectory paths relative to
`Result/`; their manifest records these transformations.

## Full scientific execution

`REPRODUCTION_FROM_RAW.md`, `reproduce_from_raw.py`,
`reproduce_secondary_analyses.py` and `reproduce_strengthening.py` describe the
raw-data, annual-simulation and additional-analysis workflows. Their scientific
inputs and EnergyPlus runtime must be present before execution. The local
independent project contains them. A source-only Git clone plus the figure data
attachment is sufficient for figure reproduction, not a substitute for all
hourly simulation inputs and outputs.

The original raw-data reconstruction and 810 logical annual operations have
been reproduced in a fresh output tree. Additional branches have individual
execution and validation receipts. A single empty-directory rerun of **all**
additional physical branches has not been performed; it is deferred for the
next computing environment. The release records this status explicitly.

## Review attachment export

The frozen supporting-table inventory is `publication/review_data_files.txt`.
After verification and a source commit, run:

```text
python -m publication.build_review_package --destination output/submission/Supplementary_Data_1.zip --code-revision <full commit>
```

The exporter checks figure-input hashes, removes unused machine-path columns,
rejects person identifiers in supporting tables, and validates every archive member.
A Git archive of that exact commit is Supplementary Code 1.
