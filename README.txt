Human thermal demand, building energy and flexibility

Software release 0.2.3.
Install

Use Python 3.11.15 and the locked environment:

    uv sync --frozen --extra notebook
Data

Download the matching study data identified in the article's Data availability
statement and extract it at the repository root, preserving relative paths.
Data_for_Figures/ contains the plotting inputs. Supporting_tables/ contains
supporting numerical results, and Supplementary_tables/ the displayed tables.
Figures and tables

    uv run --frozen python reproduce_publication.py --output-root output/publication

This creates six main figures, four Extended Data figures, six supplementary
figures and ten tables. plot/Plot.ipynb provides one figure per plotting cell.
Times New Roman regular and bold must be installed locally.
Scientific experiments

REPRODUCTION_FROM_RAW.txt gives the original input sources, reference-model
preparation, EnergyPlus installation and execution commands. The scientific
methods are in src/; fixed specifications are in data/metadata/.
Tests

    uv sync --frozen --extra notebook --extra test
    uv run --frozen python -m pytest
Licenses

Source code is BSD-3-Clause. Author-generated data and figures are CC BY 4.0.
Third-party inputs retain their original licenses; see THIRD_PARTY_NOTICES.txt.
