Human thermal demand, building energy and flexibility

Software release 0.2.4.
Install

Use Python 3.11.15 and the locked environment:

    uv sync --frozen --extra notebook
Data

Download the two archives identified in the article's data deposit and extract
them at the repository root, retaining relative paths:

- Simulation_Inputs.zip: original observations, weather and building models
  under data/, for new simulations.
- Numerical_Results.zip: figure inputs and complete numerical results under
  Data/. RESULTS_INDEX.csv identifies the additional result datasets.

The article's Supplementary Software 1 separately supplies a standalone plotting
notebook, figure data and all high-resolution PNG figures.
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
