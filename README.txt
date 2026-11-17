Human thermal demand, building energy and flexibility

Install

Use Python 3.11.15 and the locked environment:

    uv sync --frozen --extra notebook
Data

Clone the code repository, then download the data from the associated Figshare
dataset. Repository access is needed while the code is private.

    git clone https://github.com/yizheng-hub/thermal-heterogeneity-energy-service.git
    cd thermal-heterogeneity-energy-service

- Simulation_Inputs.zip: original observations, weather and building models
  under data/, for new simulations.
- Numerical_Results.zip: figure inputs and complete numerical results under
  data/figure_data/ and data/analysis_results/. data/RESULTS_INDEX.csv
  identifies the additional result datasets.

Plotting code is in plot/Plot.ipynb in this repository.

Place the ZIP files beside this README and pyproject.toml. In PowerShell,
extract them into the repository root, merging the supplied data/ directory:

    Expand-Archive -LiteralPath .\Simulation_Inputs.zip -DestinationPath . -Force
    Expand-Archive -LiteralPath .\Numerical_Results.zip -DestinationPath . -Force

Do not place the ZIPs inside data/ or create an extra enclosing directory.
After extraction, these paths should exist:

    data/upstream/frozen_buildings/Miami/baseline.idf
    data/figure_data/index.json
    data/RESULTS_INDEX.csv

Numerical_Results.zip alone is sufficient to redraw the figures and tables;
this does not require EnergyPlus. Simulation_Inputs.zip supplies the original
inputs for new simulations; the numerical results are not required to start
that workflow. Download both to run simulations and compare their results.
Figures and tables

    uv run --frozen python reproduce_publication.py --output-root output/publication

This creates six main figures, four Extended Data figures, six supplementary
figures and twelve table summaries in output/publication/. plot/Plot.ipynb has
16 independent figure cells. Each includes its complete data loading, style,
layout and drawing code. Edit the notebook directly;
the command-line runner executes these cells and does not regenerate them
from Python plotting modules.
Times New Roman regular and bold must be installed locally.
Scientific experiments

REPRODUCTION_FROM_RAW.txt gives the input locations, EnergyPlus installation
and experiment commands. The supplied reference model is ready to use. The scientific
methods are in src/; fixed specifications are in data/metadata/.

The complete workflow includes the service-limit sensitivity analysis. To
recalculate it from a completed run in output/new_run:

    uv run --frozen python src/analysis/strengthening/service_threshold_sensitivity.py --results-root output/new_run/strengthening/results --matrix-root output/new_run/matrix --memberships output/new_run/strengthening/results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv --trajectory-root output/new_run --output-dir output/new_run/strengthening/results/service_threshold_sensitivity

This analysis uses the completed operation library and retained temperature
trajectories. It writes paired service scores, reference limits, selections
at each service-limit factor, and their summary without running EnergyPlus.
Tests

    uv sync --frozen --extra notebook --extra test
    uv run --frozen python -m pytest
Licenses

Source code is BSD-3-Clause. Author-generated data and figures are CC BY 4.0.
Third-party inputs retain their original licenses; see THIRD_PARTY_NOTICES.txt.
