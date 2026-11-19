Human thermal demand, building energy and flexibility

Install

Use Python 3.11.15 and the locked environment:

    uv sync --frozen --extra notebook
Data

Clone the public code repository. The two data archives described below are
scheduled for release on figshare upon publication.

    git clone https://github.com/yizheng-hub/thermal-heterogeneity-energy-service.git
    cd thermal-heterogeneity-energy-service

- Simulation_Inputs.zip: observations, numerical inputs and a per-file source
  manifest for new simulations.
- Numerical_Results.zip: figure inputs and complete numerical results under
  data/figure_data/ and data/analysis_results/. data/RESULTS_INDEX.csv
  identifies the additional result datasets.

Plotting code is in plot/Plot.ipynb in this repository.

Place the ZIP files beside this README and pyproject.toml. In PowerShell,
extract them into the repository root, merging the supplied data/ directory:

    Expand-Archive -LiteralPath .\Simulation_Inputs.zip -DestinationPath . -Force
    Expand-Archive -LiteralPath .\Numerical_Results.zip -DestinationPath . -Force

For simulations, restore the provider-hosted inputs and prepare the reference:
    uv run --frozen python restore_simulation_inputs.py

The script verifies all 28 scientific inputs. The simulation archive includes
direct download locations and the exact DOE archive and model members.
REPRODUCTION_FROM_RAW.txt describes how to supply local copies.

Numerical_Results.zip alone supports the figure and table workflow.
Figures and tables

    uv run --frozen python reproduce_publication.py --output-root output/publication

This creates six main figures, four Extended Data figures, seven supplementary
figures and twelve table summaries. Open output/publication/main/ for Figures
1-6, output/publication/extended/ for Extended Data Figures 1-4, and
output/publication/supplementary/ for Supplementary Figures 1-7. Each
figure is available as PNG, editable SVG and PDF. The tables are in
output/publication/tables/. plot/Plot.ipynb has
17 independent figure cells containing data loading, style, layout and drawing
code. The population and command-domain sensitivity plotting function is also
available in src/thermal_heterogeneity/methods_sensitivity_figure.py. The
command-line runner executes the notebook cells.
Times New Roman regular and bold must be installed locally.
Scientific experiments

REPRODUCTION_FROM_RAW.txt gives the input locations, EnergyPlus installation
and experiment commands. The restoration step prepares the reference model. The scientific
methods are in src/; fixed specifications are in data/metadata/.

The complete workflow includes the service-limit sensitivity analysis. To
recalculate it from a completed run in output/new_run:

    uv run --frozen python src/analysis/strengthening/service_threshold_sensitivity.py --results-root output/new_run/strengthening/results --matrix-root output/new_run/matrix --memberships output/new_run/strengthening/results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv --trajectory-root output/new_run --output-dir output/new_run/strengthening/results/service_threshold_sensitivity

This analysis uses the completed operation library and retained temperature
trajectories. It writes paired service scores, reference limits, selections
at each service-limit factor, and their summary without running EnergyPlus.

Population and command-domain sensitivity use the same completed run:

    uv run --frozen python src/analysis/strengthening/population_command_sensitivity.py --demand-root output/new_run/demand --matrix-root output/new_run/matrix --results-root output/new_run/strengthening/results --output-dir output/new_run/strengthening/results/population_command_sensitivity
    uv run --frozen python src/analysis/strengthening/command_domain_sensitivity.py --matrix-root output/new_run/matrix --results-root output/new_run/strengthening/results --output-dir output/new_run/strengthening/results/command_domain_sensitivity

The population analysis samples 30 additional 267-profile compositions from
each source and scores all 31 shared commands in each weather context. The
domain analysis retains final-library operations whose zone command midpoints
all lie within 21–26 °C. Both analyses use retained temperature trajectories
and are included in the complete workflow.

To draw Fig. S3 directly from those analysis results:

    uv run --frozen python src/thermal_heterogeneity/methods_sensitivity_figure.py --analysis-root output/new_run/strengthening/results --output-root output/new_run/publication/supplementary
Tests

    uv sync --frozen --extra notebook --extra test
    uv run --frozen python -m pytest
Licenses

Source code is BSD-3-Clause. Author-generated data and figures are CC BY 4.0.
Third-party inputs retain their original licenses; see THIRD_PARTY_NOTICES.txt.
