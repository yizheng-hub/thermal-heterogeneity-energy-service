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
  results/figure_data/ and results/analysis_results/. results/RESULTS_INDEX.csv
  identifies the additional result datasets.

Plotting code is in plot/Plot.ipynb in this repository.

Place the ZIP files beside this README and pyproject.toml. In PowerShell,
extract them beside this README:

    Expand-Archive -LiteralPath .\Simulation_Inputs.zip -DestinationPath . -Force
    Expand-Archive -LiteralPath .\Numerical_Results.zip -DestinationPath . -Force

For simulations, restore the provider-hosted inputs and prepare the reference:
    uv run --frozen python restore_simulation_inputs.py

The script verifies all 28 scientific inputs. The simulation archive includes
direct download locations and the exact DOE archive and model members.
REPRODUCTION_FROM_RAW.txt describes how to supply local copies.

Numerical_Results.zip alone supports the figure and table workflow.

Results layout

results/analysis_results/ contains the supporting numerical analyses.
results/figure_data/ contains the indexed plotting inputs: Tables/ for numerical
values, Maps/ for geometry and locations, and Hourly/ for the illustrated
event trajectories. Each cell of plot/Plot.ipynb reads the files listed in
results/figure_data/index.json and writes one figure under plot/.

Figure S1e,f uses half-degree counts of estimated neutral temperatures in CTCD
office records collected from 1 June 2007 onward. Each season has its own
setpoint limit in data/metadata/policy_reference_documents.csv. Counts and
seasonal summaries are indexed as CTCD_SEASONAL_SETPOINT_BINS.csv and
CTCD_SEASONAL_SETPOINT_SUMMARY.csv.
In a checkout that also contains results/simulation_run/, that index can point
to identical tables and hourly trajectories at their original simulation paths.
results/RESULTS_INDEX.csv also points to shared tables used in more than one analysis.

Figures and tables

    uv run --frozen python reproduce_publication.py

This creates six main figures, four Extended Data figures, seven supplementary
figures and twelve table summaries. Open plot/main/ for Figures
1-6, plot/extended/ for Extended Data Figures 1-4, and
plot/supplementary/ for Supplementary Figures 1-7. Each
figure is available as PNG, editable SVG and PDF. Main-figure panel CSVs sit beside their figures. Each figure JSON lists the
input files and their SHA-256 checksums. The tables are in
plot/tables/. plot/Plot.ipynb has
17 independent figure cells containing data loading, style, layout and drawing
code. The population and command-domain sensitivity plotting function is also
available in src/thermal_heterogeneity/methods_sensitivity_figure.py. The
command-line runner executes the notebook cells.
Times New Roman regular and bold must be installed locally.

After running the simulations and analyses in results/new_run, draw the same
figures and tables directly from those results:

    uv run --frozen python reproduce_publication.py --raw-run-root results/new_run

Both input routes write into the same plot/ folders. Use --figure 5 E2 S3 to
redraw selected figures or --output-root DIRECTORY for another destination.

Scientific experiments

REPRODUCTION_FROM_RAW.txt gives the input locations, EnergyPlus installation
and experiment commands. The restoration step prepares the reference model. The scientific
methods are in src/; fixed specifications are in data/metadata/.

The complete workflow includes the service-limit sensitivity analysis. To
recalculate it from a completed run in results/new_run:

    uv run --frozen python src/analysis/strengthening/service_threshold_sensitivity.py --results-root results/new_run/strengthening/results --matrix-root results/new_run/matrix --memberships results/new_run/strengthening/results/base/annual/GLOBAL_FIXED_TAIL_MEMBERSHIPS.csv --trajectory-root results/new_run --output-dir results/new_run/strengthening/results/service_threshold_sensitivity

This analysis uses the completed operation library and retained temperature
trajectories. It writes paired service scores, reference limits, selections
at each service-limit factor, and their summary without running EnergyPlus.

Population and command-domain sensitivity use the same completed run:

    uv run --frozen python src/analysis/strengthening/population_command_sensitivity.py --demand-root results/new_run/demand --matrix-root results/new_run/matrix --results-root results/new_run/strengthening/results --output-dir results/new_run/strengthening/results/population_command_sensitivity
    uv run --frozen python src/analysis/strengthening/command_domain_sensitivity.py --matrix-root results/new_run/matrix --results-root results/new_run/strengthening/results --output-dir results/new_run/strengthening/results/command_domain_sensitivity

The population analysis samples 30 additional 267-profile compositions from
each source and scores all 31 shared commands in each weather context. The
domain analysis retains final-library operations whose zone command midpoints
all lie within 21–26 °C. Both analyses use retained temperature trajectories
and are included in the complete workflow.

To draw Fig. S3 directly from those analysis results:

    uv run --frozen python src/thermal_heterogeneity/methods_sensitivity_figure.py --analysis-root results/new_run/strengthening/results
Tests

    uv sync --frozen --extra notebook --extra test
    uv run --frozen python -m pytest
Licenses

Source code is BSD-3-Clause. Author-generated data and figures are CC BY 4.0.
Third-party inputs retain their original licenses; see THIRD_PARTY_NOTICES.txt.
