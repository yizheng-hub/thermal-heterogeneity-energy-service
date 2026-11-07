Figures and tables

Use the project environment and extract the study data at the repository root:

    uv sync --frozen --extra notebook
    uv run --frozen python reproduce_publication.py --output-root output/publication

plot/Plot.ipynb has sixteen complete, independently executable figure cells.
All data reading, plotting styles and panel code are visible in each cell.
The outputs are six main figures, four Extended Data figures, six supplementary
figures and ten tables. Extract Numerical_Results.zip at the repository root.
Inputs are in Data/;
Data/index.json and RESULTS_INDEX.csv identify the figure and analysis datasets.
Times New Roman regular and bold must be installed locally.
