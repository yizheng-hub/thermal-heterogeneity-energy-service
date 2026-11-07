Figures and tables

Use the project environment and extract the study data at the repository root:

    uv sync --frozen --extra notebook
    uv run --frozen python reproduce_publication.py --output-root output/publication

plot/Plot.ipynb has one setup cell and sixteen individual figure cells.
The outputs are six main figures, four Extended Data figures, six supplementary
figures and ten tables. Inputs are in Data_for_Figures/; the data deposit's
FIGURE_TABLE_INDEX.txt maps each figure and table to its files.
Times New Roman regular and bold must be installed locally.
