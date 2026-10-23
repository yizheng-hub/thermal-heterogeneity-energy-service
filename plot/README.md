# Figure entry points

After a successful replay, run `reproduce.py --data-root <external-data> --replay-root <completed-replay>` to redraw all 27 figures and refresh their source-data index. This reads existing numerical results and performs no new scientific or physical simulation.

| Renderer | Current output |
| --- | --- |
| `publication/scripts/render_revised_main_figures.py` | Main Figure 1 and retained supporting library panels |
| `publication/scripts/render_supplementary.py` | Supporting human evidence, sensitivity and spatial plots |
| `publication/energy_service_paper/scripts/render_control_service_figures.py` | Main Figures 2-4 |
| `publication/energy_service_paper/scripts/render_response_implementation_figures.py` | Main Figures 5-6 and event-source supplementary plot |
| `publication/energy_service_paper/scripts/render_response_traces.py` | Own-baseline response/recovery traces |
| `publication/energy_service_paper/scripts/assemble_publication_assets.py` | Six main and 21 supplementary figures in reading order |

Normal and bold Times New Roman must be installed locally. Products are written inside the supplied replay directory and remain outside Git. Source routes and hashes are in `Supplementary_Data/FIGURE_SOURCE_MAP.csv`.
