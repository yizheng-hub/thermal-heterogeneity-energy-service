# Figure entry points

After a successful replay, run `reproduce.py --data-root <external-data> --typical-day-data-root <external-supplement> --replay-root <completed-replay>` to redraw all 30 figures and refresh their source-data index. This reads existing numerical results and performs no new scientific or physical simulation.

| Renderer | Current output |
| --- | --- |
| `publication/scripts/render_revised_main_figures.py` | Main Figure 1 and retained supporting library panels |
| `publication/scripts/render_supplementary.py` | Supporting human evidence, sensitivity and spatial plots |
| `publication/energy_service_paper/scripts/render_control_service_figures.py` | Main Figures 2-4 |
| `publication/energy_service_paper/scripts/render_response_implementation_figures.py` | Main Figures 5-6 and event-source supplementary plot |
| `publication/energy_service_paper/scripts/render_response_traces.py` | Own-baseline response/recovery traces |
| `publication/energy_service_paper/scripts/assemble_publication_assets.py` | Base six main and 21 supplementary figures |

The temporal renderers `publication/seasonal/extend_main_figure_three.py` and
`publication/typical_day_dr/render.py` add the two seasonal and one date-robustness
figures. `finalize_temporal_assets.py` sets the final S1–S24 reading order and
preserves the original six panels of main Figure 3 while adding panels g and h.

Displayed supplementary figures S1–S16 follow their first citation in the main
manuscript. Figures S17–S24 are cited inside the supplement. The finalizer also
exports `SUPPLEMENTARY_READING_ORDER.json`, including the table-alias mapping;
Source Data table filenames use the current displayed numbers. Original input
panel names remain stable and are linked through `FIGURE_SOURCE_MAP.csv`.

Normal and bold Times New Roman must be installed locally. Products are written inside the supplied replay directory and remain outside Git. Source routes and hashes are in `Supplementary_Data/FIGURE_SOURCE_MAP.csv`.
