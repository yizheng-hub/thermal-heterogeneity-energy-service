import json

from thermal_heterogeneity.analysis import refresh_manifest, reproduce_analysis
from thermal_heterogeneity.config import load_config
from thermal_heterogeneity.figure_data import reproduce_figure_data
from thermal_heterogeneity.si import reproduce_si
from thermal_heterogeneity.verify import verify_manifest


def test_synthetic_level1_generates_results_48_panels_and_si(synthetic_release):
    config, output = synthetic_release
    reproduce_analysis(config, output)
    figure_manifest = reproduce_figure_data(config, output)
    si_manifest = reproduce_si(config, output)
    manifest = refresh_manifest(output, load_config(config))
    assert verify_manifest(manifest)["passed"]
    assert json.loads(figure_manifest.read_text())["panel_count"] == 48
    assert len(json.loads(si_manifest.read_text())["tables"]) == 12
