"""Verify the published projection example and the equal-area geometry."""

from pathlib import Path
import ast
import json

import numpy as np


def projection():
    path = Path(__file__).resolve().parents[2] / "plot/Plot.ipynb"
    notebook = json.loads(path.read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "main-1")
    tree = ast.parse("".join(cell["source"]))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main_equal_earth")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "Plot.ipynb:projection", "exec"), namespace)
    return namespace["main_equal_earth"]


def test_equal_earth_matches_proj_published_spherical_example():
    # PROJ documents 122 E, 47 N on a unit sphere as x=1.55, y=0.89.
    np.testing.assert_allclose(projection()(122, 47), [1.55, 0.89], atol=0.005, rtol=0)


def test_equal_earth_preserves_local_surface_area():
    project = projection()
    longitude = np.array([-130.0, 0.0, 160.0])
    latitude = np.array([-75.0, 0.0, 55.0])
    step = 1e-4
    dlon = (np.array(project(longitude + step, latitude)) -
            np.array(project(longitude - step, latitude))) / (2 * step)
    dlat = (np.array(project(longitude, latitude + step)) -
            np.array(project(longitude, latitude - step))) / (2 * step)
    jacobian = dlon[0] * dlat[1] - dlon[1] * dlat[0]
    spherical_area = np.cos(np.deg2rad(latitude)) * (np.pi / 180) ** 2
    np.testing.assert_allclose(jacobian, spherical_area, atol=0, rtol=2e-7)
