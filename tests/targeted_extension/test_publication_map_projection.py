"""Verify the published projection example and the equal-area geometry."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def projection():
    path = Path(__file__).resolve().parents[2] / "plot/strengthened_figures.py"
    spec = spec_from_file_location("strengthened_map_projection", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.equal_earth


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
