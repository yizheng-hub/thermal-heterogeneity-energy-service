from thermal_heterogeneity.building.populations import population_rng


def test_population_rng_is_context_stable():
    first = population_rng(7, "x", 2).integers(0, 100, 10)
    second = population_rng(7, "x", 2).integers(0, 100, 10)
    assert (first == second).all()
