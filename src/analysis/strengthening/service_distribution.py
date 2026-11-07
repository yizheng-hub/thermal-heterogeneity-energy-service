"""Exact weighted empirical temperature distributions for paired demand rescoring."""

import numpy as np


class ZoneDistribution:
    def __init__(self, temperature, weights):
        valid = np.asarray(weights) > 0
        t = np.asarray(temperature, dtype=float)[valid]
        w = np.asarray(weights, dtype=float)[valid]
        if not len(t) or not np.isfinite(t).all():
            raise ValueError("No valid occupied temperature distribution")
        order = np.argsort(t, kind="mergesort")
        self.t = t[order]
        w = w[order]
        self.w = np.r_[0, np.cumsum(w)]
        self.s = np.r_[0, np.cumsum(w * self.t)]
        self.total = self.w[-1]

    def absolute(self, center):
        x = np.asarray(center, dtype=float)
        i = np.searchsorted(self.t, x, side="right")
        return (x * (2 * self.w[i] - self.total) + self.s[-1] - 2 * self.s[i]) / self.total

    def covered(self, lower, upper):
        lo = np.searchsorted(self.t, np.asarray(lower) - 1e-9, side="left")
        hi = np.searchsorted(self.t, np.asarray(upper) + 1e-9, side="right")
        return (self.w[hi] - self.w[lo]) / self.total


class ServiceDistribution:
    def __init__(self, data, profiles, mask=None):
        if mask is None:
            mask = np.ones(len(data["occupied"]), dtype=bool)
        weights = data["people"] * (data["occupied"] & mask)[:, None]
        zones = [str(z).upper() for z in data["zone_ids"]]
        self.zone = np.asarray([zones.index(str(z).upper()) for z in profiles.zone_id])
        counts = np.bincount(self.zone, minlength=len(zones))
        self.distributions = [
            ZoneDistribution(data["temperatures"][:, i], weights[:, i]) for i in range(len(zones))
        ]
        self.person_hours = np.asarray([self.distributions[z].total / counts[z] for z in self.zone])
        self.weights = self.person_hours / self.person_hours.sum()

    def individual(self, centers):
        c = np.asarray(centers, dtype=float)
        result = np.zeros(c.shape)
        if c.shape[-1] != len(self.zone):
            raise ValueError("Profile axis differs")
        for z, dist in enumerate(self.distributions):
            positions = np.flatnonzero(self.zone == z)
            result[..., positions] = dist.absolute(c[..., positions])
        return result

    def score(self, centers, tail):
        individual = self.individual(centers)
        return individual @ self.weights, individual[..., tail].mean(axis=-1)

    def coverage(self, lower, upper):
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        result = np.zeros(lower.shape)
        for z, dist in enumerate(self.distributions):
            p = np.flatnonzero(self.zone == z)
            result[..., p] = dist.covered(lower[..., p], upper[..., p])
        return result


class PairedWorseningHours:
    """Exact model person-hours with larger absolute mismatch than the reference."""

    def __init__(self, candidate, reference, profiles, tolerance=1e-9):
        assert np.array_equal(candidate["zone_ids"], reference["zone_ids"])
        assert np.allclose(candidate["people"], reference["people"], atol=1e-9, rtol=0)
        assert np.array_equal(candidate["occupied"], reference["occupied"])
        zones = [str(z).upper() for z in candidate["zone_ids"]]
        self.zone = np.array([zones.index(str(z).upper()) for z in profiles.zone_id])
        counts = np.bincount(self.zone, minlength=len(zones))
        self.parts = []
        for z in range(len(zones)):
            t = candidate["temperatures"][:, z]
            r = reference["temperatures"][:, z]
            w = candidate["people"][:, z] * candidate["occupied"] / counts[z]
            parts = []
            for greater in [True, False]:
                eligible = ((t - r > tolerance) if greater else (r - t > tolerance)) & (w > 0)
                midpoint = (t[eligible] + r[eligible] + (-tolerance if greater else tolerance)) / 2
                order = np.argsort(midpoint)
                parts.append((midpoint[order], np.r_[0, np.cumsum(w[eligible][order])]))
            self.parts.append(parts)

    def person_hours(self, centers):
        centers = np.asarray(centers)
        result = np.zeros(centers.shape)
        for z, parts in enumerate(self.parts):
            p = np.flatnonzero(self.zone == z)
            x = centers[..., p]
            for greater, (midpoint, weight) in zip([True, False], parts):
                ix = np.searchsorted(midpoint, x, side="right" if greater else "left")
                result[..., p] += weight[-1] - weight[ix] if greater else weight[ix]
        return result.sum(axis=-1)
