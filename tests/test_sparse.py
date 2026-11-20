"""The spatially adaptive sparse grid (`jaxdd.sparse`) on functions known in closed form; on a
perovskite cell's table against `jaxdd.axes`: `validation/axes.py` (docs/validation.md)."""

import numpy as np

from jaxdd.sparse import _parents_1d, interpolate_sparse, refine_sparse

RNG = np.random.default_rng(7)


def test_a_multilinear_function_is_its_start() -> None:
    """A function linear along each axis is the hierarchical basis's own: every surplus beyond
    the corners is round-off, nothing is refined, and the interpolant is exact (1e-14), on log
    axes in the logarithm."""

    def f(x):
        u = np.log(x[:, 0] / 1e15) / np.log(1e3)
        return ((1 + 2 * u) * (3 - x[:, 1]) + u * x[:, 1] * x[:, 2])[:, None]

    ranges = [(1e15, 1e18), (0.0, 1.0), (0.0, 2.0)]
    g = refine_sparse(f, ranges, 1e-10, log=[True, False, False])
    assert g.rounds == 1 and g.stopped == "converged" and len(g.levels) == 5**3
    X = np.stack([np.exp(RNG.uniform(np.log(1e15), np.log(1e18), 500)),
                  RNG.uniform(0, 1, 500), RNG.uniform(0, 2, 500)], -1)  # fmt: skip
    np.testing.assert_allclose(interpolate_sparse(g, X), f(X), atol=1e-12)
    np.testing.assert_allclose(interpolate_sparse(g, g.points), g.values, atol=1e-12)


def test_a_peak_is_refined_where_it_is() -> None:
    """A peak in 2D (`exp(−r²/2σ²)`, σ = 0.1, off centre; one the start's five points per axis
    see: a narrower one between them leaves no surplus to refine, as the module says): the
    interpolant within twice the tolerance of it at random points, at its own points exactly,
    and the points where the peak is: within 0.25 of it twice as dense as elsewhere (2.3×; 433
    points), and in all fewer than the tensor product of the finest level reached (65²)."""
    centre = np.array([0.3, 0.62])

    def f(x):
        return np.exp(-np.sum((x - centre) ** 2, 1) / (2 * 0.1**2))[:, None]

    tol = 1e-2
    g = refine_sparse(f, [(0.0, 1.0)] * 2, tol)
    X = RNG.uniform(size=(20000, 2))
    assert np.abs(interpolate_sparse(g, X) - f(X)).max() < 2 * tol
    np.testing.assert_allclose(interpolate_sparse(g, g.points), g.values, atol=1e-12)
    near = np.linalg.norm(g.points - centre, axis=1) < 0.25
    area = np.pi * 0.25**2
    assert near.sum() / area > 2 * (~near).sum() / (1 - area)
    assert len(g.levels) < (2 ** g.levels.max() + 1) ** 2


def test_every_points_ancestors_are_in_the_grid() -> None:
    """The surpluses are each point's own only where every hierarchical parent is a grid point
    (`_with_ancestors`)."""
    g = refine_sparse(lambda x: np.tanh((x.sum(1) - 1.4) / 0.1)[:, None], [(0.0, 1.0)] * 3, 1e-1)
    have = {tuple(zip(lv, ix, strict=True)) for lv, ix in zip(g.levels.tolist(), g.index.tolist(),
                                                               strict=True)}  # fmt: skip
    assert g.rounds > 2
    for p in have:
        for k, (lv, i) in enumerate(p):
            for parent in _parents_1d(lv, i):
                assert (*p[:k], parent, *p[k + 1 :]) in have


def test_a_missing_result_is_left_to_the_interpolant() -> None:
    """Where `evaluate` gives NaN (an unconverged voltage) the point's surplus is 0, the
    interpolant there its ancestors', finite everywhere; the NaN are counted."""

    def f(x):
        out = np.stack([np.sin(3 * x[:, 0]) * x[:, 1], x[:, 0] ** 2], -1)
        out[(x[:, 0] > 0.6) & (x[:, 1] > 0.6), 0] = np.nan
        return out

    g = refine_sparse(f, [(0.0, 1.0)] * 2, 1e-2)
    assert g.nan > 0
    X = RNG.uniform(size=(2000, 2))
    got = interpolate_sparse(g, X)
    assert np.isfinite(got).all()
    assert np.abs(got[:, 1] - X[:, 0] ** 2).max() < 2e-2
