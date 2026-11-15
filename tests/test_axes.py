"""`jaxdd.axes`: tables whose axes are refined by the interpolation's error, on functions whose
answer is known (the solver's own table is `validation/axes.py`)."""

import numpy as np
import pytest

from jaxdd.axes import in_chunks, interpolate, refine_axes, tabulate


def peaked(points):
    """A narrow peak along the first axis, a gentle curve along the second (logarithmic, over
    two decades), two outputs."""
    x, y = points[:, 0], points[:, 1]
    peak = np.exp(-(((x - 0.3) / 0.02) ** 2)) + 0.1 * np.log10(y) ** 2
    return np.stack([peak, 0.5 * peak], axis=-1)


def _uniform(n_x, n_y):
    return [np.linspace(0, 1, n_x), np.geomspace(1, 100, n_y)]


def _test_points(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, 1, n), 10 ** rng.uniform(0, 2, n)], axis=-1)


def _table_error(axes):
    values = tabulate(peaked, axes)
    pts = _test_points()
    return float(np.max(np.abs(interpolate(axes, values, pts, log=[False, True]) - peaked(pts))))


def test_interpolation_is_exact_on_multilinear_functions_and_at_the_points() -> None:
    """Multilinear in the axes' coordinates (the second logarithmic): exact everywhere; any
    function: exact at the table's points; outside an axis refused."""
    axes = [np.array([0.0, 0.2, 1.0]), np.array([1.0, 10.0, 100.0, 1000.0])]

    def bilinear(p):
        return ((1 + 2 * p[:, 0]) * (3 - np.log(p[:, 1])))[:, None] * np.array([1.0, -2.0])

    values = tabulate(bilinear, axes)
    assert values.shape == (3, 4, 2)
    pts = np.stack([np.linspace(0, 1, 50), np.geomspace(1, 1000, 50)], axis=-1)
    got = interpolate(axes, values, pts, log=[False, True])
    assert np.max(np.abs(got - bilinear(pts))) < 1e-12
    mesh = np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], -1)
    assert np.max(np.abs(interpolate(axes, tabulate(peaked, axes), mesh, [False, True])
                         - peaked(mesh))) < 1e-15  # fmt: skip
    with pytest.raises(ValueError, match="outside"):
        interpolate(axes, values, np.array([[1.5, 10.0]]), log=[False, True])


def test_refined_axes_meet_the_tolerance_with_fewer_points_than_even_ones() -> None:
    """From a coarse start that sees the peak (eleven points along its axis, one of them on it;
    three along the other): the refined table interpolates within `tol` between its points, the
    points go where the peak is, and no evenly spaced table of twice as many points does as
    well."""
    tol = 1e-2
    r = refine_axes(peaked, _uniform(11, 3), tol, log=[False, True])
    assert r.unresolved == (0, 0) and r.nan == 0
    assert r.estimate < tol and max(r.error) <= r.estimate
    assert _table_error(r.axes) < tol
    assert np.array_equal(r.values, tabulate(peaked, r.axes))
    near = np.abs(r.axes[0] - 0.3) < 0.1
    assert near.sum() > 2 * (~near).sum()  # most of the first axis's points at the peak
    refined = len(r.axes[0]) * len(r.axes[1])
    for n_y in range(3, 10):
        assert _table_error(_uniform(2 * refined // n_y, n_y)) > tol, n_y


def test_an_axis_is_refined_for_its_part_of_the_cells_error_not_for_a_share() -> None:
    """`x² + 0.2·y²` on three points per axis misses by 0.0625 along x and 0.0125 along y in
    every cell: above 0.05 together, so x, the larger part, is halved (0.0156), and the sum is
    below it with y as it was, though y alone is above an equal share of the tolerance."""

    def bowl(p):
        return p[:, 0] ** 2 + 0.2 * p[:, 1] ** 2

    start = [np.linspace(0, 1, 3), np.linspace(0, 1, 3)]
    r = refine_axes(bowl, start, 5e-2)
    assert [len(a) for a in r.axes] == [5, 3] and r.stopped == "converged"
    assert r.estimate == pytest.approx(0.0625 / 4 + 0.0125) and r.error[1] == pytest.approx(0.0125)
    pts = np.random.default_rng(0).uniform(size=(2000, 2))
    gap = np.abs(interpolate(r.axes, r.values, pts) - bowl(pts))
    assert 0.9 * r.estimate < gap.max() <= r.estimate


def test_cells_far_apart_do_not_add_up() -> None:
    """A ridge narrow along x where y is small and one narrow along y where x is large: each
    axis's largest miss is in another corner of the table, and the estimate is about the larger
    of the two, not their sum."""

    def ridges(p):
        x, y = p[:, 0], p[:, 1]
        return np.exp(-(((x - 0.5) / 0.1) ** 2) - (y / 0.4) ** 2) + np.exp(
            -(((y - 0.5) / 0.1) ** 2) - ((x - 1) / 0.4) ** 2
        )

    start = [np.linspace(0, 1, 17), np.linspace(0, 1, 17)]
    r = refine_axes(ridges, start, 1.0)
    assert r.rounds == 0 and r.stopped == "converged"
    assert max(r.error) <= r.estimate < 0.6 * sum(r.error)


def test_a_start_that_misses_a_feature_never_finds_it() -> None:
    """The limit of any test by samples: from five points along the first axis the peak (at 0.3,
    0.02 wide) lies between them (0.25, 0.5), leaves no curvature in the table, and nothing is
    refined there."""
    r = refine_axes(peaked, _uniform(5, 3), 1e-2, log=[False, True])
    assert len(r.axes[0]) == 5 and _table_error(r.axes) > 0.5


def test_refinement_stops_at_its_limits_and_counts_nan() -> None:
    """`max_points` caps an axis and reports what it left; NaN results are left out and
    counted; each point is solved once, in one call per round, and every point solved is one of
    the table's."""
    calls = []

    def counted(p):
        calls.append(len(p))
        out = peaked(p)
        out[p[:, 0] > 0.9] = np.nan
        return out

    r = refine_axes(counted, _uniform(7, 3), 1e-4, log=[False, True], max_points=9)
    assert len(r.axes[0]) == 9 and r.unresolved[0] > 0
    assert r.nan > 0
    assert sum(calls) == r.values[..., 0].size and len(calls) <= r.rounds + 1


def test_refinement_stops_before_a_table_too_large() -> None:
    r = refine_axes(peaked, _uniform(11, 3), 1e-2, log=[False, True], max_size=200)
    assert r.stopped == "max_size" and r.values[..., 0].size <= 200
    assert refine_axes(peaked, _uniform(11, 3), 1e-2, log=[False, True]).stopped == "converged"


def test_refinement_needs_three_points_per_axis() -> None:
    with pytest.raises(ValueError, match="three points"):
        refine_axes(peaked, _uniform(9, 2), 1e-2, log=[False, True])


def test_in_chunks_keeps_one_batch_size() -> None:
    sizes = []

    def f(p):
        sizes.append(len(p))
        return p.sum(axis=1)

    pts = np.arange(23.0).reshape(-1, 1)
    assert np.array_equal(in_chunks(f, 8)(pts), pts[:, 0])
    assert sizes == [8, 8, 8]
