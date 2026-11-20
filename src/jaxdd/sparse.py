"""Samples that are not a tensor product: a spatially adaptive sparse grid.

A table on a tensor product (`jaxdd.axes`) needs its points wherever a feature crosses an axis:
a front whose place on one axis moves with the others crosses most intervals of every axis,
and every point of each such interval is solved along all the other axes too. A sparse grid
places its points one by one instead, where the interpolation misses.

The interpolant is a sum of piecewise multilinear hat functions, each centred on a grid point
and as wide as its level along each axis (the hierarchical basis): along one axis, level 0 are
the two ends (`1 − u` and `u` on the unit interval), level `ℓ ≥ 1` the hats of half-width
`2^−ℓ` around the odd multiples of `2^−ℓ`. A point's surplus is what its value adds to the
interpolant of the points before it: the solver's value there less the interpolant of its
ancestors (every other basis function vanishes at it). Where a point's surplus is above the
tolerance, the interpolant still misses by about that much around it, and its children (the
two points of the next level beside it, along each axis) are added; every ancestor of a new
point is added with it, so that the surpluses are each point's own. Each round solves its new
points in one call to `evaluate`, as `jaxdd.axes.refine_axes` does.

The estimate is the largest surplus of the last points added: the interpolant's miss where it
is still refining. Like a tensor product's refinement it cannot find a feature its start
misses entirely, so it starts from a tensor product of every axis's levels up to `start_level`
(level 2: five points per axis, `refine_axes`' usual start). Cells it does not refine keep a
linear interpolant, accurate to their own surpluses' size.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["SparseGrid", "interpolate_sparse", "refine_sparse"]


@dataclasses.dataclass
class SparseGrid:
    """A sparse grid's points and surpluses: `levels`, `index` (N × d; the point along axis `k`
    at `index/2^levels` of its range, level 0 its ends), `points` (N × d, in the parameters),
    `values` (`evaluate`'s, N × ...), `surplus` (the hierarchical coefficients, NaN results
    taken as 0), `ranges` and `log` as given; `rounds`, `stopped` (why the refinement ended),
    `estimate` (the largest surplus of the last points added), `nan` (results that were NaN)."""

    levels: np.ndarray
    index: np.ndarray
    points: np.ndarray
    values: np.ndarray
    surplus: np.ndarray
    ranges: list[tuple[float, float]]
    log: list[bool]
    rounds: int
    stopped: str
    estimate: float
    nan: int


def _unit(points, ranges, log):
    """Parameters to the unit cube (log axes in their logarithm)."""
    out = []
    for k, ((lo, hi), lg) in enumerate(zip(ranges, log, strict=True)):
        x = np.asarray(points)[:, k]
        out.append((np.log(x / lo) / np.log(hi / lo)) if lg else (x - lo) / (hi - lo))
    return np.stack(out, -1)


def _from_unit(u, ranges, log):
    out = []
    for k, ((lo, hi), lg) in enumerate(zip(ranges, log, strict=True)):
        out.append(
            np.exp(np.log(lo) + u[:, k] * np.log(hi / lo)) if lg else lo + u[:, k] * (hi - lo)
        )
    return np.stack(out, -1)


def _hat(u, level, index):
    """The 1D basis functions at `u` (broadcast): level 0 the ends' `1 − u` and `u`, level
    `ℓ ≥ 1` the hat of half-width `2^−ℓ` at `index·2^−ℓ`."""
    scale = np.ldexp(1.0, level)
    hat = np.maximum(0.0, 1.0 - np.abs(u * scale - index))
    end = np.where(index == 0, 1.0 - u, u)
    return np.where(level == 0, end, hat)


def _basis(u, levels, index, chunk=2048):
    """Each grid point's basis function at the points `u` (M × d): M × N, in chunks of `u`."""
    out = np.empty((u.shape[0], levels.shape[0]))
    for start in range(0, u.shape[0], chunk):
        part = u[start : start + chunk]
        phi = np.ones((part.shape[0], levels.shape[0]))
        for k in range(levels.shape[1]):
            phi *= _hat(part[:, k, None], levels[None, :, k], index[None, :, k])
        out[start : start + chunk] = phi
    return out


def _children_1d(level, index):
    """A 1D node's children: the ends' is the middle, a hat's the two of the next level beside
    it."""
    if level == 0:
        return [(1, 1)]
    return [(level + 1, 2 * index - 1), (level + 1, 2 * index + 1)]


def _parents_1d(level, index):
    """A 1D node's hierarchical parents: the middle's are the two ends, a finer hat's the hat of
    the level above whose support holds it (the odd one of its two neighbours)."""
    if level == 0:
        return []
    if level == 1:
        return [(0, 0), (0, 1)]
    lo, hi = (index - 1) // 2, (index + 1) // 2
    return [(level - 1, lo if lo % 2 else hi)]


def _start(d, start_level):
    """The tensor product of every axis's nodes up to `start_level`."""
    nodes = [(0, 0), (0, 1)] + [(lv, i) for lv in range(1, start_level + 1)
                                for i in range(1, 2**lv, 2)]  # fmt: skip
    return [tuple(c) for c in itertools.product(nodes, repeat=d)]


def _with_ancestors(new, known):
    """`new` points (tuples of `(level, index)` per axis) with every ancestor not in `known`."""
    out, stack = set(), list(new)
    while stack:
        p = stack.pop()
        if p in known or p in out:
            continue
        out.add(p)
        for k, (lv, i) in enumerate(p):
            for parent in _parents_1d(lv, i):
                stack.append((*p[:k], parent, *p[k + 1 :]))
    return out


def refine_sparse(
    evaluate: Callable[[np.ndarray], np.ndarray],
    ranges: Sequence[tuple[float, float]],
    tol: float,
    log: bool | Sequence[bool] = False,
    start_level: int = 2,
    max_level: int = 12,
    max_points: int | None = None,
    max_rounds: int = 30,
) -> SparseGrid:
    """A sparse grid of `evaluate` over the box `ranges` (one `(lo, hi)` per axis; `log` axes
    in their logarithm) whose interpolant (`interpolate_sparse`) is within about `tol` of it
    (module docstring): from a tensor product of every axis's levels up to `start_level`, the
    children of every point whose surplus (its largest over the outputs) exceeds `tol`, round
    by round, each round's new points in one call. No axis beyond `max_level`; the refinement
    ends when a round adds nothing, after `max_rounds`, or before a round that would pass
    `max_points`."""
    d = len(ranges)
    log = [bool(log)] * d if isinstance(log, bool) else [bool(x) for x in log]
    ranges = [(float(lo), float(hi)) for lo, hi in ranges]
    order: list[tuple] = []  # the points, in the order their surpluses were taken
    pos: dict[tuple, int] = {}
    levels = np.zeros((0, d), dtype=int)
    index = np.zeros((0, d), dtype=int)
    values = surplus = None
    refined: set[tuple] = set()
    new = set(_start(d, start_level))
    rounds, stopped, estimate, nan = 0, "converged", np.inf, 0
    while True:
        if max_points is not None and len(order) + len(new) > max_points:
            stopped = "max_points"
            break
        # One call for the round; the surpluses by the points' total level, coarse first: a
        # point's ancestors all have a smaller total, and points of one total are not each
        # other's ancestors.
        batch = sorted(new, key=lambda p: sum(lv for lv, _ in p))
        lv_new = np.array([[lv for lv, _ in p] for p in batch], dtype=int)
        ix_new = np.array([[i for _, i in p] for p in batch], dtype=int)
        u_new = np.ldexp(ix_new.astype(float), -lv_new)
        f_new = np.asarray(evaluate(_from_unit(u_new, ranges, log)), dtype=float)
        if values is None:
            values = np.zeros((0, *f_new.shape[1:]))
            surplus = np.zeros((0, *f_new.shape[1:]))
        flat = f_new.reshape(len(batch), -1)
        nan += int(np.isnan(flat).sum())
        totals = lv_new.sum(1)
        old = surplus.reshape(len(order), flat.shape[1])
        s_new = np.zeros_like(flat)
        for t in np.unique(totals):
            rows, before = np.flatnonzero(totals == t), np.flatnonzero(totals < t)
            lv_all = np.concatenate([levels, lv_new[before]])
            ix_all = np.concatenate([index, ix_new[before]])
            prior = _basis(u_new[rows], lv_all, ix_all) @ np.concatenate([old, s_new[before]])
            s_new[rows] = np.where(np.isnan(flat[rows]), 0.0, flat[rows] - prior)
        for p in batch:
            pos[p] = len(order)
            order.append(p)
        levels = np.concatenate([levels, lv_new])
        index = np.concatenate([index, ix_new])
        values = np.concatenate([values, f_new])
        surplus = np.concatenate([surplus, s_new.reshape(f_new.shape)])
        size = np.max(np.abs(s_new), axis=1)
        estimate = float(size.max()) if len(size) else 0.0
        rounds += 1
        if rounds > max_rounds:
            stopped = "max_rounds"
            break
        flagged = [p for p, s in zip(batch, size, strict=True) if s > tol and p not in refined]
        children = set()
        for p in flagged:
            refined.add(p)
            for k, (lv, i) in enumerate(p):
                if lv + 1 > max_level:
                    continue
                for c in _children_1d(lv, i):
                    children.add((*p[:k], c, *p[k + 1 :]))
        new = _with_ancestors(children, pos)
        if not new:
            break
    points = _from_unit(np.ldexp(index.astype(float), -levels), ranges, log)
    return SparseGrid(levels=levels, index=index, points=points, values=values, surplus=surplus,
                      ranges=ranges, log=log, rounds=rounds, stopped=stopped, estimate=estimate,
                      nan=nan)  # fmt: skip


def interpolate_sparse(grid: SparseGrid, points: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """The sparse grid's interpolant at `points` (M × d, in the parameters, within the ranges):
    M × the outputs' shape."""
    u = _unit(np.atleast_2d(np.asarray(points, dtype=float)), grid.ranges, grid.log)
    flat = grid.surplus.reshape(len(grid.levels), -1)
    out = np.concatenate([_basis(u[s : s + chunk], grid.levels, grid.index) @ flat
                          for s in range(0, len(u), chunk)])  # fmt: skip
    return out.reshape(len(u), *grid.surplus.shape[1:])
