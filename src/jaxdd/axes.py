"""Tables on a tensor-product grid whose axes are refined where the interpolation needs it.

A table over several parameters, interpolated multilinearly between its points, is accurate
where the axes' points are dense enough for the curvature along each axis. Chosen by hand, the
points of each axis are spaced evenly (in the parameter or its logarithm) and as densely as the
worst region of that axis needs. `refine_axes` chooses them by the interpolation's own error:
it solves the table on coarse axes, estimates from the table itself how far the interpolation
misses in each of its cells, and where that exceeds the tolerance inserts the midpoint of the
axis that misses most there; then solves the new points and estimates again. The axes only
grow, so every point solved is one of the final table's: the refinement costs the table and
nothing else. The table stays a tensor product, read as before (`interpolate`), only with
fewer points.

The estimate: multilinear interpolation of a smooth function errs by about the sum, over the
axes, of each axis's own linear interpolation error, `h²/8·|∂²f/∂x²|` for an interval `h`, the
second derivative from the table's second differences along that axis. The sum is taken in each
cell of the table and for each output on its own, and the largest is the table's estimated
error. The axes' worst places are seldom the same one: on a perovskite cell's table over four
axes the sum of each axis's largest miss was twice the largest sum in one cell, and the largest
sum the table's error within a third (`validation/axes.py`). So an axis is refined for the
cells where the sum is too large and it is the largest part, not until its own largest miss is
a fixed share of the tolerance, which would refine every axis for another's worst place.

What it cannot do:

- Find a feature its start misses. A peak that falls between the start's points leaves no
  curvature in the table, as it would escape any test by samples. So the start is a coarse
  hand-made axis that resolves every feature at least once (a peak, a step, a change of regime
  somewhere on the axis), not only the axis's ends.
- Save points on a feature that lies across the axes or moves with the output. A front whose
  place on one axis depends on the others, or on the voltage of a curve stored whole, is
  narrow in every cell and still crosses most intervals of the axis: a tensor product then
  needs the points everywhere, and the refinement gives what an evenly spaced table with the
  right number of points per axis gives.

The function is the caller's: `evaluate(points)` takes an `(N, d)` array of parameter points
and returns an `(N, ...)` array of results (a curve per point, or a few numbers), NaN where
it has none (an unconverged voltage). jaxdd's solver is one such function; `in_chunks` makes
it take any `N` with one compiled batch size. Each round of the refinement calls it once, with
every new point of the table, so a batched solver solves them together.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["Refinement", "in_chunks", "interpolate", "refine_axes", "tabulate"]


def _coords(axis: np.ndarray, log: bool) -> np.ndarray:
    return np.log(axis) if log else np.asarray(axis, dtype=float)


def _from_coords(c: np.ndarray, log: bool) -> np.ndarray:
    return np.exp(c) if log else c


def _check_axes(axes: Sequence[np.ndarray], log: Sequence[bool]) -> list[np.ndarray]:
    out = []
    for k, (a, lg) in enumerate(zip(axes, log, strict=True)):
        a = np.asarray(a, dtype=float)
        if a.ndim != 1 or a.size < 2:
            raise ValueError(f"axis {k}: at least two points, one-dimensional")
        if np.any(np.diff(a) <= 0):
            raise ValueError(f"axis {k}: its points must increase")
        if lg and a[0] <= 0:
            raise ValueError(f"axis {k}: a logarithmic axis needs positive points")
        out.append(a)
    return out


def _log_flags(axes: Sequence[np.ndarray], log: bool | Sequence[bool] | None) -> list[bool]:
    if log is None or isinstance(log, bool):
        return [bool(log)] * len(axes)
    if len(log) != len(axes):
        raise ValueError(f"log: one flag per axis, {len(axes)}, got {len(log)}")
    return [bool(v) for v in log]


def in_chunks(evaluate: Callable[[np.ndarray], np.ndarray], size: int) -> Callable:
    """`evaluate` called on chunks of exactly `size` points (the last padded with copies of its
    last point, their results dropped), so that a solver compiled for one batch size is not
    compiled again for every other `N` (`solve_batch` compiles per shape)."""

    def chunked(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        parts = []
        for i in range(0, len(points), size):
            part = points[i : i + size]
            n = len(part)
            if n < size:
                part = np.concatenate([part, np.repeat(part[-1:], size - n, axis=0)])
            parts.append(np.asarray(evaluate(part))[:n])
        return np.concatenate(parts)

    return chunked


def tabulate(
    evaluate: Callable[[np.ndarray], np.ndarray], axes: Sequence[np.ndarray]
) -> np.ndarray:
    """`evaluate` at every point of the tensor product of `axes`: an array of shape
    `(len(axes[0]), …, len(axes[-1]), ...)`, the last axis varying fastest."""
    axes = [np.asarray(a, dtype=float) for a in axes]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=-1)
    values = np.asarray(evaluate(points))
    return values.reshape(*(len(a) for a in axes), *values.shape[1:])


def interpolate(
    axes: Sequence[np.ndarray],
    values: np.ndarray,
    points: np.ndarray,
    log: bool | Sequence[bool] | None = None,
) -> np.ndarray:
    """The table `values` (`tabulate`'s shape) interpolated multilinearly at `points` (`(N, d)`),
    linear in each axis's parameter or, where `log` says so, in its logarithm. Points outside an
    axis raise `ValueError`: a table does not extrapolate."""
    flags = _log_flags(axes, log)
    axes = _check_axes(axes, flags)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    d = len(axes)
    if points.shape[1] != d:
        raise ValueError(f"points: {d} coordinates each, got {points.shape[1]}")
    lo, frac = [], []
    for k, (a, lg) in enumerate(zip(axes, flags, strict=True)):
        c, x = _coords(a, lg), _coords(points[:, k], lg)
        if np.any(x < c[0] - 1e-12 * abs(c[0])) or np.any(x > c[-1] + 1e-12 * abs(c[-1])):
            raise ValueError(f"axis {k}: points outside [{a[0]:g}, {a[-1]:g}]")
        j = np.clip(np.searchsorted(c, x, side="right") - 1, 0, len(c) - 2)
        lo.append(j)
        frac.append(np.clip((x - c[j]) / (c[j + 1] - c[j]), 0.0, 1.0))
    out = 0.0
    for corner in range(2**d):
        bits = [(corner >> k) & 1 for k in range(d)]
        weight = np.ones(len(points))
        for k, b in enumerate(bits):
            weight = weight * (frac[k] if b else 1.0 - frac[k])
        v = values[tuple(lo[k] + b for k, b in enumerate(bits))]
        out = out + weight.reshape(-1, *([1] * (v.ndim - 1))) * v
    return out


@dataclasses.dataclass(frozen=True)
class Refinement:
    """What `refine_axes` found: the refined `axes` and the table on them (`values`, as
    `tabulate` returns it); `rounds`, the rounds that inserted points; `estimate`, the table's
    estimated error at the end (the largest sum of the axes' misses in one cell, for one
    output); `error`, per axis its largest miss alone (`h²/8·|f''|`); `unresolved`, per axis
    the intervals that are the largest part of a cell above the tolerance and were not split
    (`min_width`, `max_points`, or the end by `max_rounds` or `max_size`); `nan`, the table's
    points with a NaN result; `stopped`, why it ended: "converged" (nothing left to insert),
    "max_rounds" or "max_size" (the next round's table would have been larger)."""

    axes: tuple[np.ndarray, ...]
    values: np.ndarray
    rounds: int
    estimate: float
    error: tuple[float, ...]
    unresolved: tuple[int, ...]
    nan: int
    stopped: str


def _misses(values: np.ndarray, coords: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Per axis, the estimated miss `h²/8·|f''|` of its interval in every cell of the table, for
    every output: arrays of shape `(n_0 − 1, …, n_{d−1} − 1, outputs)`. `f''` is the larger second
    divided difference at the interval's two ends (at an axis's end, its neighbour's), and of a
    cell's corners along the other axes the largest; a NaN counts as none."""
    d = len(coords)
    flat = values.reshape(*values.shape[:d], -1)
    shape = (-1,) + (1,) * d
    out = []
    for a, c in enumerate(coords):
        f = np.moveaxis(flat, a, 0)
        h = np.diff(c).reshape(shape)
        second = 2.0 * np.diff(np.diff(f, axis=0) / h, axis=0) / (c[2:] - c[:-2]).reshape(shape)
        second = np.nan_to_num(np.abs(second), nan=0.0)
        kappa = np.maximum(np.concatenate([second[:1], second]),
                           np.concatenate([second, second[-1:]]))  # fmt: skip
        miss = h**2 / 8.0 * kappa  # the axis's intervals × the other axes' points × outputs
        for k in range(1, d):  # the other axes' points to their intervals
            miss = np.maximum(np.take(miss, range(miss.shape[k] - 1), axis=k),
                              np.take(miss, range(1, miss.shape[k]), axis=k))  # fmt: skip
        out.append(np.moveaxis(miss, 0, a))
    return out


def refine_axes(
    evaluate: Callable[[np.ndarray], np.ndarray],
    axes: Sequence[np.ndarray],
    tol: float,
    log: bool | Sequence[bool] | None = None,
    min_width: float | Sequence[float] = 0.0,
    max_points: int | Sequence[int] = 129,
    max_rounds: int = 20,
    max_size: int | None = None,
    even: bool = False,
) -> Refinement:
    """Axes for a table of `evaluate` that interpolates it within `tol` (module docstring), and
    the table on them.

    Starts from `axes`, at least three points each, coarse but resolving every feature (the
    module docstring says why), and refines in rounds. Each round solves the table on the
    current axes (only its new points: the axes only grow, so every earlier point is one of
    the table's), and estimates the interpolation's miss in each cell, for each output, as the
    sum over the axes of `h²/8·|f''|`, in the axis's parameter or, where `log` says so, its
    logarithm: `h` the cell's interval of that axis, `f''` the larger second divided difference
    at the interval's two ends (at an axis's end, its neighbour's), the largest of the cell's
    corners along the other axes. Where the sum exceeds `tol`, the midpoint of the interval
    with the largest part is inserted (the largest that may still be split). An interval
    narrower than `min_width` (in the axis's coordinate) is not split; an axis stops at
    `max_points`, its worst intervals first. The refinement ends when a round inserts nothing,
    after `max_rounds`, or before a round whose table would hold more than `max_size` points
    (its axes are not taken). NaN results are left out of the estimate and counted
    (`Refinement.nan`). With `even` an axis with any interval to split has every interval
    split: evenly spaced axes, each halved until the table passes the same test, as a careful
    hand would choose them (for comparison, `validation/axes.py`). The estimate holds as many
    numbers as the table per axis."""
    flags = _log_flags(axes, log)
    axes = _check_axes(axes, flags)
    if any(len(a) < 3 for a in axes):
        raise ValueError("refine_axes needs at least three points per axis (for f'')")
    d = len(axes)
    widths = np.broadcast_to(np.asarray(min_width, dtype=float), (d,))
    limits = np.broadcast_to(np.asarray(max_points), (d,))
    cache: dict[tuple[float, ...], np.ndarray] = {}
    rounds = 0
    estimate = np.nan
    error = [np.nan] * d
    unresolved = [0] * d

    def table(axes):
        mesh = np.meshgrid(*axes, indexing="ij")
        points = [tuple(p) for p in np.stack([m.ravel() for m in mesh], -1).tolist()]
        new = [p for p in points if p not in cache]
        if new:
            values = np.asarray(evaluate(np.array(new)))
            if values.shape[0] != len(new):
                raise ValueError(f"evaluate returned {values.shape[0]} results for {len(new)}")
            cache.update(zip(new, values, strict=True))
        out = np.stack([cache[p] for p in points])
        return out.reshape(*(len(a) for a in axes), *out.shape[1:])

    stopped = "converged"
    while True:
        values = table(axes)
        changed = False
        new_axes = []
        wanted_all = [0] * d
        coords = [_coords(axes[a], flags[a]) for a in range(d)]
        miss = np.stack(_misses(values, coords))  # axes × cells × outputs
        total = miss.sum(axis=0)
        over = total > tol
        estimate = float(total.max())
        # each place above the tolerance asks for its largest contributor that may be split
        free = [(np.diff(coords[a]) > widths[a]) & (len(axes[a]) < limits[a]) for a in range(d)]
        along = [tuple(-1 if k == a else 1 for k in range(d + 1)) for a in range(d)]
        allowed = np.stack([np.where(free[a].reshape(along[a]), miss[a], -np.inf)
                            for a in range(d)])  # fmt: skip
        asked = np.where(over & np.isfinite(allowed.max(axis=0)), allowed.argmax(axis=0), -1)
        largest = np.where(over, miss.argmax(axis=0), -1)
        for a in range(d):
            c = coords[a]
            others = tuple(k for k in range(d + 1) if k != a)
            split = (asked == a).any(axis=others)
            wanted = (largest == a).any(axis=others) | split
            if even and split.any():
                split = np.diff(c) > widths[a]
            room = int(limits[a]) - len(axes[a])
            if split.sum() > room:  # the worst intervals first
                worst = np.where(asked == a, total, -np.inf).max(axis=others)
                keep = np.argsort(-np.where(split, worst, -np.inf), kind="stable")[: max(room, 0)]
                split = np.zeros_like(split)
                split[keep] = True
            unresolved[a] = int((wanted & ~split).sum())
            wanted_all[a] = int((wanted | split).sum())
            error[a] = float(miss[a].max())
            if split.any():
                changed = True
                mids = _from_coords(0.5 * (c[:-1] + c[1:]), flags[a])
                new_axes.append(np.sort(np.concatenate([axes[a], mids[split]])))
            else:
                new_axes.append(axes[a])
        if not changed:
            break
        if rounds == max_rounds:
            stopped = "max_rounds"
        elif max_size is not None and np.prod([len(x) for x in new_axes]) > max_size:
            stopped = "max_size"
        if stopped != "converged":  # the round's intervals are not split
            unresolved = wanted_all
            break
        rounds += 1
        axes = new_axes

    nan = int(np.isnan(values.reshape(*values.shape[:d], -1)).any(axis=-1).sum())
    return Refinement(axes=tuple(axes), values=values, rounds=rounds, estimate=estimate,
                      error=tuple(error), unresolved=tuple(unresolved), nan=nan,
                      stopped=stopped)  # fmt: skip
