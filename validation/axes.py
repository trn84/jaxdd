"""M6: a table of a perovskite cell's photo-collection, its axes refined by `jaxdd.axes`
against evenly spaced ones.

    python validation/axes.py [tol] [chunk] [test_points] [max_size] [ways]
    # e.g. 5e-2 512 4096 100000 refined,even,equal,sparse

The cell is `tests/data/frozen_1e17_x1_thin_light` (ions frozen after a pre-bias, the case
`validation/hard.py` varies), over four axes as a table of such cells has them: the absorber's
trap density (three decades, logarithmic), its ion density (1e15–4e18 cm⁻³, logarithmic), the
share `x_ion` of the built-in voltage the ions screen (the pre-bias at `(1 − x_ion)·V_bi`,
0–1) and its thickness (250–1000 nm, logarithmic). Each point is a light and a dark curve; the
table holds `η(V) = (J_dark − J_light)/J_gen` from −0.2 to 1.2 V, NaN where a voltage did not
converge.

Three ways, each from five points per axis and none beyond a table of `max_size` points:
"refined" (`refine_axes`), "even" (evenly spaced axes, each halved until the table passes the
same test: the numbers per axis a careful hand would choose) and "equal" (the same number of
evenly spaced points on every axis, doubled until the table's error at the test points is
within the tolerance); and "sparse" (`jaxdd.sparse.refine_sparse`, a spatially adaptive
sparse grid from the same five points per axis, not a tensor product).

Prints, for each, the axes' points, the table's size (the devices a table of it costs, two per
point), its estimated error, and its error at `test_points` random points between its points,
against the solver there. The largest error is in under 1 % of the volume: 512 test points
gave 7.2e-2 for nine points per axis where 4096 give 1.0e-1 to 1.3e-1.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.axes import in_chunks, interpolate, refine_axes
from jaxdd.batch import device_tree, solve_batch
from jaxdd.device import device_parameters
from jaxdd.simss_input import read_setup
from jaxdd.sparse import interpolate_sparse, refine_sparse

CASE = Path(__file__).resolve().parents[1] / "tests" / "data" / "frozen_1e17_x1_thin_light"
V = np.round(np.arange(-0.2, 1.2 + 1e-9, 0.05), 6)
ABSORBER = 1
NAMES = ("N_t_bulk", "N_ion", "x_ion", "L")
LOG = (True, True, False, True)
SEED = 20261115


def _setup():
    setup = read_setup(CASE / "simulation_setup.txt")
    structure, params = device_parameters(setup)
    params = {k: jnp.asarray(v) for k, v in params.items()}
    nt = float(params["N_t_bulk"][ABSORBER])
    ranges = ((nt * 10**-1.5, nt * 10**1.5), (1e21, 4e24), (0.0, 1.0), (250e-9, 1000e-9))
    return structure, params, ranges, setup.f("W_R") - setup.f("W_L")


STRUCTURE, PARAMS, RANGES, V_BI = _setup()


def _params(point, g_frac):
    nt, ions, x_ion, L = point[0], point[1], point[2], point[3]
    p = dict(PARAMS)
    p["N_t_bulk"] = p["N_t_bulk"].at[ABSORBER].set(nt)
    p["N_anion"] = p["N_anion"].at[ABSORBER].set(ions)
    p["N_cation"] = p["N_cation"].at[ABSORBER].set(ions)
    p["L"] = p["L"].at[ABSORBER].set(L)
    p["V_pre"] = jnp.asarray((1.0 - x_ion) * V_BI)
    p["G_frac"] = jnp.asarray(g_frac)
    return p


@jax.jit
def _batch(points):
    """The light devices, then the dark ones, of `points`, as one batch."""
    build = jax.vmap(lambda q, g: device_tree(STRUCTURE, _params(q, g)))
    light = build(points, jnp.ones(len(points)))
    dark = build(points, jnp.zeros(len(points)))
    return jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), light, dark)


def eta(points: np.ndarray) -> np.ndarray:
    """`η(V)` at each point, NaN where either curve did not converge."""
    batch = _batch(jnp.asarray(points))
    J, ok, _ = solve_batch(batch, V)
    J, ok = np.asarray(J), np.asarray(ok)
    n = len(points)
    j_gen = np.asarray(batch["J_scale"])[:n]  # the light devices' generation current, q ∫ G dx
    out = (J[n:] - J[:n]) / j_gen[:, None]
    return np.where(ok[:n] & ok[n:], out, np.nan)


def _even(n):
    return [
        np.geomspace(*r, n) if lg else np.linspace(*r, n) for r, lg in zip(RANGES, LOG, strict=True)
    ]


def _error(r, test, truth):
    gap = np.abs(interpolate(r.axes, r.values, test, log=LOG) - truth)
    return float(np.nanmax(gap)), float(np.nanpercentile(np.nanmax(gap, axis=1), 99))


def _memo(evaluate):
    """`evaluate` that solves a point once: the three ways share their points (all are
    midpoints of the same start)."""
    seen: dict[tuple[float, ...], np.ndarray] = {}

    def memo(points):
        keys = [tuple(p) for p in np.asarray(points).tolist()]
        new = [k for k in dict.fromkeys(keys) if k not in seen]
        if new:
            seen.update(zip(new, evaluate(np.array(new)), strict=True))
        return np.stack([seen[k] for k in keys])

    return memo


def main(tol: float = 5e-2, chunk: int = 512, n_test: int = 4096, max_size: int = 100000,
         ways: str = "refined,even,equal") -> None:  # fmt: skip
    print(f"JAX {jax.__version__} on {jax.devices()}; tol {tol}, chunk {chunk}", flush=True)
    chunked = in_chunks(eta, chunk)

    def solve(points):  # one line per call: what each round costs
        start = time.perf_counter()
        out = chunked(points)
        print(f"  solved {len(points)} points in {time.perf_counter() - start:.1f} s", flush=True)
        return out

    rng = np.random.default_rng(SEED)
    unit = rng.uniform(size=(n_test, 4))
    test = np.stack([np.exp(np.log(lo) + u * np.log(hi / lo)) if lg else lo + u * (hi - lo)
                     for (lo, hi), lg, u in zip(RANGES, LOG, unit.T, strict=True)], -1)  # fmt: skip
    truth = solve(test)
    print(f"{n_test} test points, {int(np.isnan(truth).any(axis=1).sum())} with an unconverged "
          f"voltage", flush=True)  # fmt: skip
    evaluate = _memo(solve)

    def report(name, r):
        size = r.values[..., 0].size
        worst, p99 = _error(r, test, truth)
        print(f"{name:8s} points per axis {[len(a) for a in r.axes]} = {size} ({2 * size} "
              f"devices), {r.rounds} rounds, {r.stopped}; estimate {r.estimate:.1e} (per axis "
              f"{' '.join(f'{e:.1e}' for e in r.error)}), unresolved {r.unresolved}, NaN {r.nan}; "
              f"table error max {worst:.2e}, 99th pct {p99:.2e}", flush=True)  # fmt: skip
        return worst

    ways = ways.split(",")
    if "sparse" in ways:
        print("sparse:", flush=True)
        g = refine_sparse(evaluate, RANGES, tol, log=LOG, max_points=max_size)
        gap = np.abs(interpolate_sparse(g, test) - truth)
        print(f"sparse   {len(g.levels)} points ({2 * len(g.levels)} devices), {g.rounds} rounds, "
              f"{g.stopped}; estimate {g.estimate:.1e}, NaN {g.nan}; error max "
              f"{np.nanmax(gap):.2e}, 99th pct {np.nanpercentile(np.nanmax(gap, axis=1), 99):.2e}; "
              f"levels per axis up to {g.levels.max(0).tolist()}", flush=True)  # fmt: skip
    for name, even in (("refined", False), ("even", True)):
        if name not in ways:
            continue
        print(f"{name}:", flush=True)
        r = refine_axes(evaluate, _even(5), tol, log=LOG, max_size=max_size, even=even)
        report(name, r)
        for n, a in zip(NAMES, r.axes, strict=True):
            print(f"    {n:9s}", np.array2string(a, precision=3, max_line_width=200))
    if "equal" not in ways:
        return
    print("equal: the same number of evenly spaced points on every axis", flush=True)
    n = 5
    while n**4 <= max_size:
        r = refine_axes(evaluate, _even(n), tol, log=LOG, max_rounds=0)
        if report("equal", r) < tol:
            break
        n = 2 * n - 1


if __name__ == "__main__":
    args = sys.argv[1:]
    main(float(args[0]) if args else 5e-2, *(int(a) for a in args[1:4]), *args[4:5])
