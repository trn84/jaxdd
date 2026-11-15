"""A fit of a transient by its gradients through time (Stage C, M7): a device's parameters
recovered from its own transient.

    .venv/bin/python validation/fit_transient.py CASE [--steps N] [--start S] [--fit NAME:LAYER ...]

The case is a ZimT input in tests/data (`CASE`, e.g. `zimt_ions_fast_light`); its transient as
jaxdd solves it at the input's parameters is the curve to fit (`Jext` over its scale, or `Vint`
in V at the open circuit), over its first `N` steps. The fit starts with each fitted
parameter `S` times its true value (the first half up, the second half down), builds the
device in JAX from its parameters (`batch.device_tree`), and minimises the misfit by
`scipy.optimize.least_squares` in the parameters' logarithms, its Jacobian from `jax.jacfwd`:
forward tangents carried from step to step, each step's by the implicit function at its
converged solution.

Printed: the parameters fitted against the true ones, the misfit at the start and the end, the
evaluations and their time.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from jaxdd.batch import device_tree
from jaxdd.device import device_parameters
from jaxdd.simss_input import read_setup, read_tvg
from jaxdd.transient import solve_transient

DATA = Path(__file__).parents[1] / "tests" / "data"
#: What a case fits by default: (parameter, layer).
DEFAULT = {
    "zimt_ions_fast_light": [("mu_anion", 1), ("N_anion", 1)],
    "zimt_traps_photo_light": [("N_t_bulk", 1), ("C_n_bulk", 1), ("mu_p", 1)],
    "zimt_tpv_light": [("N_t_bulk", 1), ("C_p_bulk", 1)],
}


def problem(case: str, fitted, steps: int):
    """The case's residual function of the fitted logarithms (scaled by the curve's scale), its
    Jacobian, and the true logarithms."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    tvg = read_tvg(setup)
    fields = ("t", "Vext", "oc", "G_frac", "track", "dti")
    tvg = dataclasses.replace(tvg, **{k: getattr(tvg, k)[:steps] for k in fields})
    structure, p0 = device_parameters(setup)
    oc = bool(tvg.oc.any())
    vint = oc or float(p0["R_series"]) > 0
    ions = any(float(np.max(p0[k])) > 0 for k in ("N_anion", "N_cation")) and any(
        float(np.max(p0[k])) > 0 for k in ("mu_anion", "mu_cation")
    )

    def curve(theta):
        p = {k: jnp.asarray(v) for k, v in p0.items()}
        for (name, layer), x in zip(fitted, theta, strict=True):
            p[name] = p[name].at[layer].set(jnp.exp(x))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        r = solve_transient(batch, tvg, bernoulli_name="simss", ions=ions, vint=vint)
        return (r.V_int[0] if oc else r.J_ext[0]), r.converged[0]

    true = jnp.log(jnp.array([float(np.asarray(p0[n])[k]) for n, k in fitted]))
    target, ok = jax.jit(curve)(true)
    assert bool(ok.all()), "the true parameters' transient does not converge"
    scale = float(jnp.max(jnp.abs(target))) if not oc else 1.0

    @jax.jit
    def residuals(theta):
        # A step not converged costs a whole scale (as validation/fit.py's voltages).
        y, ok = curve(theta)
        return jnp.where(ok, (y - target) / scale, 1.0)

    return residuals, jax.jit(jax.jacfwd(residuals)), np.asarray(true)


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    args.add_argument("case")
    args.add_argument("--steps", type=int, default=60)
    args.add_argument("--start", type=float, default=2.0)
    args.add_argument("--fit", nargs="*", default=None, help="NAME:LAYER (layers from 0)")
    a = args.parse_args()
    jax.config.update("jax_enable_x64", True)
    fitted = ([(f.split(":")[0], int(f.split(":")[1])) for f in a.fit] if a.fit
              else DEFAULT[a.case])  # fmt: skip
    t0 = time.perf_counter()
    residuals, jacobian, true = problem(a.case, fitted, a.steps)
    signs = np.where(np.arange(len(true)) < (len(true) + 1) // 2, 1.0, -1.0)
    start = true + signs * np.log(a.start)
    calls = {"r": 0, "J": 0}

    def fun(theta):
        calls["r"] += 1
        return np.asarray(residuals(jnp.asarray(theta)))

    def jac(theta):
        calls["J"] += 1
        return np.asarray(jacobian(jnp.asarray(theta)))

    r0 = fun(start)
    jac(start)
    t1 = time.perf_counter()
    out = least_squares(fun, start, jac=jac, x_scale=1.0, xtol=1e-12, ftol=1e-12, gtol=1e-12)
    t2 = time.perf_counter()
    print(f"{a.case}: {a.steps} steps, {len(fitted)} parameters, start x{a.start} off; "
          f"compiled in {t1 - t0:.0f} s, fitted in {t2 - t1:.0f} s "
          f"({calls['r']} curves, {calls['J']} Jacobians)")  # fmt: skip
    print(f"  rms misfit: start {np.sqrt(np.mean(r0**2)):.2e}, end "
          f"{np.sqrt(np.mean(out.fun**2)):.2e} of the scale")  # fmt: skip
    for (name, layer), x, x0, xt in zip(fitted, out.x, start, true, strict=True):
        print(f"  {name:>10} L{layer + 1}: true {np.exp(xt):.6e}, start {np.exp(x0):.3e}, "
              f"fitted {np.exp(x):.6e} ({np.exp(x - xt) - 1:+.1e})")  # fmt: skip


if __name__ == "__main__":
    main()
