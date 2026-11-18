"""The SnO2 half-stack's three measured decays (Kober-Czerny et al. 2025, glass/SnO2/FAPbI3: lit
through the glass at 4.8e11 and 1.5e11 photons/cm², from the film at 4.8e11) fitted with jaxdd, both
ends free.

Seven parameters: the SnO2/FAPbI3 interface's SRV, the transfer velocity over it, SnO2's
conduction band and doping, the perovskite's mobility and deep-trap density, and its traps' hole
capture over their electron capture. Least squares on log10(model/data) in log-spaced time bins,
the model averaged over the 2 ns bins as the measurement (`trpl_common.as_measured`); the
Jacobian by forward differences, every perturbed parameter set in one batch.

    .venv/bin/python validation/trpl_fit.py     # writes validation/trpl_fit.json

docs/validation.md, "TRPL without electrodes".
"""

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares
from trpl_common import (
    FLUENCE,
    REPO,
    T_PULSE,
    C,
    as_measured,
    binned,
    fine_times,
    log_bins,
    pl_of,
    pulse,
    setup,
    sno2_data,
)  # fmt: skip

from jaxdd.batch import device_tree
from jaxdd.device import device_parameters
from jaxdd.transient import solve_transient, tj_columns

MEASURED = (
    (True, FLUENCE),
    (True, FLUENCE / 10**0.5),
    (False, FLUENCE),
)  # (through the glass, fluence)
NAMES = ("log10 S_int (m/s)", "E_c SnO2 (eV)", "log10 N_D SnO2 (m⁻³)", "log10 ν_int (m/s)",
         "log10 μ FAPbI3 (m²/Vs)", "log10 N_deep (m⁻³)", "log10 C_p/C_n")  # fmt: skip
LO = np.array([-6, 4.0, 20, 0, -8, 14, -6])
HI = np.array([3, 4.6, 25.5, 7, -3, 22, 0])
START = np.array([np.log10(0.09), 4.05, 22, 5, np.log10(4.1e-5), np.log10(1.6e4 / C), -3])
FLOOR = 3e-3  # the data below this are noise

_STRUCT = {}


def _base(glass):
    if glass not in _STRUCT:  # the fluence enters by scaling the generation (below)
        _STRUCT[glass] = device_parameters(setup("SnO2", lit_from_glass=glass, fluence=1.0))
    return _STRUCT[glass]


def _params(theta, glass, fluence):
    st, p = _base(glass)
    S, Ec, ND, nu, mu, Nd, rp = theta
    p = dict(p)
    p["N_t_int"] = jnp.asarray(p["N_t_int"]).at[0].set(10.0**S / C)
    p["E_c"] = jnp.asarray(p["E_c"]).at[0].set(Ec)
    p["N_D"] = jnp.asarray(p["N_D"]).at[0].set(10.0**ND)
    for k in ("nu_int_n", "nu_int_p"):
        p[k] = jnp.asarray(p[k]).at[0].set(10.0**nu)
    for k in ("mu_n", "mu_p"):
        p[k] = jnp.asarray(p[k]).at[1].set(10.0**mu)
    p["bulk_trap_N"] = jnp.asarray(p["bulk_trap_N"]).at[1, 0].set(10.0**Nd)
    p["C_p_bulk"] = jnp.asarray(p["C_p_bulk"]).at[1].set(C * 10.0**rp)
    p["gen_G"] = jnp.asarray(p["gen_G"]) * fluence
    return st, p


def simulate(thetas, t):
    """PL after the pulse (sets × measurements × steps) and the steps' convergence."""
    trees = [device_tree(*_params(th, g, f)) for th in thetas for g, f in MEASURED]
    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *trees)
    r = solve_transient(batch, pulse(t), ions=False, vint=False)
    pl = np.stack([pl_of(tj_columns(r, k), 2) for k in range(len(trees))])
    ok = np.asarray(r.converged).reshape(len(thetas), len(MEASURED), -1)
    return pl.reshape(len(thetas), len(MEASURED), -1), ok


class Problem:
    def __init__(self):
        self.t = fine_times()
        self.tm = self.t[1:] - T_PULSE
        self.td, y = sno2_data()
        self.which = log_bins(self.td, self.td.max(), 60)
        yb = binned(y, self.which)
        self.yb = np.where(yb > FLOOR, yb, np.nan)
        self.keep = np.isfinite(self.yb).ravel()
        self.calls = 0

    def residuals_many(self, thetas):
        pl, ok = simulate(thetas, self.t)
        self.calls += len(thetas)
        out = []
        for c in pl:
            model = np.stack([as_measured(self.tm, c[k], self.td) for k in range(len(MEASURED))])
            r = (np.log10(binned(model, self.which)) - np.log10(self.yb)).ravel()[self.keep]
            out.append(r)
        return np.array(out), ok

    def fun(self, theta):
        return self.residuals_many([theta])[0][0]

    def jac(self, theta, h=0.02):
        thetas = [theta] + [theta + h * e for e in np.eye(len(theta))]
        R, _ = self.residuals_many(thetas)
        return ((R[1:] - R[0]) / h).T


def main():
    prob = Problem()
    t0 = time.time()
    res = least_squares(prob.fun, START, jac=prob.jac, bounds=(LO, HI), x_scale=1.0, max_nfev=200)
    rms = float(np.sqrt(np.mean(res.fun**2)))
    pl, ok = simulate([res.x], prob.t)
    print(f"{time.time() - t0:.0f} s, {prob.calls} solves of 3 transients; rms {rms:.4f} decades;"
          f" every step converged: {bool(ok.all())}")  # fmt: skip
    for n, v, lo, hi in zip(NAMES, res.x, LO, HI, strict=True):
        print(f"   {n:24s} {v:8.3f}" + ("   (at its bound)" if min(v - lo, hi - v) < 1e-3 else ""))
    print(
        f"   the interface's SRV: {10 ** res.x[0] * 100:.1f} cm/s (Kober-Czerny et al.: 9 (4–17))"
    )
    model = [as_measured(prob.tm, pl[0, k], prob.td) for k in range(len(MEASURED))]
    # (-1 where a bin has no measured point above the noise, or none at all)
    out = {"theta": res.x.tolist(), "names": NAMES, "rms": rms, "converged": bool(ok.all()),
           "t_bins": [float(prob.td[prob.which == b].mean()) if (prob.which == b).any() else None
                      for b in range(prob.which.max() + 1)],
           "data": np.where(np.isfinite(prob.yb), prob.yb, -1).tolist(),
           "model": np.nan_to_num(binned(np.stack(model), prob.which), nan=-1).tolist(),
           }  # fmt: skip
    path = REPO / "validation" / "trpl_fit.json"
    path.write_text(json.dumps(out))
    print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    main()
