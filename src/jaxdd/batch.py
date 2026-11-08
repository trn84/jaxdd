"""Many devices at once: the solver of `solver.py` as one pure function, `vmap`ped (M3).

Every device of a batch has the same number of grid points (SIMsalabim's `NP`) and the same
voltages; everything else — layers, doping, ions, pre-bias — may differ. A device is a pytree of
arrays (`prepare`); `solve_batch` runs Newton per voltage in a `lax.while_loop`, the voltages
in a `lax.scan`, the pre-bias and the freezing of the ions by masks rather than branches, and the
current in JAX. Under `vmap` a while loop runs until its last lane has converged, so a batch
costs what its slowest device costs (docs/batching.md): converged lanes are held, and the
iteration cap bounds the rest.

Convergence, per device and voltage: the Newton step below `tol`; or the round-off floor (a step
below `ROUND_OFF_FLOOR` that has stopped falling); or the current settled — unchanged to
`CURRENT_TOL` of its scale over `STALL_STEPS` steps while the step is below `CURRENT_STEP_CAP`.
The last is what a J-V curve needs: where strong frozen charge leaves the potential a few 1e-5
`V_T` of noise in a carrier-free bulk, the current is settled long before (docs/solver.md).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.constants import Q
from jaxdd.device import Device
from jaxdd.solver import (
    BERNOULLI,
    ROUND_OFF_FLOOR,
    STALL_STEPS,
    _block_thomas,
    _ion_constraints,
    _ion_logs,
    _ion_setup,
    _kernels,
    _poisson_jac,
    _poisson_val,
    _recombination,
    _stencils,
)

__all__ = ["prepare", "solve_batch"]

#: The current settled: unchanged to this share of its scale (the generation's current plus
#: its own size) over `STALL_STEPS` Newton steps. The noise strong frozen charge leaves is up to
#: 3e-6 of the scale (1e18 cm⁻³ frozen at x_ion 1, l 0.2); SimSS's own errJ is larger.
CURRENT_TOL = 1e-5
#: ... while the Newton step is already below this (V_T, relative densities).
CURRENT_STEP_CAP = 1e-3
#: In the quadratic regime (a step below `QUADRATIC_TOL` and below `QUADRATIC_RATIO` of the one
#: before) the error left is about the step squared, so Newton stops there.
QUADRATIC_TOL = 1e-5
QUADRATIC_RATIO = 1e-2


def _device_tree(dev: Device) -> dict:
    """One device as the arrays the pure solver reads."""
    NP = len(dev.x) - 2
    iface = np.zeros(NP + 1, dtype=bool)
    for i in dev.i1[:-1]:
        iface[i] = True
    generated = Q * dev.L * np.sum(dev.h[:-1] * dev.G[:-1]) / np.sum(dev.h[:-1])
    return {
        "s": _stencils(dev),
        "ion": _ion_setup(dev),
        "Vt": dev.Vt, "g_frac": dev.G_frac, "L": dev.L,
        "x": dev.x, "h": dev.h, "N_c": dev.N_c, "ni": dev.ni, "E_c": dev.E_c,
        "V0": dev.V0, "VL": dev.VL, "W_L": dev.W_L, "W_R": dev.W_R,
        "ln_n_left": np.log(dev.n_left), "ln_n_right": np.log(dev.n_right),
        "ln_p_left": np.log(dev.p_left), "ln_p_right": np.log(dev.p_right),
        "offset_n": dev.offset_n, "offset_p": dev.offset_p,
        "mu_n": dev.mu_n, "mu_p": dev.mu_p, "G": dev.G,
        "rec": {"kd": dev.k_direct, "ni2": dev.ni**2, "Nt": dev.srh_Nt, "Cn": dev.srh_Cn,
                "Cp": dev.srh_Cp, "nt0": dev.srh_nt0, "pt0": dev.srh_pt0},
        "iface": iface,
        "J_scale": max(generated, 1.0),
        "pre_cond": float(dev.pre_cond), "V_pre": dev.V_pre, "fix_ions": float(dev.fix_ions),
    }  # fmt: skip


def prepare(devices: list[Device]) -> dict:
    """A batch: every leaf of every device stacked along a new first axis."""
    sizes = {len(d.x) for d in devices}
    if len(sizes) != 1:
        raise ValueError(f"a batch needs one grid size, got NP + 2 in {sorted(sizes)}")
    trees = [_device_tree(d) for d in devices]
    return jax.tree_util.tree_map(lambda *leaves: jnp.asarray(np.stack(leaves)), *trees)


def _bounds(d, Vint):
    return jnp.array([
        [d["V0"] - 0.5 * Vint, d["ln_n_left"], d["ln_p_left"]],
        [d["VL"] + 0.5 * Vint, d["ln_n_right"], d["ln_p_right"]],
    ])  # fmt: skip


def _initial(d, Vint):
    """SIMsalabim's `Init_Pot_Dens_Ions_Traps`: linear potential, linear Fermi level."""
    x = d["x"]
    V = d["V0"] - 0.5 * Vint + x * (d["VL"] - d["V0"] + Vint) / d["L"]
    EF = d["W_L"] - x * (d["W_L"] - d["W_R"]) / d["L"]
    ln_n = jnp.log(d["N_c"]) - (EF - d["E_c"]) / d["Vt"]
    ln_p = 2 * jnp.log(d["ni"]) - ln_n
    return jnp.stack([V, ln_n, ln_p], -1)


def _current(d, full, g_frac):
    """SIMsalabim's `Calc_Curr_Int` for both carriers, averaged (`Calc_All_Currents`)."""
    V, n, p = full[:, 0], jnp.exp(full[:, 1]), jnp.exp(full[:, 2])
    Vt, x, h, L = d["Vt"], d["x"], d["h"], d["L"]
    NP = x.shape[0] - 2
    U = _recombination(n, p, d["rec"]) - d["G"] * g_frac
    U = U.at[0].set(0.0).at[-1].set(0.0)
    dx = jnp.diff(x[: NP + 1])

    def carrier(sn, Vg, dens, mu):
        ev = jnp.exp(sn * Vg[: NP + 1] / Vt)
        int_u = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * dx * (U[:NP] + U[1 : NP + 1]))])
        single = jnp.sum(0.5 * dx * (ev[:-1] / mu[:NP] + ev[1:] / mu[1 : NP + 1]))
        double = jnp.sum(
            0.5 * dx * (ev[:-1] * int_u[:-1] / mu[:NP] + ev[1:] * int_u[1:] / mu[1 : NP + 1])
        )
        K = (Vt * (sn * dens[NP + 1] * jnp.exp(sn * Vg[NP + 1] / Vt)
                   - sn * dens[0] * jnp.exp(sn * Vg[0] / Vt)) - sn * double) / single  # fmt: skip
        J = Q * K + jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(sn * Q * L * h[1 : NP + 1] * U[1 : NP + 1])]
        )
        # Across an interface SIMsalabim takes the current before it (no interface traps).
        return jnp.where(d["iface"], jnp.roll(J, 1), J)

    Jn = carrier(-1.0, V + d["offset_n"], n, d["mu_n"])
    Jp = carrier(1.0, V + d["offset_p"], p, d["mu_p"])
    hh = h[: NP + 1]
    return jnp.sum(hh * (Jn + Jp)) / jnp.sum(hh)


def _poisson_solve(y, d, s, ion, Vint, max_iter):
    """Poisson's equation alone, the densities following the potential (`solver._poisson_step`
    in a while loop)."""
    bounds = _bounds(d, Vint)
    Vt = d["Vt"]

    def body(carry):
        y, _, it = carry
        full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
        V = full[:, 0]
        la, lc = _ion_logs(V, ion)
        args = (s, Vt, full[:-2], full[1:-1], full[2:], la, lc)
        F = _poisson_val(V[:-2], V[1:-1], V[2:], *args)
        lo, mid, up = _poisson_jac(V[:-2], V[1:-1], V[2:], *args)
        dV = jax.lax.linalg.tridiagonal_solve(
            lo.at[0].set(0.0), mid, up.at[-1].set(0.0), -F[:, None]
        )[:, 0]
        dV = jnp.clip(dV, -10.0 * Vt, 10.0 * Vt)
        shift = dV / Vt
        return y + jnp.stack([dV, shift, -shift], axis=-1), jnp.max(jnp.abs(shift)), it + 1

    def cond(carry):
        _, size, it = carry
        return (size > 1e-8) & (it < max_iter)

    y, _, _ = jax.lax.while_loop(cond, body, (y, jnp.inf, 0))
    return y


def _newton_solve(y, logs, d, s, ion, Vint, g_frac, B, tol, max_iter, strict=False):
    """Newton at one voltage until converged (module docstring) or `max_iter`."""
    row_val, row_jac, row_ions = _kernels(B)
    bounds = _bounds(d, Vint)
    Vt = d["Vt"]

    def step(y, logs):
        la, lc = logs[0], logs[1]
        full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
        left, mid, right = full[:-2], full[1:-1], full[2:]
        F = row_val(left, mid, right, s, Vt, g_frac, la, lc)
        A, Bb, C = row_jac(left, mid, right, s, Vt, g_frac, la, lc)
        U = jnp.stack(row_ions(left, mid, right, s, Vt, g_frac, la, lc), axis=-1)

        def constraints(V_interior, logs_):
            V_full = jnp.concatenate([bounds[0][:1], V_interior, bounds[1][:1]])
            return _ion_constraints(V_full, logs_[0], logs_[1], ion)

        g = constraints(full[1:-1, 0], logs)
        # Two outputs of NP inputs: reverse mode, two passes (forward mode took NP passes and
        # 90 % of a Newton step).
        dg_dV, D = jax.jacrev(constraints, argnums=(0, 1))(full[1:-1, 0], logs)
        D = D + jnp.diag(jnp.where(jnp.stack([ion["a_total"], ion["c_total"]]) > 0, 0.0, 1.0))
        W = jnp.zeros((y.shape[0], 3, 2)).at[:, 0, :].set(dg_dV.T)
        X = _block_thomas(A, Bb, C, jnp.concatenate([-F[..., None], U], axis=-1))
        XF, XU = X[..., 0], X[..., 1:]
        schur = D - jnp.einsum("irk,irl->kl", W, XU)
        d_logs = jnp.linalg.solve(schur, -g - jnp.einsum("irk,ir->k", W, XF))
        delta = XF - XU @ d_logs
        caps = jnp.array([10.0, 5.0, 5.0]) * jnp.array([Vt, 1.0, 1.0])
        stepv = jnp.clip(delta, -caps, caps)
        d_logs = jnp.clip(d_logs, -5.0, 5.0)
        new = y + stepv
        dens = jnp.exp(new[:, 1:])
        weight = dens / (dens + s["n_floor"][:, None])
        size = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(stepv[:, 0])) / Vt, jnp.max(jnp.abs(stepv[:, 1:]) * weight)
            ),
            jnp.max(jnp.abs(d_logs)),
        )
        return new, logs + d_logs, size

    def current(y):
        return _current(d, jnp.concatenate([bounds[0][None], y, bounds[1][None]]), g_frac)

    hist0 = jnp.full(STALL_STEPS + 1, jnp.inf)
    J0 = jnp.full(STALL_STEPS + 1, jnp.inf)

    def converged(size_hist, J_hist, it):
        size = size_hist[-1]
        stalled = (
            (it > STALL_STEPS)
            & (jnp.max(size_hist) < ROUND_OFF_FLOOR)
            & (jnp.min(size_hist[1:]) >= 0.1 * size_hist[0])
        )
        J_now = J_hist[-1]
        scale = d["J_scale"] + jnp.abs(J_now)
        settled = (
            (it > STALL_STEPS)
            & (size < CURRENT_STEP_CAP)
            & (jnp.max(jnp.abs(J_hist - J_now)) < CURRENT_TOL * scale)
        )
        # Newton converges quadratically: a step that is small and a hundredth of the one
        # before leaves an error of about its square (1e-10 at 1e-5), so the next step, which
        # would only confirm it, is not taken.
        prev = size_hist[-2]
        quadratic = (it > 1) & (size < QUADRATIC_TOL) & (size < QUADRATIC_RATIO * prev)
        if strict:  # only the step below `tol`, or the round-off floor
            return (size < tol) | stalled
        return (size < tol) | stalled | settled | quadratic

    def body(carry):
        y, logs, size_hist, J_hist, it, _ = carry
        y, logs, size = step(y, logs)
        size_hist = jnp.roll(size_hist, -1).at[-1].set(size)
        J_hist = jnp.roll(J_hist, -1).at[-1].set(current(y))
        it = it + 1
        return y, logs, size_hist, J_hist, it, converged(size_hist, J_hist, it)

    def cond(carry):
        *_, it, done = carry
        return (~done) & (it < max_iter)

    y, logs, _, J_hist, it, done = jax.lax.while_loop(
        cond, body, (y, logs, hist0, J0, 0, jnp.array(False))
    )
    return y, logs, J_hist[-1], done, it


def _freeze(s, ion, y, logs, Vt, flag):
    """With `flag`, the mobile ions of `y` as fixed densities (`solver._freeze`)."""
    V = y[:, 0]
    anion = s["a_fix"] + s["a_mask"] * jnp.exp(logs[0] + V / Vt)
    cation = s["c_fix"] + s["c_mask"] * jnp.exp(logs[1] - V / Vt)
    s = dict(s)
    s["a_fix"] = jnp.where(flag, anion, s["a_fix"])
    s["c_fix"] = jnp.where(flag, cation, s["c_fix"])
    s["a_mask"] = jnp.where(flag, 0.0, s["a_mask"])
    s["c_mask"] = jnp.where(flag, 0.0, s["c_mask"])
    ion = dict(ion)
    ion["a_mask"] = jnp.where(flag, 0.0, ion["a_mask"])
    ion["c_mask"] = jnp.where(flag, 0.0, ion["c_mask"])
    ion["a_total"] = jnp.where(flag, 0.0, ion["a_total"])
    ion["c_total"] = jnp.where(flag, 0.0, ion["c_total"])
    return s, ion


def _solve_one(d, voltages, B, tol, max_iter, strict):
    """One device's J-V curve: the pre-bias stage, then the scan."""
    s, ion, Vt, g_frac = d["s"], d["ion"], d["Vt"], d["g_frac"]
    first = jnp.where(d["pre_cond"] > 0, d["V_pre"], voltages[0])
    y = _initial(d, first)[1:-1]
    y = _poisson_solve(y, d, s, ion, first, max_iter)
    bounds = _bounds(d, first)
    logs = _ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion)
    y, logs, _, pre_ok, _ = _newton_solve(
        y, logs, d, s, ion, first, g_frac, B, tol, max_iter, strict
    )
    s, ion = _freeze(s, ion, y, logs, Vt, (d["pre_cond"] > 0) & (d["fix_ions"] > 0))

    def scan(carry, Vint):
        # The start: the last two solutions extrapolated linearly in the voltage (SIMsalabim's
        # `Extrapolate_Solution`); at the first voltage, the pre-bias solution as it is.
        y, logs, y_prev, logs_prev, V_last, V_before = carry
        ratio = jnp.where(V_last != V_before, (Vint - V_last) / (V_last - V_before), 0.0)
        guess = y + ratio * (y - y_prev)
        logs_guess = logs + ratio * (logs - logs_prev)
        y_new, logs_new, J, ok, it = _newton_solve(
            guess, logs_guess, d, s, ion, Vint, g_frac, B, tol, max_iter, strict
        )
        return (y_new, logs_new, y, logs, Vint, V_last), (J, ok, it)

    start = (y, logs, y, logs, first, first)
    _, (J, ok, its) = jax.lax.scan(scan, start, voltages)
    return J, ok & pre_ok, its


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "max_iter", "strict"))
def _solve_batch(batch, voltages, tol, max_iter, bernoulli_name, strict):
    B = BERNOULLI[bernoulli_name]
    one = functools.partial(_solve_one, B=B, tol=tol, max_iter=max_iter, strict=strict)
    return jax.vmap(one, in_axes=(0, None))(batch, voltages)


def solve_batch(batch, voltages, tol=1e-10, max_iter=60, bernoulli_name="exact", strict=False):
    """J-V curves of a batch (`prepare`) at `voltages`: `(J, converged, iterations)`, each
    batch × voltages.
    `strict`: Newton only to `tol` (or the round-off floor) — no quadratic stop, no settled
    current; for checking the shortcuts, at about one more step per voltage."""
    return _solve_batch(batch, voltages, tol, max_iter, bernoulli_name, strict)
