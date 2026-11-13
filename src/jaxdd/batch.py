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
The last is what a J-V curve needs: the current settles before the potential's last digits
(docs/solver.md).
"""

from __future__ import annotations

import contextlib
import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.constants import Q
from jaxdd.device import Device, Structure, device_arrays
from jaxdd.solver import (
    BERNOULLI,
    ION_LOG_CAP,
    ROUND_OFF_FLOOR,
    STALL_STEPS,
    STEP_CAPS,
    _assemble,
    _interface_points,
    _ion_constraints,
    _ion_logs,
    _ion_setup,
    _on_points,
    _poisson_jac,
    _poisson_val,
    _recombination,
    _stencils,
    step_caps,
)
from jaxdd.sweep import block_sweep

__all__ = ["device_tree", "external_voltages", "prepare", "solve_batch"]

#: The current settled: unchanged to this share of its scale (the generation's current plus
#: its own size) over `STALL_STEPS` Newton steps; SimSS's own errJ is larger.
CURRENT_TOL = 1e-5
#: ... while the Newton step is already below this (V_T, relative densities).
CURRENT_STEP_CAP = 1e-3
#: In the quadratic regime (a step below `QUADRATIC_TOL` and below `QUADRATIC_RATIO` of the one
#: before) the error left is about the step squared, so Newton stops there.
QUADRATIC_TOL = 1e-5
QUADRATIC_RATIO = 1e-2


def _device_tree(dev: Device, xp=np) -> dict:
    """One device as the arrays the pure solver reads (`xp` as `device.device_arrays`)."""
    NP = len(dev.x) - 2
    iface = np.zeros(NP + 1, dtype=bool)
    for i in dev.i1[:-1]:
        iface[i] = True
    generated = Q * dev.L * xp.sum(dev.h[:-1] * dev.G[:-1]) / xp.sum(dev.h[:-1])
    return {
        "s": _stencils(dev, xp),
        "ion": _ion_setup(dev),
        "Vt": dev.Vt, "g_frac": dev.G_frac, "L": dev.L,
        "x": dev.x, "h": dev.h, "N_c": dev.N_c, "ni": dev.ni, "E_c": dev.E_c,
        "V0": dev.V0, "VL": dev.VL, "W_L": dev.W_L, "W_R": dev.W_R,
        "ln_n_left": xp.log(dev.n_left), "ln_n_right": xp.log(dev.n_right),
        "ln_p_left": xp.log(dev.p_left), "ln_p_right": xp.log(dev.p_right),
        "offset_n": dev.offset_n, "offset_p": dev.offset_p,
        "mu_n": dev.mu_n, "mu_p": dev.mu_p, "G": dev.G,
        "rec": {"kd": dev.k_direct, "ni2": dev.ni**2, "Nt": dev.srh_Nt, "Cn": dev.srh_Cn,
                "Cp": dev.srh_Cp, "nt0": dev.srh_nt0, "pt0": dev.srh_pt0},
        "iface": iface,
        "J_scale": xp.maximum(generated, 1.0),
        "R_series": dev.R_series,
        "G_shunt": xp.where(dev.R_shunt > 0, 1 / xp.where(dev.R_shunt > 0, dev.R_shunt, 1), 0.0),
        "pre_cond": float(dev.pre_cond), "V_pre": dev.V_pre, "fix_ions": float(dev.fix_ions),
    }  # fmt: skip


def device_tree(structure: Structure, params: dict) -> dict:
    """One device of `prepare`'s batch, in `jax.numpy` from its structure and parameters
    (`device.device_parameters`): differentiable in `params`, and batched by `vmap` over them
    for devices of one structure, e.g.
    `jax.vmap(functools.partial(device_tree, structure))(stacked_params)`."""
    tree = _device_tree(device_arrays(structure, params, jnp), jnp)
    return jax.tree_util.tree_map(jnp.asarray, tree)


def prepare(devices: list[Device]) -> dict:
    """A batch: every leaf of every device stacked along a new first axis."""
    sizes = {len(d.x) for d in devices}
    if len(sizes) != 1:
        raise ValueError(f"a batch needs one grid size, got NP + 2 in {sorted(sizes)}")
    trees = [_device_tree(d) for d in devices]
    with_traps = [t["s"]["ti"] for t in trees if "ti" in t["s"]]
    if with_traps:  # the others with none: zero on every edge
        for t in trees:
            t["s"].setdefault("ti", {k: np.zeros_like(v) for k, v in with_traps[0].items()})
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
    R = _recombination(n, p, d["rec"])
    R_int = (jnp.zeros_like(R), jnp.zeros_like(R))
    if "ti" in d["s"]:
        R_int = _interface_points(n, p, d["s"]["ti"])[:2]
    dx = jnp.diff(x[: NP + 1])

    def carrier(sn, Vg, dens, mu, R_i):
        U = R + R_i - d["G"] * g_frac
        U = U.at[0].set(0.0).at[-1].set(0.0)
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
        # Across an interface SIMsalabim takes the current before it plus the current through
        # the interface traps.
        through = sn * 0.5 * Q * L * h[: NP + 1] * (R_i[: NP + 1] + R_i[1:])
        return jnp.where(d["iface"], jnp.roll(J, 1) + through, J)

    Jn = carrier(-1.0, V + d["offset_n"], n, d["mu_n"], R_int[0])
    Jp = carrier(1.0, V + d["offset_p"], p, d["mu_p"], R_int[1])
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
        args = (_on_points(s), Vt, full[:-2], full[1:-1], full[2:], la, lc)
        F = _poisson_val(V[:-2], V[1:-1], V[2:], *args)
        lo, mid, up = _poisson_jac(V[:-2], V[1:-1], V[2:], *args)
        dV = jax.lax.linalg.tridiagonal_solve(
            lo.at[0].set(0.0), mid, up.at[-1].set(0.0), -F[:, None]
        )[:, 0]
        dV = jnp.clip(dV, -STEP_CAPS[0] * Vt, STEP_CAPS[0] * Vt)
        shift = dV / Vt
        return y + jnp.stack([dV, shift, -shift], axis=-1), jnp.max(jnp.abs(shift)), it + 1

    def cond(carry):
        _, size, it = carry
        return (size > 1e-8) & (it < max_iter)

    y, _, _ = jax.lax.while_loop(cond, body, (y, jnp.inf, 0))
    return y


def _held(ion):
    """1 for a species without a mobile region, whose normalisation is held, else 0."""
    return jnp.where(jnp.stack([ion["a_total"], ion["c_total"]]) > 0, 0.0, 1.0)


def _residual(y, logs, bounds, s, ion, Vt, g_frac, B):
    """Newton's system at `(y, logs)`: the rows' residual (NP × 3) and the ions' constraints,
    a held normalisation's row its own value (an identity row, as in `_system`)."""
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    F = _assemble(full, s, Vt, g_frac, logs[0], logs[1], B)[0]
    return F, _ion_constraints(full[:, 0], logs[0], logs[1], ion) + _held(ion) * logs


def _system(y, logs, bounds, s, ion, Vt, g_frac, B):
    """`_residual` and its Jacobian in `(y, logs)`: `(F, g, blocks)`, the blocks
    `(A, B, C, U, W, D)` of the bordered system `[[T, U], [Wᵀ, D]]` — `T` block tridiagonal
    (`A, B, C`), `U` the rows' derivatives in the normalisations, `W` the constraints' in the
    potentials (column 0 of each point), `D` theirs in the normalisations."""
    la, lc = logs[0], logs[1]
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    F, A, Bb, C, U = _assemble(full, s, Vt, g_frac, la, lc, B)

    def constraints(V_interior, logs_):
        V_full = jnp.concatenate([bounds[0][:1], V_interior, bounds[1][:1]])
        return _ion_constraints(V_full, logs_[0], logs_[1], ion)

    g = constraints(full[1:-1, 0], logs)
    # Two outputs of NP inputs: reverse mode, two passes (forward mode took NP passes and
    # 90 % of a Newton step).
    dg_dV, D = jax.jacrev(constraints, argnums=(0, 1))(full[1:-1, 0], logs)
    D = D + jnp.diag(_held(ion))
    W = jnp.zeros((y.shape[0], 3, 2)).at[:, 0, :].set(dg_dV.T)
    return F, g, (A, Bb, C, U, W, D)


def _bordered_solve(blocks, r, rg):
    """`[[T, U], [Wᵀ, D]]·(x, z) = (r, rg)` (`_system`'s blocks): the block sweep on three
    right-hand sides and a 2×2 Schur complement (Sherman–Morrison–Woodbury, docs/solver.md)."""
    A, Bb, C, U, W, D = blocks
    X = block_sweep(A, Bb, C, jnp.concatenate([r[..., None], U], axis=-1))
    XF, XU = X[..., 0], X[..., 1:]
    schur = D - jnp.einsum("irk,irl->kl", W, XU)
    z = jnp.linalg.solve(schur, rg - jnp.einsum("irk,ir->k", W, XF))
    return XF - XU @ z, z


def _transposed(blocks):
    """The blocks of the transposed system: block tridiagonal again, `Aᵢ ← Cᵢ₋₁ᵀ`, `Bᵢ ← Bᵢᵀ`,
    `Cᵢ ← Aᵢ₊₁ᵀ`, and the border's `U` and `W` exchanged."""
    A, Bb, C, U, W, D = blocks
    zero = jnp.zeros_like(A[:1])
    return (
        jnp.concatenate([zero, jnp.swapaxes(C[:-1], -1, -2)]),
        jnp.swapaxes(Bb, -1, -2),
        jnp.concatenate([jnp.swapaxes(A[1:], -1, -2), zero]),
        W,
        U,
        D.T,
    )


def _apply(blocks, x, z):
    """`[[T, U], [Wᵀ, D]]·(x, z)`."""
    A, Bb, C, U, W, D = blocks
    pad = jnp.zeros_like(x[:1])
    left, right = jnp.concatenate([pad, x[:-1]]), jnp.concatenate([x[1:], pad])
    rows = (jnp.einsum("irc,ic->ir", A, left) + jnp.einsum("irc,ic->ir", Bb, x)
            + jnp.einsum("irc,ic->ir", C, right) + U @ z)  # fmt: skip
    return rows, jnp.einsum("irk,ir->k", W, x) + D @ z


def _newton_solve(y, logs, d, s, ion, Vint, g_frac, B, tol, max_iter, strict=False):
    """Newton at one voltage until converged (module docstring) or `max_iter`:
    `(y, logs, J, converged, iterations)`. Differentiable in everything but the start: the
    derivatives are those of the converged solution (`_converged_jvp`), not of the iterations."""
    guess = jax.lax.stop_gradient((y, logs))
    return _converged(B, max_iter, strict, guess, (d, s, ion, Vint, g_frac), tol)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _converged(B, max_iter, strict, guess, params, tol):
    """`_newton_solve` from `guess`, without derivatives of its own."""
    d, s, ion, Vint, g_frac = params
    return _newton_loop(*guess, d, s, ion, Vint, g_frac, B, tol, max_iter, strict)


@_converged.defjvp
def _converged_jvp(B, max_iter, strict, primals, tangents):
    """Implicit differentiation (M7): at a root of `_residual`, `∂r/∂y·ẏ = −∂r/∂θ·θ̇`. One linear
    solve with Newton's own Jacobian, whatever the iterations were; its transpose, for reverse
    mode, the same sweep on the transposed blocks (`custom_linear_solve`). A voltage not
    converged (or not finite) gets zero derivatives, so that it cannot spoil a gradient summed
    over the curve."""
    guess, params, tol = primals
    out = _converged(B, max_iter, strict, guess, params, tol)
    y, logs, _, done, it = out
    valid = done & jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(logs))
    y_dot, logs_dot, J_dot = _tangent(B, valid, y, logs, params, tangents[1])
    zero = functools.partial(np.zeros, dtype=jax.dtypes.float0)
    return out, (y_dot, logs_dot, J_dot, zero(np.shape(done)), zero(np.shape(it)))


@functools.partial(jax.checkpoint, static_argnums=(0,))
def _tangent(B, valid, y, logs, params, params_dot):
    """The converged solution's and current's tangents. Under `jax.checkpoint`, so that reverse
    mode keeps the solution per voltage, not its Jacobian blocks and the residual's
    linearisation: these are computed again in the backward pass."""
    y = jnp.where(valid, y, 0.0)
    logs = jnp.where(valid, logs, 0.0)

    def residual(params):
        d, s, ion, Vint, g_frac = params
        return _residual(y, logs, _bounds(d, Vint), s, ion, d["Vt"], g_frac, B)

    _, (r, rg) = jax.jvp(residual, (params,), (params_dot,))
    d, s, ion, Vint, g_frac = params
    _, _, blocks = _system(y, logs, _bounds(d, Vint), s, ion, d["Vt"], g_frac, B)
    # An invalid voltage: the identity, whose solve of a zero right-hand side is zero even
    # where its own blocks are not finite.
    eye = jnp.eye(3)
    identity = (0.0, eye, 0.0, 0.0, 0.0, jnp.eye(2))
    blocks = tuple(jnp.where(valid, b, jnp.broadcast_to(i, b.shape))
                   for b, i in zip(blocks, identity, strict=True))  # fmt: skip

    # One flat right-hand side: JAX transposes a linear solve only where every part of it is
    # linear in the tangents, and the constraints' part is a constant zero where no tangent
    # reaches the ions.
    def flat(f, blocks):
        def g(v):
            x, z = f(blocks, v[:-2].reshape(r.shape), v[-2:])
            return jnp.concatenate([x.ravel(), z])

        return g

    sol = jax.lax.custom_linear_solve(
        flat(_apply, blocks),
        -jnp.concatenate([r.ravel(), rg]),
        solve=lambda _, b: flat(_bordered_solve, blocks)(b),
        transpose_solve=lambda _, b: flat(_bordered_solve, _transposed(blocks))(b),
    )
    y_dot, logs_dot = sol[:-2].reshape(r.shape), sol[-2:]
    y_dot = jnp.where(valid, y_dot, 0.0)
    logs_dot = jnp.where(valid, logs_dot, 0.0)

    def current(y, params):
        d, _, _, Vint, g_frac = params
        bounds = _bounds(d, Vint)
        return _current(d, jnp.concatenate([bounds[0][None], y, bounds[1][None]]), g_frac)

    _, J_dot = jax.jvp(current, (y, params), (y_dot, params_dot))
    return y_dot, logs_dot, jnp.where(valid, J_dot, 0.0)


def _newton_loop(y, logs, d, s, ion, Vint, g_frac, B, tol, max_iter, strict):
    """Newton's iterations of `_newton_solve`."""
    bounds = _bounds(d, Vint)
    Vt = d["Vt"]

    def step(y, logs):
        F, g, blocks = _system(y, logs, bounds, s, ion, Vt, g_frac, B)
        delta, d_logs = _bordered_solve(blocks, -F, -g)
        caps = step_caps(Vt)
        stepv = jnp.clip(delta, -caps, caps)
        d_logs = jnp.clip(d_logs, -ION_LOG_CAP, ION_LOG_CAP)
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


def _solve_one(d, voltages, B, tol, max_iter, start_iter, strict):
    """One device's J-V curve: the pre-bias stage, then the scan outwards from it. `voltages`
    ascending; `max_iter` caps Newton per voltage of the scan, `start_iter` Poisson and Newton
    at the pre-bias (the final pass's cap in both passes: a first pass changes no device's
    start, and a low first cap does not fail the pre-bias, which a batch pays once)."""
    s, ion, Vt, g_frac = d["s"], d["ion"], d["Vt"], d["g_frac"]
    first = jnp.where(d["pre_cond"] > 0, d["V_pre"], voltages[0])
    # The start: no derivatives (`_newton_solve` has them, of its converged solution; reverse
    # mode has none through a while loop).
    d0, s0, ion0, first0 = jax.lax.stop_gradient((d, s, ion, first))
    y = _poisson_solve(_initial(d0, first0)[1:-1], d0, s0, ion0, first0, start_iter)
    bounds = _bounds(d0, first0)
    logs = _ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion0)
    y, logs, _, pre_ok, _ = _newton_solve(
        y, logs, d, s, ion, first, g_frac, B, tol, start_iter, strict
    )
    s, ion = _freeze(s, ion, y, logs, Vt, (d["pre_cond"] > 0) & (d["fix_ions"] > 0))

    caps = step_caps(Vt)

    def scan(carry, x):
        # The start: the last two converged solutions extrapolated linearly in the voltage
        # (SIMsalabim's `Extrapolate_Solution`), the change held to one Newton step's caps; at
        # the first voltage of each direction, the pre-bias solution as it is. Unclipped, the
        # extrapolation carries the undetermined logarithm of a density below `n_floor` (30–100
        # apart between neighbouring voltages) into the start, e^100 off (docs/solver.md).
        Vint, turn = x
        carry = jax.tree_util.tree_map(lambda s0, c: jnp.where(turn, s0, c), start, carry)
        y, logs, y_prev, logs_prev, V_last, V_before = carry
        ratio = jnp.where(V_last != V_before, (Vint - V_last) / (V_last - V_before), 0.0)
        guess = y + jnp.clip(ratio * (y - y_prev), -caps, caps)
        logs_guess = logs + jnp.clip(ratio * (logs - logs_prev), -ION_LOG_CAP, ION_LOG_CAP)
        y_new, logs_new, J, ok, it = _newton_solve(
            guess, logs_guess, d, s, ion, Vint, g_frac, B, tol, max_iter, strict
        )
        # Only a converged, finite solution is carried on: one failed voltage costs that
        # voltage, not the rest of the scan.
        keep = ok & jnp.all(jnp.isfinite(y_new)) & jnp.all(jnp.isfinite(logs_new))
        # The carry only starts the next voltage: without derivatives.
        carry = jax.tree_util.tree_map(
            lambda new, old: jnp.where(keep, new, old),
            jax.lax.stop_gradient((y_new, logs_new, y, logs, Vint, V_last)),
            carry,
        )
        return carry, (J, ok, it)

    # Outwards from the pre-bias (docs/solver.md): up from the first voltage at or above it,
    # then, from the pre-bias solution again, down from the last below it. Without a pre-bias
    # `first` is the lowest voltage and the scan runs upwards only.
    n = voltages.shape[0]
    below = jnp.searchsorted(voltages, first)
    j = jnp.arange(n)
    order = jnp.where(j < n - below, below + j, n - 1 - j)
    start = jax.lax.stop_gradient((y, logs, y, logs, first, first))
    _, (J, ok, its) = jax.lax.scan(scan, start, (voltages[order], j == n - below))
    J, ok, its = (jnp.zeros_like(a).at[order].set(a) for a in (J, ok, its))
    return J, ok & pre_ok, its


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "max_iter", "start_iter", "strict"))
def _solve_batch(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict):
    """One pass. `_solve_one` wants the voltages ascending: sorted here, inside the jit (they
    may be traced), and the results put back in the caller's order."""
    B = BERNOULLI[bernoulli_name]
    one = functools.partial(
        _solve_one, B=B, tol=tol, max_iter=max_iter, start_iter=start_iter, strict=strict
    )
    order = jnp.argsort(voltages)
    J, ok, its = jax.vmap(one, in_axes=(0, None))(batch, voltages[order])
    back = jnp.argsort(order)
    # The shunt, at the internal voltage as SimSS has it.
    J = J[:, back] + batch["G_shunt"][:, None] * voltages[None, :]
    return J, ok[:, back], its[:, back]


def external_voltages(batch, voltages, J):
    """The external voltage of each device at each voltage `solve_batch` solved at (the internal
    one), from its curve `J`: `V_ext = V_int + J·R_series` (SimSS's); the voltages themselves
    without a series resistance."""
    return jnp.asarray(voltages)[None, :] + J * batch["R_series"][:, None]


def _stragglers(ok):
    """The devices with a voltage not converged, and the same indices padded to a power of two
    (at most the batch) by repeating the last: few shapes to compile, whatever their number."""
    idx = np.flatnonzero(~np.asarray(ok).all(axis=1))
    if idx.size == 0:
        return idx, idx
    padded = min(len(ok), 1 << (idx.size - 1).bit_length())
    return idx, np.concatenate([idx, np.full(padded - idx.size, idx[-1])])


@contextlib.contextmanager
def _quiet():
    """Without the deprecation warning of Pallas's Triton backend, which the sweep kernel uses
    (jaxdd.sweep, docs/solver.md) and which JAX gives when it compiles it: known, and noise to
    a caller (one whose tests make warnings errors failed on it)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The Pallas Triton backend is deprecated",
                                DeprecationWarning)  # fmt: skip
        yield


def _traced(*trees) -> bool:
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(trees))


def solve_batch(batch, voltages, tol=1e-10, max_iter=60, first_max_iter=6,
                bernoulli_name="exact", strict=False):  # fmt: skip
    """J-V curves of a batch (`prepare`) at `voltages`: `(J, converged, iterations)`, each
    batch × voltages, in the order of `voltages`.

    Each device's scan runs outwards from its pre-bias, up, then down (docs/solver.md); the
    steady state at a voltage does not depend on the way there.
    Two passes (docs/batching.md): every device with at most `first_max_iter` Newton steps per
    voltage, then the devices with a voltage that missed it again, together, with `max_iter`,
    so that a slow device costs the batch `first_max_iter` steps at its voltages, not its own
    number or `max_iter`. Where no device misses it, that is one pass.
    A device that converges in the first pass gets the same result as with `max_iter` alone;
    a re-solved one returns its second pass, and its `iterations` are that pass's. With
    `first_max_iter` None (or not below `max_iter`) one pass; one pass too where `batch` or
    `voltages` are traced (inside `jax.jit`, `jax.jvp`, `jax.vmap`): the stragglers' number,
    which sets the second pass's shape, is known only once the first has run.
    `strict`: Newton only to `tol` (or the round-off floor) — no quadratic stop, no settled
    current; for checking the shortcuts, at about one more step per voltage."""
    with _quiet():
        return _passes(batch, jnp.asarray(voltages, dtype=float), tol, max_iter, first_max_iter,
                       bernoulli_name, strict)  # fmt: skip


def _passes(batch, voltages, tol, max_iter, first_max_iter, bernoulli_name, strict):
    """`solve_batch`'s one or two passes."""
    if first_max_iter is None or first_max_iter >= max_iter or _traced(batch, voltages):
        return _solve_batch(batch, voltages, tol, max_iter, max_iter, bernoulli_name, strict)
    J, ok, its = _solve_batch(
        batch, voltages, tol, first_max_iter, max_iter, bernoulli_name, strict
    )
    idx, padded = _stragglers(ok)
    if idx.size == 0:
        return J, ok, its
    part = jax.tree_util.tree_map(lambda leaf: leaf[padded], batch)
    J2, ok2, its2 = _solve_batch(part, voltages, tol, max_iter, max_iter, bernoulli_name, strict)
    n = idx.size
    return J.at[idx].set(J2[:n]), ok.at[idx].set(ok2[:n]), its.at[idx].set(its2[:n])
