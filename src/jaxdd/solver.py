"""Steady state of one device by a coupled Newton method, and its J-V curve.

The discrete equations are SIMsalabim 5.36's (`Units/DDRoutines.pas`), so that a converged
solution is SIMsalabim's to its tolerance:

* Poisson (`Solve_Poisson`): `fac_l·V[i−1] + fac_m·V[i] + fac_u·V[i+1] = q·(n − p + N_A − N_D)`,
  with SIMsalabim's coefficients for a permittivity that jumps between layers.
* Continuity (`Cont_Eq_Elec`, `Cont_Eq_Holes`): Scharfetter–Gummel in the generalised
  potentials `Vgn = V + E_c − E_c[0] + V_T·ln(N_c/N_c[0])` (and `Vgp`), which carry the band
  offsets; at an interface the mobility of the interval across it is `L·h·ν_int/V_T`
  (`Calc_Elec_Mob`), so the transfer is thermionic at `ν_int`.
* Recombination: direct `k_direct·(n·p − n_i²)` and bulk SRH through one neutral level.
* Current (`Calc_Curr_Int`, `currDiffInt = 2`): the integral form of De Mari, averaged over
  the device (`Calc_All_Currents`).

What differs is only how they are solved: SIMsalabim iterates Poisson and the two continuity
equations in turn (Gummel); here all three are solved at once, Newton's method on the
unknowns `(V, ln n, ln p)` at every interior point. The Jacobian is block-tridiagonal with
3×3 blocks; each row's three blocks come from `jax.jacfwd` of that row's residual in its
three neighbours (PLAN.md §3.2), and the system is solved by a block Thomas sweep.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.constants import Q
from jaxdd.device import Device

jax.config.update("jax_enable_x64", True)

__all__ = ["JVResult", "bernoulli", "solve_jv"]

#: Densities below this share of the largest `N_c` do not count for convergence (m⁻³ there).
DENSITY_FLOOR = 1e-12

#: Below |x| < this, `B(x)` by its series: the full expression is 0/0 at 0.
_SERIES_BELOW = 1e-2


def bernoulli(x):
    """`B(x) = x/(eˣ − 1)`, finite with finite derivatives everywhere.

    The "double where": the full expression is evaluated at a safe argument where the series
    is used, so neither the value nor its derivative sees 0/0.
    """
    small = jnp.abs(x) < _SERIES_BELOW
    safe = jnp.where(small, 1.0, x)
    full = safe / jnp.expm1(safe)
    series = 1.0 - x / 2.0 + x * x / 12.0 - x**4 / 720.0
    return jnp.where(small, series, full)


def bernoulli_simss(x):
    """SIMsalabim 5.36's `B` (`Units/DDRoutines.pas`): Taylor of order 1 below |x| 0.05, 2 below
    0.45, 4 below 1.5, the full expression above. Its error reaches 2e-4 relative at the
    first switch and 8e-4 at the last; used to reproduce SIMsalabim's numbers, not as the
    better function."""
    ax = jnp.abs(x)
    safe = jnp.where(ax < 1.5, 2.0, x)
    full = safe / jnp.expm1(safe)
    order1 = 1.0 - 0.5 * x
    order2 = 1.0 - x * (0.5 - x / 12.0)
    order4 = 1.0 + x * (-0.5 + x * (1.0 / 12.0 - x * x / 720.0))
    return jnp.where(
        ax < 0.05, order1, jnp.where(ax < 0.45, order2, jnp.where(ax < 1.5, order4, full))
    )


#: Which Bernoulli function the discretisation uses: `"exact"`, or `"simss"` to reproduce
#: SIMsalabim's approximation (its numbers to round-off rather than to 1e-4).
BERNOULLI = {"exact": bernoulli, "simss": bernoulli_simss}


@dataclass(frozen=True)
class JVResult:
    V: np.ndarray
    J: np.ndarray  # A/m², SIMsalabim's sign: photocurrent negative
    converged: np.ndarray
    iterations: np.ndarray


def _stencils(dev: Device) -> dict[str, jnp.ndarray]:
    """Every static quantity row i (1..NP) needs, as arrays over the interior points."""
    NP = len(dev.x) - 2
    i = np.arange(1, NP + 1)
    h, L = dev.h, dev.L
    eps = dev.eps
    fac_m = (eps[i] - eps[i - 1]) / (h[i - 1] ** 2 * L**2) - 2 * eps[i] * (
        1 / (h[i] * (h[i] + h[i - 1]) * L**2) + 1 / (h[i - 1] * (h[i] + h[i - 1]) * L**2)
    )
    fac_l = 2 * eps[i] / (h[i - 1] * (h[i] + h[i - 1]) * L**2) - (eps[i] - eps[i - 1]) / (
        h[i - 1] ** 2 * L**2
    )
    fac_u = 2 * eps[i] / (h[i] * (h[i] + h[i - 1]) * L**2)
    return {
        "fac_l": fac_l, "fac_m": fac_m, "fac_u": fac_u,
        "hm": h[i - 1], "hp": h[i],
        "fac": 0.5 * L**2 * h[i] * h[i - 1] * (h[i] + h[i - 1]),
        "mun_m": dev.mu_n[i - 1], "mun_p": dev.mu_n[i],
        "mup_m": dev.mu_p[i - 1], "mup_p": dev.mu_p[i],
        "offn": np.stack([dev.offset_n[i - 1], dev.offset_n[i], dev.offset_n[i + 1]], -1),
        "offp": np.stack([dev.offset_p[i - 1], dev.offset_p[i], dev.offset_p[i + 1]], -1),
        "dop": dev.N_A[i] - dev.N_D[i],
        "kd": dev.k_direct[i], "ni2": dev.ni[i] ** 2,
        "Nt": dev.srh_Nt[i], "Cn": dev.srh_Cn[i], "Cp": dev.srh_Cp[i],
        "nt0": dev.srh_nt0[i], "pt0": dev.srh_pt0[i],
        "G": dev.G[i],
        "n_floor": np.full(NP, DENSITY_FLOOR * dev.N_c.max()),
        # Row scales: each equation over the size of its terms, for a well-conditioned solve.
        "sP": 1.0 / (Q * dev.N_c[i]),
        "sC": 1.0 / (np.maximum(dev.mu_n[i], dev.mu_p[i]) * dev.Vt * dev.N_c[i] * h[i]),
    }  # fmt: skip


def _recombination(n, p, s):
    srh = s["Nt"] * s["Cn"] * s["Cp"] * (n * p - s["nt0"] * s["pt0"])
    srh = srh / jnp.where(s["Nt"] > 0, s["Cn"] * (n + s["nt0"]) + s["Cp"] * (p + s["pt0"]), 1.0)
    return s["kd"] * (n * p - s["ni2"]) + jnp.where(s["Nt"] > 0, srh, 0.0)


def _row(left, mid, right, s, Vt, g_frac, B=bernoulli):
    """The three residuals of one interior point from `(V, ln n, ln p)` at it and its two
    neighbours."""
    V = jnp.stack([left[0], mid[0], right[0]])
    n = jnp.exp(jnp.stack([left[1], mid[1], right[1]]))
    p = jnp.exp(jnp.stack([left[2], mid[2], right[2]]))
    poisson = (
        s["fac_l"] * V[0] + s["fac_m"] * V[1] + s["fac_u"] * V[2] - Q * (n[1] - p[1] + s["dop"])
    )
    gn = (V + s["offn"]) / Vt
    gp = (V + s["offp"]) / Vt
    lo_n = s["hp"] * s["mun_m"] * Vt * B(gn[0] - gn[1])
    up_n = s["hm"] * s["mun_p"] * Vt * B(gn[2] - gn[1])
    m_n = -(s["hm"] * s["mun_p"] * Vt * B(gn[1] - gn[2])
            + s["hp"] * s["mun_m"] * Vt * B(gn[1] - gn[0]))  # fmt: skip
    lo_p = s["hp"] * s["mup_m"] * Vt * B(gp[1] - gp[0])
    up_p = s["hm"] * s["mup_p"] * Vt * B(gp[1] - gp[2])
    m_p = -(s["hm"] * s["mup_p"] * Vt * B(gp[2] - gp[1])
            + s["hp"] * s["mup_m"] * Vt * B(gp[0] - gp[1]))  # fmt: skip
    net = s["G"] * g_frac - _recombination(n[1], p[1], s)
    elec = lo_n * n[0] + m_n * n[1] + up_n * n[2] + s["fac"] * net
    hole = lo_p * p[0] + m_p * p[1] + up_p * p[2] + s["fac"] * net
    return jnp.stack([poisson * s["sP"], elec * s["sC"], hole * s["sC"]])


def _kernels(B):
    """The batched residual and Jacobian blocks of `_row` with Bernoulli function `B`."""
    row = functools.partial(_row, B=B)
    return (
        jax.vmap(row, in_axes=(0, 0, 0, 0, None, None)),
        jax.vmap(jax.jacfwd(row, argnums=(0, 1, 2)), in_axes=(0, 0, 0, 0, None, None)),
    )


def _block_thomas(A, B, C, d):
    """Solve the block-tridiagonal system `A[i]·x[i−1] + B[i]·x[i] + C[i]·x[i+1] = d[i]`."""

    def forward(carry, blocks):
        c_prev, d_prev = carry
        a, b, c, r = blocks
        m = b - a @ c_prev
        c_new = jnp.linalg.solve(m, c)
        d_new = jnp.linalg.solve(m, r - a @ d_prev)
        return (c_new, d_new), (c_new, d_new)

    zero_c = jnp.zeros_like(C[0])
    zero_d = jnp.zeros_like(d[0])
    _, (cp, dp) = jax.lax.scan(forward, (zero_c, zero_d), (A, B, C, d))

    def backward(x_next, blocks):
        c, r = blocks
        x = r - c @ x_next
        return x, x

    _, x = jax.lax.scan(backward, zero_d, (cp, dp), reverse=True)
    return x


@functools.partial(jax.jit, static_argnames="bernoulli_name")
def _newton_step(y, bounds, s, Vt, g_frac, bernoulli_name="exact"):
    """One damped Newton step on the interior unknowns `y` (NP × 3)."""
    row_val, row_jac = _kernels(BERNOULLI[bernoulli_name])
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    left, mid, right = full[:-2], full[1:-1], full[2:]
    F = row_val(left, mid, right, s, Vt, g_frac)
    A, B, C = row_jac(left, mid, right, s, Vt, g_frac)
    delta = _block_thomas(A, B, C, -F)
    # Damping per point, as SIMsalabim limits its potential update (maxDelV): no potential
    # step past 10 V_T and no density step past a factor e^5.
    caps = jnp.array([10.0 * Vt, 5.0, 5.0])
    step = jnp.clip(delta, -caps, caps)
    new = y + step
    # Converged by the potential in V_T and the densities relatively — a density far below any
    # that matters (n_floor) is weighted out: its logarithm is round-off at the end.
    dens = jnp.exp(new[:, 1:])
    weight = dens / (dens + s["n_floor"][:, None])
    size = jnp.maximum(jnp.max(jnp.abs(step[:, 0])) / Vt, jnp.max(jnp.abs(step[:, 1:]) * weight))
    return new, size


def _poisson_row(left, mid, right, s, Vt, ref_left, ref_mid, ref_right):
    """Poisson's residual with the quasi-Fermi levels held: `n ∝ exp(V/V_T)`, `p ∝ exp(−V/V_T)`
    from the reference state, as SIMsalabim's `Solve_Poisson` moves them."""
    shift = jnp.stack([left - ref_left[0], mid - ref_mid[0], right - ref_right[0]]) / Vt
    ln_n = jnp.stack([ref_left[1], ref_mid[1], ref_right[1]]) + shift
    ln_p = jnp.stack([ref_left[2], ref_mid[2], ref_right[2]]) - shift
    V = jnp.stack([left, mid, right])
    n, p = jnp.exp(ln_n[1]), jnp.exp(ln_p[1])
    poisson = s["fac_l"] * V[0] + s["fac_m"] * V[1] + s["fac_u"] * V[2] - Q * (n - p + s["dop"])
    return poisson * s["sP"]


_poisson_jac = jax.vmap(
    jax.jacfwd(_poisson_row, argnums=(0, 1, 2)), in_axes=(0, 0, 0, 0, None, 0, 0, 0)
)
_poisson_val = jax.vmap(_poisson_row, in_axes=(0, 0, 0, 0, None, 0, 0, 0))


@jax.jit
def _poisson_step(y, bounds, s, Vt):
    """One Newton step of Poisson's equation alone; the densities follow the potential."""
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    V = full[:, 0]
    F = _poisson_val(V[:-2], V[1:-1], V[2:], s, Vt, full[:-2], full[1:-1], full[2:])
    lo, mid, up = _poisson_jac(V[:-2], V[1:-1], V[2:], s, Vt, full[:-2], full[1:-1], full[2:])
    dV = jax.lax.linalg.tridiagonal_solve(lo.at[0].set(0.0), mid, up.at[-1].set(0.0), -F[:, None])[
        :, 0
    ]
    dV = jnp.clip(dV, -10.0 * Vt, 10.0 * Vt)
    shift = dV / Vt
    new = y + jnp.stack([dV, shift, -shift], axis=-1)
    return new, jnp.max(jnp.abs(shift))


def _current(dev: Device, V, n, p, g_frac) -> float:
    """SIMsalabim's `Calc_Curr_Int` for both carriers, averaged as `Calc_All_Currents`."""
    Vt, x, h, L = dev.Vt, dev.x, dev.h, dev.L
    NP = len(x) - 2
    s_all = {key: np.zeros(NP + 2) for key in ("kd", "ni2", "Nt", "Cn", "Cp", "nt0", "pt0")}
    s_all.update(kd=dev.k_direct, ni2=dev.ni**2, Nt=dev.srh_Nt, Cn=dev.srh_Cn,
                 Cp=dev.srh_Cp, nt0=dev.srh_nt0, pt0=dev.srh_pt0)  # fmt: skip
    R = np.asarray(_recombination(n, p, s_all))
    U = R - dev.G * g_frac
    U[0] = U[-1] = 0.0

    def carrier(sn, Vg, dens, mu):
        dx = np.diff(x[: NP + 1])
        ev = np.exp(sn * Vg[: NP + 1] / Vt)
        int_u = np.concatenate([[0.0], np.cumsum(0.5 * dx * (U[:NP] + U[1 : NP + 1]))])
        single = np.sum(0.5 * dx * (ev[:-1] / mu[:NP] + ev[1:] / mu[1 : NP + 1]))
        double = np.sum(
            0.5 * dx * (ev[:-1] * int_u[:-1] / mu[:NP] + ev[1:] * int_u[1:] / mu[1 : NP + 1])
        )
        K = (Vt * (sn * dens[NP + 1] * np.exp(sn * Vg[NP + 1] / Vt)
                   - sn * dens[0] * np.exp(sn * Vg[0] / Vt)) - sn * double) / single  # fmt: skip
        J = np.zeros(NP + 1)
        J[0] = Q * K
        J[1:] = Q * K + np.cumsum(sn * Q * L * h[1 : NP + 1] * U[1 : NP + 1])
        # SIMsalabim then sets the current across each interface to the one before it (the
        # current through interface traps, none here); the points after keep their values.
        for i in dev.i1[:-1]:
            J[i] = J[i - 1]
        return J

    Jn = carrier(-1, V + dev.offset_n, n, dev.mu_n)
    Jp = carrier(1, V + dev.offset_p, p, dev.mu_p)
    total = Jn + Jp
    return float(np.sum(h[: NP + 1] * total) / np.sum(h[: NP + 1]))


def _initial(dev: Device, Vint: float):
    """SIMsalabim's `Init_Pot_Dens_Ions_Traps`: linear potential, linear Fermi level."""
    x, L = dev.x, dev.L
    V = dev.V0 - 0.5 * Vint + x * (dev.VL - dev.V0 + Vint) / L
    EF = dev.W_L - x * (dev.W_L - dev.W_R) / L
    n = dev.N_c * np.exp(-(EF - dev.E_c) / dev.Vt)
    p = dev.ni**2 / n
    return np.stack([V, np.log(n), np.log(p)], -1)


def solve_jv(dev: Device, voltages: np.ndarray, g_frac: float | None = None,
             tol: float = 1e-10, max_iter: int = 200,
             bernoulli_name: str = "exact") -> JVResult:  # fmt: skip
    """The J-V curve at `voltages` (internal = external here), each solved from the last."""
    g_frac = dev.G_frac if g_frac is None else g_frac
    s = {k: jnp.asarray(v) for k, v in _stencils(dev).items()}
    y = jnp.asarray(_initial(dev, float(voltages[0]))[1:-1])
    J, conv, its = [], [], []
    for Vint in voltages:
        bounds = jnp.array([
            [dev.V0 - 0.5 * Vint, np.log(dev.n_left), np.log(dev.p_left)],
            [dev.VL + 0.5 * Vint, np.log(dev.n_right), np.log(dev.p_right)],
        ])  # fmt: skip
        if Vint == voltages[0]:
            for _ in range(max_iter):
                y, size = _poisson_step(y, bounds, s, dev.Vt)
                if float(size) < 1e-8:
                    break
        ok = False
        it = 0
        while it < max_iter:
            it += 1
            y, size = _newton_step(y, bounds, s, dev.Vt, g_frac, bernoulli_name)
            if float(size) < tol:
                ok = True
                break
        full = np.concatenate(
            [np.asarray(bounds[0])[None], np.asarray(y), np.asarray(bounds[1])[None]]
        )
        J.append(_current(dev, full[:, 0], np.exp(full[:, 1]), np.exp(full[:, 2]), g_frac))
        conv.append(ok)
        its.append(it)
    return JVResult(np.asarray(voltages), np.array(J), np.array(conv), np.array(its))
