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
3×3 blocks, assembled from each interval's fluxes and their derivatives (`_assemble`; `_row`
and `jax.jacfwd` of it are the reference it is tested against, docs/solver.md), and the
system is solved by a block Thomas sweep.
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

#: A Newton step below this (V_T, relative densities) that no longer falls is round-off: the
#: solution, three orders below SIMsalabim's own `tolDens` of 1e-5.
ROUND_OFF_FLOOR = 1e-6
STALL_STEPS = 3

#: Densities below this share of the largest `N_c` do not count for convergence (m⁻³ there).
DENSITY_FLOOR = 1e-12

#: The largest change of an unknown in one Newton step, per point: the potential in V_T (as
#: SIMsalabim's `maxDelV`), `ln n` and `ln p` (a factor e^5). The batched scan's extrapolated
#: start is held to the same caps (`batch._solve_one`, docs/solver.md).
STEP_CAPS = (10.0, 5.0, 5.0)
#: The same for the ions' normalisation logarithms.
ION_LOG_CAP = 5.0


def step_caps(Vt):
    """`STEP_CAPS` in the unknowns' units: `(V, ln n, ln p)`."""
    return jnp.array(STEP_CAPS) * jnp.array([Vt, 1.0, 1.0])


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
    V: np.ndarray  # the internal voltage
    J: np.ndarray  # A/m², SIMsalabim's sign: photocurrent negative; with the shunt's current
    converged: np.ndarray
    iterations: np.ndarray
    V_ext: np.ndarray | None = None  # V + J·R_series


def _stencils(dev: Device, xp=np) -> dict[str, jnp.ndarray]:
    """Every static quantity row i (1..NP) needs, as arrays over the interior points; in NumPy,
    or in `jax.numpy` for a device built there (`device.device_arrays`)."""
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
    out = {
        "fac_l": fac_l, "fac_m": fac_m, "fac_u": fac_u,
        "hm": h[i - 1], "hp": h[i],
        "fac": 0.5 * L**2 * h[i] * h[i - 1] * (h[i] + h[i - 1]),
        "mun_m": dev.mu_n[i - 1], "mun_p": dev.mu_n[i],
        "mup_m": dev.mu_p[i - 1], "mup_p": dev.mu_p[i],
        "offn": xp.stack([dev.offset_n[i - 1], dev.offset_n[i], dev.offset_n[i + 1]], -1),
        "offp": xp.stack([dev.offset_p[i - 1], dev.offset_p[i], dev.offset_p[i + 1]], -1),
        "dop": dev.N_A[i] - dev.N_D[i],
        "kd": dev.k_direct[i], "ni2": dev.ni[i] ** 2,
        "Nt": dev.srh_Nt[i], "Cn": dev.srh_Cn[i], "Cp": dev.srh_Cp[i],
        "nt0": dev.srh_nt0[i], "pt0": dev.srh_pt0[i],
        "G": dev.G[i],
        # Ions: a mobile region's mask (Boltzmann in V with a normalisation solved for) and
        # fixed densities (immobile, or frozen after a pre-bias).
        "a_mask": dev.anion_mask[i].astype(float), "c_mask": dev.cation_mask[i].astype(float),
        "a_fix": dev.anion_fixed[i], "c_fix": dev.cation_fixed[i],
        "n_floor": xp.full(NP, DENSITY_FLOOR * xp.max(dev.N_c)),
        # Row scales: each equation over the size of its terms, for a well-conditioned solve.
        "sP": 1.0 / (Q * dev.N_c[i]),
        "sC": 1.0 / (xp.maximum(dev.mu_n[i], dev.mu_p[i]) * dev.Vt * dev.N_c[i] * h[i]),
    }  # fmt: skip
    if dev.trap_charge is not None:  # charged bulk traps, only where there are any
        out["tq"], out["tcwe"] = dev.trap_charge[i], dev.trap_cwe[i]
    if dev.interface is not None:  # on the edges (`Device.interface`), only where there are any
        out["ti"] = dict(dev.interface)
    return out


def _per_level(levels, *values):
    """`values` with a trailing axis where the trap arrays carry levels (`levels` has more axes
    than `values[0]`): a trap file's several levels, summed by the caller."""
    if jnp.ndim(levels) > jnp.ndim(values[0]):
        return tuple(jnp.asarray(v)[..., None] for v in values)
    return values


def _summed(levels, value, like):
    return jnp.sum(value, -1) if jnp.ndim(levels) > jnp.ndim(like) else value


def _recombination(n, p, s):
    """Direct recombination and SRH through the bulk traps, summed over their levels."""
    nl, pl, Cn, Cp = _per_level(s["Nt"], n, p, s["Cn"], s["Cp"])
    srh = s["Nt"] * Cn * Cp * (nl * pl - s["nt0"] * s["pt0"])
    srh = srh / jnp.where(s["Nt"] > 0, Cn * (nl + s["nt0"]) + Cp * (pl + s["pt0"]), 1.0)
    return s["kd"] * (n * p - s["ni2"]) + _summed(s["Nt"], jnp.where(s["Nt"] > 0, srh, 0.0), n)


def _trap_charge(n, p, s):
    """The charged bulk traps' charge density (m⁻³, positive): `Σ |type|·(cwe − f)·N_t` over the
    levels, `f = (C_n·n + C_p·p_t0)/(C_n·(n + n_t0) + C_p·(p + p_t0))` each level's occupancy
    (SIMsalabim's `Calc_f_tb`, steady state)."""
    nl, pl, Cn, Cp = _per_level(s["tq"], n, p, s["Cn"], s["Cp"])
    q = s["tq"] > 0
    f = (Cn * nl + Cp * s["pt0"]) / jnp.where(q, Cn * (nl + s["nt0"]) + Cp * (pl + s["pt0"]), 1.0)
    return _summed(s["tq"], jnp.where(q, s["tq"] * (s["tcwe"] - f), 0.0), n)


def _interface_edges(nL, nR, pL, pR, ti):
    """Interface traps on each edge (`Device.interface`) from the densities at its left (`L`) and
    right (`R`) point: `(Rn_L, Rn_R, Rp_L, Rp_R, Q)`, each carrier's net capture at either point
    (m⁻³ s⁻¹) and the traps' charge density at both (m⁻³, positive). SIMsalabim's
    `Calc_Recombination_n/p` and `Calc_Trap_Filling_Charge`, steady state: the traps' occupancy
    from both points' densities, each point capturing from its own; zero where there are none."""
    Nt = ti["Nt"]
    like = nL
    nL, nR, pL, pR, Cn, Cp, iw, charge, cwe = _per_level(
        Nt, nL, nR, pL, pR, ti["Cn"], ti["Cp"], ti["iw"], ti["charge"], ti["cwe"])  # fmt: skip
    an, bp = Cn * Nt, Cp * Nt
    a = an * (nL + nR)
    b = bp * (pL + pR)
    c = an * (ti["nt0_lo"] + ti["nt0_hi"])
    d = bp * (ti["pt0_lo"] + ti["pt0_hi"])
    traps = Nt > 0
    inv = jnp.where(traps, 1.0 / jnp.where(traps, a + b + c + d, 1.0), 0.0)
    full, empty = (a + d) * inv, (b + c) * inv  # occupied, empty share
    return tuple(_summed(Nt, v, like) for v in (
        iw * an * (nL * empty - ti["nt0_lo"] * full),
        iw * an * (nR * empty - ti["nt0_hi"] * full),
        iw * bp * (pL * full - ti["pt0_lo"] * empty),
        iw * bp * (pR * full - ti["pt0_hi"] * empty),
        charge * (cwe - full) * 0.5 * Nt * iw,
    ))  # fmt: skip


def _interface_points(n, p, ti):
    """`_interface_edges` on the points of the full grid (`NP + 2`): `(Rn, Rp, Q)`, each point the
    sum of its two edges'."""
    Rn_L, Rn_R, Rp_L, Rp_R, Q = _interface_edges(n[:-1], n[1:], p[:-1], p[1:], ti)

    def points(lo, hi):  # point k is the right point of edge k − 1 and the left one of edge k
        return jnp.concatenate([lo[:1], hi[:-1] + lo[1:], hi[-1:]])

    return points(Rn_L, Rn_R), points(Rp_L, Rp_R), points(Q, Q)


def _interface_terms(n, p, ti):
    """The interface traps' `(Q, Rn, Rp)` on the interior points (3 × NP) and their derivatives
    in `(ln n, ln p)` (3 × NP × 2) of the point itself, the next and the one before."""
    args = (n[:-1], n[1:], p[:-1], p[1:])

    def f(*a):
        return jnp.stack(_interface_edges(*a, ti))

    z = jnp.zeros_like(args[0])
    value = f(*args)
    d_nL, d_nR, d_pL, d_pR = (
        jax.jvp(f, args, tuple(a if k == j else z for k, a in enumerate(args)))[1] for j in range(4)
    )
    lo, hi = (
        np.array([4, 0, 2]),
        np.array([4, 1, 3]),
    )  # (Q, Rn, Rp) of an edge's left point, of its right point

    def at_points(m):
        return m[hi][:, :-1] + m[lo][:, 1:]

    own = jnp.stack([d_nR[hi][:, :-1] + d_nL[lo][:, 1:], d_pR[hi][:, :-1] + d_pL[lo][:, 1:]], -1)
    up = jnp.stack([d_nR[lo][:, 1:], d_pR[lo][:, 1:]], -1)
    down = jnp.stack([d_nL[hi][:, :-1], d_pL[hi][:, :-1]], -1)
    return at_points(value), own, up, down


def _ion_densities(V, la, lc, s, Vt):
    """Anions and cations at a point: fixed, plus Boltzmann in V where they are mobile (the
    zero-current profile of SIMsalabim's `Calc_Ion_Distribution_Steady_State`)."""
    anion = s["a_fix"] + s["a_mask"] * jnp.exp(la + V / Vt)
    cation = s["c_fix"] + s["c_mask"] * jnp.exp(lc - V / Vt)
    return anion, cation


def _row(left, mid, right, s, Vt, g_frac, la, lc, B=bernoulli):
    """The three residuals of one interior point from `(V, ln n, ln p)` at it and its two
    neighbours, and the ions' normalisations `la`, `lc`."""
    V = jnp.stack([left[0], mid[0], right[0]])
    n = jnp.exp(jnp.stack([left[1], mid[1], right[1]]))
    p = jnp.exp(jnp.stack([left[2], mid[2], right[2]]))
    anion, cation = _ion_densities(V[1], la, lc, s, Vt)
    poisson = (
        s["fac_l"] * V[0]
        + s["fac_m"] * V[1]
        + s["fac_u"] * V[2]
        - Q * (n[1] - p[1] + s["dop"] + anion - cation)
    )
    if "tq" in s:
        poisson = poisson + Q * _trap_charge(n[1], p[1], s)
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
    """The batched residual, its Jacobian blocks and its derivative in the ions'
    normalisations, of `_row` with Bernoulli function `B`."""
    row = functools.partial(_row, B=B)
    axes = (0, 0, 0, 0, None, None, None, None)
    return (
        jax.vmap(row, in_axes=axes),
        jax.vmap(jax.jacfwd(row, argnums=(0, 1, 2)), in_axes=axes),
        jax.vmap(jax.jacfwd(row, argnums=(6, 7)), in_axes=axes),
    )


def _assemble(full, s, Vt, g_frac, la, lc, B=bernoulli):
    """`_row`'s residual (NP × 3), Jacobian blocks `A, B, C` (NP × 3 × 3, in the left, own and
    right point's unknowns) and ion columns (NP × 3 × 2) for every interior point at once,
    from `full` (NP + 2 × 3, contacts included).

    By edges: each interval's Scharfetter–Gummel fluxes and their derivatives are computed
    once, `B` and `B'` in one forward pass, and each row takes its two intervals' (autodiff of
    each row in its three neighbours evaluated every flux twice, and the residual a third
    time; docs/solver.md). Equal to `_kernels` to round-off (tests).
    """
    V, ln_n, ln_p = full[:, 0], full[:, 1], full[:, 2]
    n, p = jnp.exp(ln_n), jnp.exp(ln_p)
    # Per edge e = 0 … NP, between points e and e + 1.
    offn = jnp.concatenate([s["offn"][:, 0], s["offn"][-1, 1:]])
    offp = jnp.concatenate([s["offp"][:, 0], s["offp"][-1, 1:]])
    cn = Vt * jnp.concatenate([s["mun_m"], s["mun_p"][-1:]])
    cp = Vt * jnp.concatenate([s["mup_m"], s["mup_p"][-1:]])
    xn = jnp.diff(V + offn) / Vt
    xp = jnp.diff(V + offp) / Vt
    args = jnp.concatenate([xn, -xn, xp, -xp])
    Bv, dB = jax.jvp(B, (args,), (jnp.ones_like(args),))
    Bn_f, Bn_b, Bp_f, Bp_b = jnp.split(Bv, 4)
    dBn_f, dBn_b, dBp_f, dBp_b = jnp.split(dB, 4)
    n0, n1, p0, p1 = n[:-1], n[1:], p[:-1], p[1:]
    zero = jnp.zeros_like(xn)
    # Electrons: Fn = cn·(B(x)·n₁ − B(−x)·n₀); holes: Fp = cp·(B(x)·p₀ − B(−x)·p₁).
    Fn = cn * (Bn_f * n1 - Bn_b * n0)
    gn = cn * (dBn_f * n1 + dBn_b * n0) / Vt
    dFn0 = jnp.stack([-gn, -cn * Bn_b * n0, zero], -1)
    dFn1 = jnp.stack([gn, cn * Bn_f * n1, zero], -1)
    Fp = cp * (Bp_f * p0 - Bp_b * p1)
    gp = cp * (dBp_f * p0 + dBp_b * p1) / Vt
    dFp0 = jnp.stack([-gp, zero, cp * Bp_f * p0], -1)
    dFp1 = jnp.stack([gp, zero, -cp * Bp_b * p1], -1)

    # Per interior point i = 1 … NP: its left edge i − 1 ([:-1]) and right edge i ([1:]).
    Vm, nm, pm = V[1:-1], n[1:-1], p[1:-1]
    hm, hp, fac = s["hm"][:, None], s["hp"][:, None], s["fac"]
    R, dR_n = jax.jvp(lambda a: _recombination(a, pm, s), (nm,), (nm,))
    _, dR_p = jax.jvp(lambda b: _recombination(nm, b, s), (pm,), (pm,))
    net = s["G"] * g_frac - R
    zm = jnp.zeros_like(Vm)
    dnet = jnp.stack([zm, -dR_n, -dR_p], -1)
    elec = hm[:, 0] * Fn[1:] - hp[:, 0] * Fn[:-1] + fac * net
    hole = hp[:, 0] * Fp[:-1] - hm[:, 0] * Fp[1:] + fac * net
    anion = s["a_mask"] * jnp.exp(la + Vm / Vt)
    cation = s["c_mask"] * jnp.exp(lc - Vm / Vt)
    poisson = (
        s["fac_l"] * V[:-2]
        + s["fac_m"] * Vm
        + s["fac_u"] * V[2:]
        - Q * (nm - pm + s["dop"] + s["a_fix"] + anion - s["c_fix"] - cation)
    )

    P_A = jnp.stack([s["fac_l"], zm, zm], -1)
    P_C = jnp.stack([s["fac_u"], zm, zm], -1)
    P_B = jnp.stack([s["fac_m"] - Q * (anion + cation) / Vt, -Q * nm, Q * pm], -1)
    if "tq" in s:  # charged bulk traps: their charge in Poisson, from the point's own densities
        Qt, dQ_n = jax.jvp(lambda a: _trap_charge(a, pm, s), (nm,), (nm,))
        _, dQ_p = jax.jvp(lambda b: _trap_charge(nm, b, s), (pm,), (pm,))
        poisson = poisson + Q * Qt
        P_B = P_B + Q * jnp.stack([zm, dQ_n, dQ_p], -1)
    sP, sC = s["sP"][:, None], s["sC"][:, None]
    A = jnp.stack([P_A * sP, -hp * dFn0[:-1] * sC, hp * dFp0[:-1] * sC], -2)
    Bb = jnp.stack([
        P_B * sP,
        (-hp * dFn1[:-1] + hm * dFn0[1:] + fac[:, None] * dnet) * sC,
        (hp * dFp1[:-1] - hm * dFp0[1:] + fac[:, None] * dnet) * sC,
    ], -2)  # fmt: skip
    C = jnp.stack([P_C * sP, hm * dFn1[1:] * sC, -hm * dFp1[1:] * sC], -2)
    if "ti" in s:  # interface traps: their charge in Poisson, their capture in continuity
        X, own, up, down = _interface_terms(n, p, s["ti"])
        scale = jnp.stack([Q * s["sP"], -fac * s["sC"], -fac * s["sC"]], -1)  # rows (Q, Rn, Rp)
        poisson = poisson + Q * X[0]
        elec = elec - fac * X[1]
        hole = hole - fac * X[2]

        def block(d):  # 3 × NP × 2 → NP × 3 × 3, nothing in the potential's column
            d = jnp.moveaxis(d, 0, 1) * scale[..., None]
            return jnp.concatenate([jnp.zeros_like(d[..., :1]), d], -1)

        A, Bb, C = A + block(down), Bb + block(own), C + block(up)
    F = jnp.stack([poisson * s["sP"], elec * s["sC"], hole * s["sC"]], -1)
    U = jnp.stack([
        jnp.stack([-Q * anion * s["sP"], zm, zm], -1),
        jnp.stack([Q * cation * s["sP"], zm, zm], -1),
    ], -1)  # fmt: skip
    return F, A, Bb, C, U


def _ion_constraints(V_full, la, lc, ion):
    """`(g_a, g_c)`: each mobile species' number over its total, less 1 (trapezoidal over its
    region, as SIMsalabim's `Rescale_Ion_Density`); 0 for a species without one."""

    def species(mask, total, sign, log_norm):
        dens = mask * jnp.exp(log_norm + sign * V_full / ion["Vt"])
        number = jnp.sum(0.5 * (dens[1:] + dens[:-1]) * mask[1:] * mask[:-1] * ion["hL"])
        return jnp.where(total > 0, number / jnp.where(total > 0, total, 1.0) - 1.0, 0.0)

    return jnp.stack([
        species(ion["a_mask"], ion["a_total"], 1.0, la),
        species(ion["c_mask"], ion["c_total"], -1.0, lc),
    ])  # fmt: skip


#: Grid points per kernel in the block Thomas sweep: unrolling lets XLA fuse several of the
#: sweep's sequential steps into one kernel, which on a GPU is what the sweep costs (M3).
SWEEP_UNROLL = 8


def _inverse3(m):
    """The inverse of 3×3 matrices (`... × 3 × 3`) by the adjugate: elementwise arithmetic only,
    so that XLA fuses it instead of calling a batched LU per grid point (M3). The rows are
    scaled to their terms' size (`_stencils`), which keeps the determinant well away from
    round-off."""
    a, b, c = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    d, e, f = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    g, h, i = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    co = jnp.stack([
        jnp.stack([e * i - f * h, c * h - b * i, b * f - c * e], -1),
        jnp.stack([f * g - d * i, a * i - c * g, c * d - a * f], -1),
        jnp.stack([d * h - e * g, b * g - a * h, a * e - b * d], -1),
    ], -2)  # fmt: skip
    det = a * co[..., 0, 0] + b * co[..., 1, 0] + c * co[..., 2, 0]
    return co / det[..., None, None]


def _mm(a, b):
    """`a @ b` for 3×3 blocks (`... × 3 × 3` times `... × 3 × k`), written as multiply and sum
    so that XLA fuses it with the arithmetic around it. `@` on such small matrices becomes a
    batched GEMM of its own on a GPU: a kernel per product, launched in every step of a sweep
    and every level of the reduction (M3)."""
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


def _block_thomas(A, B, C, d):
    """Solve the block-tridiagonal system `A[i]·x[i−1] + B[i]·x[i] + C[i]·x[i+1] = d[i]`.

    Parallel cyclic reduction and a hybrid of it with this sweep were built and measured (M3,
    commits 13a1f61, 9fda49c) and taken out again: slower everywhere — on an A100 at batch 16384
    the sweep 1198 devices/s, the hybrid 824, PCR about a third of the sweep, its log₂ N passes
    over every block bandwidth-bound. docs/solver.md."""

    def forward(carry, blocks):
        c_prev, d_prev = carry
        a, b, c, r = blocks
        m_inv = _inverse3(b - _mm(a, c_prev))  # one block
        c_new = _mm(m_inv, c)
        d_new = _mm(m_inv, r - _mm(a, d_prev))
        return (c_new, d_new), (c_new, d_new)

    zero_c = jnp.zeros_like(C[0])
    zero_d = jnp.zeros_like(d[0])
    _, (cp, dp) = jax.lax.scan(forward, (zero_c, zero_d), (A, B, C, d), unroll=SWEEP_UNROLL)

    def backward(x_next, blocks):
        c, r = blocks
        x = r - _mm(c, x_next)
        return x, x

    _, x = jax.lax.scan(backward, zero_d, (cp, dp), reverse=True, unroll=SWEEP_UNROLL)
    return x


@functools.partial(jax.jit, static_argnames="bernoulli_name")
def _newton_step(y, logs, bounds, s, ion, Vt, g_frac, bernoulli_name="exact"):
    """One damped Newton step on the interior unknowns `y` (NP × 3) and the ions'
    normalisations `logs` = `(ln C_anion, ln C_cation)`.

    Free ions make the system block-tridiagonal bordered by two rows and columns (each
    normalisation enters every Poisson row of its region, its constraint every potential):
    solved by the block Thomas sweep on three right-hand sides and a 2×2 Schur complement
    (Sherman–Morrison–Woodbury, docs/solver.md).
    """
    la, lc = logs[0], logs[1]
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    F, A, B, C, U = _assemble(full, s, Vt, g_frac, la, lc, BERNOULLI[bernoulli_name])

    def constraints(V_interior, logs_):
        V_full = jnp.concatenate([bounds[0][:1], V_interior, bounds[1][:1]])
        return _ion_constraints(V_full, logs_[0], logs_[1], ion)

    g = constraints(full[1:-1, 0], logs)
    # Two outputs of NP + 2 inputs: reverse mode, two passes (forward mode took NP; M3).
    dg_dV, D = jax.jacrev(constraints, argnums=(0, 1))(full[1:-1, 0], logs)  # 2 × NP, 2 × 2
    # A species without a mobile region: its normalisation is held (an identity row).
    D = D + jnp.diag(jnp.where(jnp.array([ion["a_total"], ion["c_total"]]) > 0, 0.0, 1.0))
    W = jnp.zeros((y.shape[0], 3, 2)).at[:, 0, :].set(dg_dV.T)

    X = _block_thomas(A, B, C, jnp.concatenate([-F[..., None], U], axis=-1))
    XF, XU = X[..., 0], X[..., 1:]
    schur = D - jnp.einsum("irk,irl->kl", W, XU)
    d_logs = jnp.linalg.solve(schur, -g - jnp.einsum("irk,ir->k", W, XF))
    delta = XF - XU @ d_logs
    # Damping per point, as SIMsalabim limits its potential update (maxDelV): no potential
    # step past 10 V_T and no density step past a factor e^5 (`STEP_CAPS`).
    caps = step_caps(Vt)
    step = jnp.clip(delta, -caps, caps)
    d_logs = jnp.clip(d_logs, -ION_LOG_CAP, ION_LOG_CAP)
    new = y + step
    # Converged by the potential in V_T and the densities relatively — a density far below any
    # that matters (n_floor) is weighted out: its logarithm is round-off at the end.
    dens = jnp.exp(new[:, 1:])
    weight = dens / (dens + s["n_floor"][:, None])
    size = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(step[:, 0])) / Vt, jnp.max(jnp.abs(step[:, 1:]) * weight)),
        jnp.max(jnp.abs(d_logs)),
    )
    return new, logs + d_logs, size


def _poisson_row(left, mid, right, s, Vt, ref_left, ref_mid, ref_right, la, lc):
    """Poisson's residual with the quasi-Fermi levels held: `n ∝ exp(V/V_T)`, `p ∝ exp(−V/V_T)`
    from the reference state, as SIMsalabim's `Solve_Poisson` moves them."""
    shift = jnp.stack([left - ref_left[0], mid - ref_mid[0], right - ref_right[0]]) / Vt
    ln_n = jnp.stack([ref_left[1], ref_mid[1], ref_right[1]]) + shift
    ln_p = jnp.stack([ref_left[2], ref_mid[2], ref_right[2]]) - shift
    V = jnp.stack([left, mid, right])
    n, p = jnp.exp(ln_n[1]), jnp.exp(ln_p[1])
    anion, cation = _ion_densities(V[1], la, lc, s, Vt)
    poisson = (
        s["fac_l"] * V[0]
        + s["fac_m"] * V[1]
        + s["fac_u"] * V[2]
        - Q * (n - p + s["dop"] + anion - cation)
    )
    if "tq" in s:
        poisson = poisson + Q * _trap_charge(n, p, s)
    return poisson * s["sP"]


def _on_points(s):
    """The stencils on the interior points: without the interface traps' edges."""
    return {k: v for k, v in s.items() if k != "ti"}


_P_AXES = (0, 0, 0, 0, None, 0, 0, 0, None, None)
_poisson_jac = jax.vmap(jax.jacfwd(_poisson_row, argnums=(0, 1, 2)), in_axes=_P_AXES)
_poisson_val = jax.vmap(_poisson_row, in_axes=_P_AXES)


def _ion_logs(V_full, ion):
    """The normalisations that put each mobile species' total into its region at `V`."""

    def species(mask, total, sign):
        dens = mask * jnp.exp(sign * V_full / ion["Vt"])
        number = jnp.sum(0.5 * (dens[1:] + dens[:-1]) * mask[1:] * mask[:-1] * ion["hL"])
        return jnp.where(total > 0, jnp.log(jnp.where(total > 0, total, 1.0) / number), 0.0)

    return jnp.stack([species(ion["a_mask"], ion["a_total"], 1.0),
                      species(ion["c_mask"], ion["c_total"], -1.0)])  # fmt: skip


@jax.jit
def _poisson_step(y, bounds, s, ion, Vt):
    """One Newton step of Poisson's equation alone; the densities follow the potential, the
    mobile ions' normalisations are re-fitted to it (held during the step)."""
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    V = full[:, 0]
    la, lc = _ion_logs(V, ion)
    args = (_on_points(s), Vt, full[:-2], full[1:-1], full[2:], la, lc)
    F = _poisson_val(V[:-2], V[1:-1], V[2:], *args)
    lo, mid, up = _poisson_jac(V[:-2], V[1:-1], V[2:], *args)
    dV = jax.lax.linalg.tridiagonal_solve(lo.at[0].set(0.0), mid, up.at[-1].set(0.0), -F[:, None])[
        :, 0
    ]
    dV = jnp.clip(dV, -STEP_CAPS[0] * Vt, STEP_CAPS[0] * Vt)
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
    R_int = (np.zeros(NP + 2), np.zeros(NP + 2))
    if dev.interface is not None:
        R_int = tuple(np.asarray(r) for r in _interface_points(n, p, dev.interface)[:2])

    def carrier(sn, Vg, dens, mu, R_i):
        U = R + R_i - dev.G * g_frac
        U[0] = U[-1] = 0.0
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
        # SIMsalabim then sets the current across each interface to the one before it plus the
        # current through the interface traps; the points after keep their values.
        for i in dev.i1[:-1]:
            J[i] = J[i - 1] + sn * 0.5 * Q * L * h[i] * (R_i[i] + R_i[i + 1])
        return J

    Jn = carrier(-1, V + dev.offset_n, n, dev.mu_n, R_int[0])
    Jp = carrier(1, V + dev.offset_p, p, dev.mu_p, R_int[1])
    total = Jn + Jp
    # With the anode on the left SIMsalabim turns every current's sign.
    return -dev.left_elec * float(np.sum(h[: NP + 1] * total) / np.sum(h[: NP + 1]))


def _initial(dev: Device, Vint: float):
    """SIMsalabim's `Init_Pot_Dens_Ions_Traps`: linear potential, linear Fermi level."""
    x, L = dev.x, dev.L
    e = dev.left_elec  # −1: the left electrode the cathode, 1 the anode
    V = dev.V0 + 0.5 * e * Vint + x * (dev.VL - dev.V0 - e * Vint) / L
    EF = dev.W_L - x * (dev.W_L - dev.W_R) / L
    n = dev.N_c * np.exp(-(EF - dev.E_c) / dev.Vt)
    p = dev.ni**2 / n
    return np.stack([V, np.log(n), np.log(p)], -1)


def _ion_setup(dev: Device) -> dict:
    """What the ions' constraints need, over the full grid."""
    return {
        "Vt": dev.Vt, "hL": dev.h[:-1] * dev.L,
        "a_mask": dev.anion_mask.astype(float), "c_mask": dev.cation_mask.astype(float),
        "a_total": dev.anion_total, "c_total": dev.cation_total,
    }  # fmt: skip


def _freeze(s: dict, ion: dict, y, logs, bounds, Vt) -> tuple[dict, dict]:
    """The mobile ions of the solution `y` as fixed densities, no longer solved for (SimSS's
    `fixIons` with `simss/fixed_ions.patch`: frozen after the pre-bias, in Poisson's equation
    too)."""
    V = y[:, 0]
    s = dict(s)
    s["a_fix"] = s["a_fix"] + s["a_mask"] * jnp.exp(logs[0] + V / Vt)
    s["c_fix"] = s["c_fix"] + s["c_mask"] * jnp.exp(logs[1] - V / Vt)
    s["a_mask"] = jnp.zeros_like(s["a_mask"])
    s["c_mask"] = jnp.zeros_like(s["c_mask"])
    ion = dict(ion, a_mask=ion["a_mask"] * 0, c_mask=ion["c_mask"] * 0, a_total=0.0, c_total=0.0)
    return s, ion


def solve_jv(dev: Device, voltages: np.ndarray, g_frac: float | None = None,
             tol: float = 1e-10, max_iter: int = 200,
             bernoulli_name: str = "exact") -> JVResult:  # fmt: skip
    """The J-V curve at `voltages` (internal = external here), each solved from the last.

    With `preCond`, the device is first solved at `Vpre` with its ions free; with `fixIons`
    they are then held for the scan (SIMsalabim's sequence, `fixIons` as patched in `simss/`).
    """
    g_frac = dev.G_frac if g_frac is None else g_frac
    s = jax.tree_util.tree_map(jnp.asarray, _stencils(dev))
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in _ion_setup(dev).items()}  # fmt: skip
    first = dev.V_pre if dev.pre_cond else float(voltages[0])
    y_full = _initial(dev, first)
    y = jnp.asarray(y_full[1:-1])
    logs = _ion_logs(jnp.asarray(y_full[:, 0]), ion)
    if dev.fix_ions and not dev.pre_cond:
        # Held from the start: the ions as SIMsalabim initialises them, uniform in their layer.
        s, ion = _freeze(s, ion, y, logs, None, dev.Vt)

    def bounds_at(Vint):
        return jnp.array([
            [dev.V0 + 0.5 * dev.left_elec * Vint, np.log(dev.n_left), np.log(dev.p_left)],
            [dev.VL - 0.5 * dev.left_elec * Vint, np.log(dev.n_right), np.log(dev.p_right)],
        ])  # fmt: skip

    def settle(y, logs, s, ion, Vint):
        """Newton to `tol`, or to the round-off floor: a step below `ROUND_OFF_FLOOR` that has
        stopped falling for `STALL_STEPS` iterations."""
        bounds = bounds_at(Vint)
        sizes: list[float] = []
        while len(sizes) < max_iter:
            y, logs, size = _newton_step(y, logs, bounds, s, ion, dev.Vt, g_frac, bernoulli_name)
            sizes.append(float(size))
            if sizes[-1] < tol:
                return y, logs, True, len(sizes)
            recent = sizes[-STALL_STEPS - 1 :]
            if (len(recent) > STALL_STEPS and max(recent) < ROUND_OFF_FLOOR
                    and min(recent[1:]) >= 0.1 * recent[0]):  # fmt: skip
                return y, logs, True, len(sizes)
        return y, logs, False, len(sizes)

    for _ in range(max_iter):
        y, size = _poisson_step(y, bounds_at(first), s, ion, dev.Vt)
        if float(size) < 1e-8:
            break
    logs = _ion_logs(
        jnp.concatenate([bounds_at(first)[0][:1], y[:, 0], bounds_at(first)[1][:1]]), ion
    )
    if "tq" in s:
        # Charged bulk traps: from the start Newton finds no way to the solution (each density
        # step moves the traps' charge and so the potential, every step at its cap); from the
        # solution with the traps neutral it does, in a few steps (docs/solver.md).
        y_n, logs_n, ok, _ = settle(y, logs, dict(s, tq=jnp.zeros_like(s["tq"])), ion, first)
        if ok:
            y, logs = y_n, logs_n
    if dev.pre_cond:
        y, logs, ok, _ = settle(y, logs, s, ion, dev.V_pre)
        if not ok:
            raise RuntimeError(f"no convergence at the pre-bias {dev.V_pre} V")
        if dev.fix_ions:
            s, ion = _freeze(s, ion, y, logs, None, dev.Vt)

    J, conv, its = [], [], []
    for Vint in voltages:
        y_new, logs_new, ok, it = settle(y, logs, s, ion, float(Vint))
        bounds = bounds_at(float(Vint))
        full = np.concatenate(
            [np.asarray(bounds[0])[None], np.asarray(y_new), np.asarray(bounds[1])[None]]
        )
        # Each voltage starts from the last converged solution: a failed one costs that
        # voltage, not the rest of the scan.
        if ok and np.isfinite(full).all() and np.isfinite(np.asarray(logs_new)).all():
            y, logs = y_new, logs_new
        J.append(
            _current(dev, full[:, 0], np.exp(full[:, 1]), np.exp(full[:, 2]), g_frac)
            + (float(Vint) / dev.R_shunt if dev.R_shunt > 0 else 0.0)
        )
        conv.append(ok)
        its.append(it)
    V, J = np.asarray(voltages, dtype=float), np.array(J)
    return JVResult(V, J, np.array(conv), np.array(its), V + J * dev.R_series)
