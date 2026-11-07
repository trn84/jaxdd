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

#: A Newton step below this (V_T, relative densities) that no longer falls is round-off: the
#: solution, three orders below SIMsalabim's own `tolDens` of 1e-5.
ROUND_OFF_FLOOR = 1e-6
STALL_STEPS = 3

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
        # Ions: a mobile region's mask (Boltzmann in V with a normalisation solved for) and
        # fixed densities (immobile, or frozen after a pre-bias).
        "a_mask": dev.anion_mask[i].astype(float), "c_mask": dev.cation_mask[i].astype(float),
        "a_fix": dev.anion_fixed[i], "c_fix": dev.cation_fixed[i],
        "n_floor": np.full(NP, DENSITY_FLOOR * dev.N_c.max()),
        # Row scales: each equation over the size of its terms, for a well-conditioned solve.
        "sP": 1.0 / (Q * dev.N_c[i]),
        "sC": 1.0 / (np.maximum(dev.mu_n[i], dev.mu_p[i]) * dev.Vt * dev.N_c[i] * h[i]),
    }  # fmt: skip


def _recombination(n, p, s):
    srh = s["Nt"] * s["Cn"] * s["Cp"] * (n * p - s["nt0"] * s["pt0"])
    srh = srh / jnp.where(s["Nt"] > 0, s["Cn"] * (n + s["nt0"]) + s["Cp"] * (p + s["pt0"]), 1.0)
    return s["kd"] * (n * p - s["ni2"]) + jnp.where(s["Nt"] > 0, srh, 0.0)


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
    """Solve the block-tridiagonal system `A[i]·x[i−1] + B[i]·x[i] + C[i]·x[i+1] = d[i]`."""

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


def _pcr_levels(A, B, C, d, levels):
    """`levels` levels of cyclic reduction: afterwards row `i` couples only to `i ± 2^levels`."""
    n = A.shape[0]
    eye = jnp.broadcast_to(jnp.eye(3), B.shape)
    zero_m = jnp.zeros_like(A)
    zero_d = jnp.zeros_like(d)

    def down(X, s, fill):  # X[i - s], `fill` where i - s < 0
        return jnp.concatenate([fill[:s], X[: n - s]])

    def up(X, s, fill):  # X[i + s], `fill` where i + s >= n
        return jnp.concatenate([X[s:], fill[:s]])

    s = 1
    for _ in range(levels):
        if s >= n:
            break
        B_inv = _inverse3(B)
        alpha = -_mm(A, down(B_inv, s, eye))
        gamma = -_mm(C, up(B_inv, s, eye))
        A, B, C, d = (
            _mm(alpha, down(A, s, zero_m)),
            B + _mm(alpha, down(C, s, zero_m)) + _mm(gamma, up(A, s, zero_m)),
            _mm(gamma, up(C, s, zero_m)),
            d + _mm(alpha, down(d, s, zero_d)) + _mm(gamma, up(d, s, zero_d)),
        )
        s *= 2
    return A, B, C, d


#: Levels of cyclic reduction before the hybrid hands over to the sweep: 2^3 = 8 independent
#: systems of N/8 rows, swept side by side (Zhang, Cohen and Owens, PPoPP 2010).
HYBRID_LEVELS = 3


def _block_hybrid(A, B, C, d, levels=HYBRID_LEVELS):
    """Cyclic reduction for `levels` levels, then the block Thomas sweep on the `2^levels`
    systems it leaves, all at once: `2·N/2^levels` sequential steps instead of `2·N`, at the
    cost of `levels` levels of extra arithmetic instead of `log₂ N` (PCR)."""
    n = A.shape[0]
    stride = 2**levels
    pad = (-n) % stride
    if pad:
        eye = jnp.broadcast_to(jnp.eye(3), (pad, 3, 3))
        A = jnp.concatenate([A, jnp.zeros((pad, 3, 3))])
        B = jnp.concatenate([B, eye])
        C = jnp.concatenate([C, jnp.zeros((pad, 3, 3))])
        d = jnp.concatenate([d, jnp.zeros((pad, *d.shape[1:]))])
    A, B, C, d = _pcr_levels(A, B, C, d, levels)
    m = A.shape[0] // stride

    def split(X):  # row i = j·stride + r  →  X[j, r]
        return X.reshape(m, stride, *X.shape[1:])

    x = jax.vmap(_block_thomas, in_axes=1, out_axes=1)(split(A), split(B), split(C), split(d))
    return x.reshape(m * stride, *x.shape[2:])[:n]


def _block_pcr(A, B, C, d):
    """The same system by parallel cyclic reduction (M3): each level, every row eliminates its
    neighbours at distance `s` at once and couples to those at `2s`; after ⌈log₂ N⌉ levels the
    rows are uncoupled and `x = B⁻¹·d`. Nine levels for 400 points instead of 800 sequential
    steps — more arithmetic (N log N), far less latency, which is what a GPU pays for here.

    `A[0]` and `C[N−1]` must be zero (they are: the boundary values are known). Outside the
    system a row is the identity with no right-hand side.
    """
    n = A.shape[0]
    levels = max(1, (n - 1).bit_length())
    A, B, C, d = _pcr_levels(A, B, C, d, levels)
    return _mm(_inverse3(B), d)


#: How the Newton system is solved: the block Thomas sweep (least arithmetic; the CPU's),
#: parallel cyclic reduction (least latency, but log₂ N passes over every block: bandwidth-bound
#: on an A100, 81 against the sweep's 265 devices/s), or the hybrid of the two.
LINEAR_SOLVERS = {"thomas": _block_thomas, "pcr": _block_pcr, "hybrid": _block_hybrid}


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "linear"))
def _newton_step(y, logs, bounds, s, ion, Vt, g_frac, bernoulli_name="exact", linear="thomas"):
    """One damped Newton step on the interior unknowns `y` (NP × 3) and the ions'
    normalisations `logs` = `(ln C_anion, ln C_cation)`.

    Free ions make the system block-tridiagonal bordered by two rows and columns (each
    normalisation enters every Poisson row of its region, its constraint every potential):
    solved by the block Thomas sweep on three right-hand sides and a 2×2 Schur complement
    (Sherman–Morrison–Woodbury, PLAN.md §3.3).
    """
    row_val, row_jac, row_ions = _kernels(BERNOULLI[bernoulli_name])
    la, lc = logs[0], logs[1]
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    left, mid, right = full[:-2], full[1:-1], full[2:]
    F = row_val(left, mid, right, s, Vt, g_frac, la, lc)
    A, B, C = row_jac(left, mid, right, s, Vt, g_frac, la, lc)
    U = jnp.stack(row_ions(left, mid, right, s, Vt, g_frac, la, lc), axis=-1)  # NP × 3 × 2

    def constraints(V_interior, logs_):
        V_full = jnp.concatenate([bounds[0][:1], V_interior, bounds[1][:1]])
        return _ion_constraints(V_full, logs_[0], logs_[1], ion)

    g = constraints(full[1:-1, 0], logs)
    dg_dV, D = jax.jacfwd(constraints, argnums=(0, 1))(full[1:-1, 0], logs)  # 2 × NP, 2 × 2
    # A species without a mobile region: its normalisation is held (an identity row).
    D = D + jnp.diag(jnp.where(jnp.array([ion["a_total"], ion["c_total"]]) > 0, 0.0, 1.0))
    W = jnp.zeros((y.shape[0], 3, 2)).at[:, 0, :].set(dg_dV.T)

    X = LINEAR_SOLVERS[linear](A, B, C, jnp.concatenate([-F[..., None], U], axis=-1))
    XF, XU = X[..., 0], X[..., 1:]
    schur = D - jnp.einsum("irk,irl->kl", W, XU)
    d_logs = jnp.linalg.solve(schur, -g - jnp.einsum("irk,ir->k", W, XF))
    delta = XF - XU @ d_logs
    # Damping per point, as SIMsalabim limits its potential update (maxDelV): no potential
    # step past 10 V_T and no density step past a factor e^5.
    caps = jnp.array([10.0 * Vt, 5.0, 5.0])
    step = jnp.clip(delta, -caps, caps)
    d_logs = jnp.clip(d_logs, -5.0, 5.0)
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
    return poisson * s["sP"]


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
    args = (s, Vt, full[:-2], full[1:-1], full[2:], la, lc)
    F = _poisson_val(V[:-2], V[1:-1], V[2:], *args)
    lo, mid, up = _poisson_jac(V[:-2], V[1:-1], V[2:], *args)
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


def _ion_setup(dev: Device) -> dict:
    """What the ions' constraints need, over the full grid."""
    return {
        "Vt": dev.Vt, "hL": dev.h[:-1] * dev.L,
        "a_mask": dev.anion_mask.astype(float), "c_mask": dev.cation_mask.astype(float),
        "a_total": dev.anion_total, "c_total": dev.cation_total,
    }  # fmt: skip


def _freeze(s: dict, ion: dict, y, logs, bounds, Vt) -> tuple[dict, dict]:
    """The mobile ions of the solution `y` as fixed densities, no longer solved for (FauxPV's
    `fixIons` with its patch: frozen after the pre-bias, in Poisson's equation too)."""
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
    they are then held for the scan (SIMsalabim's sequence, FauxPV's patched `fixIons`).
    """
    g_frac = dev.G_frac if g_frac is None else g_frac
    s = {k: jnp.asarray(v) for k, v in _stencils(dev).items()}
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
            [dev.V0 - 0.5 * Vint, np.log(dev.n_left), np.log(dev.p_left)],
            [dev.VL + 0.5 * Vint, np.log(dev.n_right), np.log(dev.p_right)],
        ])  # fmt: skip

    def settle(y, logs, s, ion, Vint):
        """Newton to `tol`, or to the round-off floor: a step below `ROUND_OFF_FLOOR` that has
        stopped falling for `STALL_STEPS` iterations (strong frozen charge leaves the potential
        a few 1e-8 V_T of noise)."""
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
    if dev.pre_cond:
        y, logs, ok, _ = settle(y, logs, s, ion, dev.V_pre)
        if not ok:
            raise RuntimeError(f"no convergence at the pre-bias {dev.V_pre} V")
        if dev.fix_ions:
            s, ion = _freeze(s, ion, y, logs, None, dev.Vt)

    J, conv, its = [], [], []
    for Vint in voltages:
        y, logs, ok, it = settle(y, logs, s, ion, float(Vint))
        bounds = bounds_at(float(Vint))
        full = np.concatenate(
            [np.asarray(bounds[0])[None], np.asarray(y), np.asarray(bounds[1])[None]]
        )
        J.append(_current(dev, full[:, 0], np.exp(full[:, 1]), np.exp(full[:, 2]), g_frac))
        conv.append(ok)
        its.append(it)
    return JVResult(np.asarray(voltages), np.array(J), np.array(conv), np.array(its))
