"""Impedance in the frequency domain, from the steady state's Jacobian (jaxdd's; ZimT measures it
in time, a transient per frequency).

At a steady state `y₀` (`batch.solve_states`) a small voltage `δV·e^(iωt)` drives the linear
system `(J + iω·M)·δy = −(∂F/∂V)·δV`: `J` the steady residual's Jacobian, `M` the time terms
implicit Euler adds over a step (`−fac·n` in each continuity row's own `ln n`; with mobile ions
their rows too, the ions then unknowns of their own as in a transient, `solver._assemble_ions`,
each slot's number a constraint as there), one complex block-tridiagonal solve per frequency by
the steady state's sweep. The bulk traps' occupations are states with their own time,
`df/dt = b − a·f` (`solver._trap_step`): eliminated point by point,
`δf = (∂b − f·∂a)·δy/(iω + a)`, which replaces the steady Jacobian's instantaneous occupation
(`iω = 0`) in the rates and the traps' charge. The current is a transient's
(`batch._current`: SIMsalabim's integral form over each point's control volume, the ions' and
the displacement current, averaged over the intervals), linearised: `Y = δJ/δV`; with a series
resistance and a shunt the external `Z = 1/(Y + G_shunt) + R_series`.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.batch import _bounds, _current, _full, _quiet, _sum_widths, solve_states
from jaxdd.constants import Q
from jaxdd.solver import (
    BERNOULLI,
    ION_FLUXES,
    _assemble,
    _assemble_ions,
    _block_thomas,
    _degenerate,
    _interface_degenerate,
    _ion_constraints_time,
    _ion_full,
    _ion_membership,
    _per_level,
    _slots,
    _summed,
    _time_terms,
)

__all__ = ["ImpedanceResult", "solve_impedance"]


@dataclass(frozen=True)
class ImpedanceResult:
    """Per device and frequency (batch × frequencies): `Z` the external impedance (Ω·m², with
    the series resistance and the shunt), `Y` the internal admittance `δJ/δV_int` (S/m², in
    SIMsalabim's sign: its `J`'s derivative); per device the steady state's `J` (A/m²) and
    `converged`. `capacitance` is `Im(Y)/ω` (F/m²)."""

    frequencies: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    J: np.ndarray
    converged: np.ndarray

    @property
    def capacitance(self) -> np.ndarray:
        return self.Y.imag / (2 * np.pi * self.frequencies)


def _trap_terms(s, n, p):
    """The bulk traps' occupation as a state of its own, per interior point and level: its rate
    `a = C_n·(n + n_t0) + C_p·(p + p_t0)`, the drive `∂b − f·∂a` in `ln n` and `ln p` (`f = b/a`
    the steady occupation, `b = C_n·n + C_p·p_t0`; with another statistics the degenerate
    `n_t0`, `p_t0`, `solver._degenerate`), and the occupation's part in the electrons' and the
    holes' rates and in the charge: `∂R_n/∂f = −N_t·C_n·(n + n_t0)`,
    `∂R_p/∂f = N_t·C_p·(p + p_t0)`, `∂Q_t/∂f = −tq`."""

    def ab(n_, p_):
        sd = _degenerate(s, n_, p_)
        nl, pl, Cn, Cp = _per_level(s["Nt"], n_, p_, s["Cn"], s["Cp"])
        a = Cn * (nl + sd["nt0"]) + Cp * (pl + sd["pt0"])
        b = Cn * nl + Cp * sd["pt0"]
        return a, b, Cn * (nl + sd["nt0"]), Cp * (pl + sd["pt0"])

    (a, b, en, ep), (da_n, db_n, _, _) = jax.jvp(ab, (n, p), (n, jnp.zeros_like(p)))
    _, (da_p, db_p, _, _) = jax.jvp(ab, (n, p), (jnp.zeros_like(n), p))
    traps = s["Nt"] > 0
    a = jnp.where(traps, a, 1.0)
    f = b / a
    drive = jnp.stack([db_n - f * da_n, db_p - f * da_p], -1)  # ... × levels × 2
    dR = jnp.where(traps, jnp.stack([-s["Nt"] * en, s["Nt"] * ep], 0), 0.0)  # 2 × ... × levels
    dQ = -jnp.where(traps, s["tq"], 0.0) if "tq" in s else None
    return a, drive, dR, dQ


def _beyond_steady(s, n, p, lap, dt=None):
    """The traps' occupation at `iω` = `lap` beyond its steady (instantaneous) response
    `δf₀ = drive·δy/a`, per unit of `δ ln n` and `δ ln p` (... × levels × 2): in the rates and in
    the charge, `δf − δf₀ = −δf₀·iω/(iω + a)` both. With `dt` what `solver._trap_step` gives over
    implicit Euler's steps of `dt` (`lap` then `(1 − z⁻¹)/dt`): the occupation at a step's end
    `δf = (1 − E)·δf₀/(1 − E·z⁻¹)`, `E = e^(−a·dt)`, in the charge, and its mean over the step
    `(1 − m)·δf₀ + m·z⁻¹·δf`, `m = (1 − E)/(a·dt)`, in the rates. With `_trap_terms`' `dR` and
    `dQ`."""
    a, drive, dR, dQ = _trap_terms(s, n, p)
    unit = drive / a[..., None]  # δf₀ per unit of the logarithms
    h_rate, h_charge = _step_transfer(a, lap, dt)
    return unit * h_rate[..., None], unit * h_charge[..., None], dR, dQ


def _step_transfer(a, lap, dt):
    """`(δf − δf₀)/δf₀` of a trap level of rate `a` at `iω` = `lap`, in the rates and in the
    charge (`_beyond_steady`): continuous both `−iω/(iω + a)`; over implicit Euler's steps of
    `dt` the mean over a step and its end."""
    if dt is None:
        h = -lap / (lap + a)
        return h, h
    x = a * dt
    E, m = jnp.exp(-x), -jnp.expm1(-x) / x
    den = 1.0 - E * (1.0 - lap * dt)  # 1 − E·z⁻¹
    return -m * lap * dt / den, -E * lap * dt / den


def _interface_beyond(ti, n, p, lap, dt=None):
    """The interface traps' occupation on each edge (NP + 1, over its levels) beyond its steady
    response, per unit of the four logarithms it follows (`ln n`, `ln p` at the edge's left and
    right point, `solver._interface_edges`: `f = b/a` with `a = C_n·(n_L + n_R + n_t0) + C_p·(p_L +
    p_R + p_t0)`, `b = C_n·(n_L + n_R) + C_p·p_t0`), in the rates and in the charge
    (`_step_transfer`), and the occupation's part in the four point rates `(R_n,L, R_n,R, R_p,L,
    R_p,R)` and the charge: `(rate, charge, dR)`, `rate` and `charge` edges × levels × 4, `dR`
    5 × edges × levels."""
    Nt = ti["Nt"]

    def terms(nL, nR, pL, pR):
        t = _interface_degenerate(ti, nL, nR, pL, pR)
        nl, nr, pl, pr, Cn, Cp, iw, charge = _per_level(
            Nt, nL, nR, pL, pR, t["Cn"], t["Cp"], t["iw"], t["charge"])  # fmt: skip
        nt0, pt0 = t["nt0_lo"] + t["nt0_hi"], t["pt0_lo"] + t["pt0_hi"]
        a = Cn * (nl + nr + nt0) + Cp * (pl + pr + pt0)
        b = Cn * (nl + nr) + Cp * pt0
        an, bp = Cn * Nt, Cp * Nt
        dR = jnp.stack([-iw * an * (nl + t["nt0_lo"]), -iw * an * (nr + t["nt0_hi"]),
                        iw * bp * (pl + t["pt0_lo"]), iw * bp * (pr + t["pt0_hi"]),
                        -charge * 0.5 * Nt * iw])  # fmt: skip
        return a, b, dR

    args = (n[:-1], n[1:], p[:-1], p[1:])
    zero = jnp.zeros_like(args[0])
    a, b, dR = terms(*args)
    drive = []
    for j in range(4):
        tangent = tuple(v if k == j else zero for k, v in enumerate(args))
        _, (da, db, _) = jax.jvp(terms, args, tangent)
        drive.append(db - b / jnp.where(a > 0, a, 1.0) * da)
    traps = Nt > 0
    a = jnp.where(traps, a, 1.0)
    unit = jnp.stack(drive, -1) / a[..., None]  # edges × levels × 4
    h_rate, h_charge = _step_transfer(a, lap, dt)
    dR = jnp.where(traps, dR, 0.0)
    return unit * h_rate[..., None], unit * h_charge[..., None], dR


def _interface_blocks(s, n, p, lap, dt=None):
    """The interface traps' correction to the blocks `(A, B, C)` (NP × 3 × 3 each, complex) at
    `iω` = `lap`: each edge's occupation beyond its steady response (`_interface_beyond`) in
    its two points' rates and charge, through their own and each other's logarithms, with the
    rows' scales (`solver._assemble`: Poisson's `+Q·X`, the continuity rows' `−fac·R`)."""
    ti = s["ti"]
    rate, charge, dR = _interface_beyond(ti, n, p, lap, dt)
    NP = n.shape[0] - 2

    def summed(v):  # over the levels where a trap file has several
        return _summed(ti["Nt"], v, n[:-1])

    def block(side, cols):  # edges × 3 rows × 2 columns: the point `side`'s rows, the columns
        # (ln n, ln p) of the point `cols` (0 left, 1 right)
        jn, jp = (0, 2) if cols == 0 else (1, 3)
        rows = [
            (dR[4], charge),
            (dR[0 if side == 0 else 1], rate),
            (dR[2 if side == 0 else 3], rate),
        ]
        return jnp.stack([jnp.stack([summed(d * x[..., jn]), summed(d * x[..., jp])], -1)
                          for d, x in rows], -2)  # fmt: skip

    def scaled(b, k):  # the rows of interior point `k` (its scales)
        rs = jnp.stack([Q * s["sP"][k], -s["fac"][k] * s["sC"][k], -s["fac"][k] * s["sC"][k]], -1)
        return jnp.concatenate([jnp.zeros_like(b[..., :1]), b], -1) * rs[..., None]

    e = jnp.arange(1, NP)  # edges with both points interior
    dA = jnp.zeros((NP, 3, 3), dtype=rate.dtype)
    dB = jnp.zeros((NP, 3, 3), dtype=rate.dtype)
    dC = jnp.zeros((NP, 3, 3), dtype=rate.dtype)
    dB = dB.at[e - 1].add(scaled(block(0, 0)[e], e - 1)).at[e].add(scaled(block(1, 1)[e], e))
    dC = dC.at[e - 1].add(scaled(block(0, 1)[e], e - 1))
    dA = dA.at[e].add(scaled(block(1, 0)[e], e))
    return dA, dB, dC


def _interface_rates(s, n, p, lap, dy3, dt=None):
    """Each point's electron and hole rate (NP + 2) of the interface traps beyond their steady
    occupation, at the response `dy3` (NP × 3, complex): `batch._current_parts`' `tr["dRi"]`."""
    ti = s["ti"]
    rate, _, dR = _interface_beyond(ti, n, p, lap, dt)
    full = jnp.pad(dy3, ((1, 1), (0, 0)))  # the contacts' logarithms do not move with traps
    logs = jnp.stack([full[:-1, 1], full[1:, 1], full[:-1, 2], full[1:, 2]], -1)  # edges × 4
    df = jnp.sum(rate * (logs[:, None, :] if rate.ndim == 3 else logs), -1)  # edges × levels

    def summed(v):
        return _summed(ti["Nt"], v, n[:-1])

    def points(lo, hi):  # point k is the right point of edge k − 1 and the left one of edge k
        return jnp.concatenate([lo[:1], hi[:-1] + lo[1:], hi[-1:]])

    return (points(summed(dR[0] * df), summed(dR[1] * df)),
            points(summed(dR[2] * df), summed(dR[3] * df)))  # fmt: skip


def _trap_blocks(s, n, p, lap, dt=None):
    """The traps' correction to the own-point blocks (NP × 3 × 3, complex) at `iω` = `lap`:
    `_beyond_steady` in the rates and the charge, with the rows' scales."""
    rate, charge, dR, dQ = _beyond_steady(s, n, p, jnp.asarray(lap), dt)

    def by_column(v, excess):  # NP × 2 (ln n, ln p), summed over the levels where there are
        return jnp.stack([_summed(s["Nt"], v * excess[..., k], n) for k in (0, 1)], -1)

    zero = jnp.zeros((n.shape[0], 2), dtype=rate.dtype)
    rows = [zero if dQ is None else Q * s["sP"][:, None] * by_column(dQ, charge)]
    rows += [-(s["fac"] * s["sC"])[:, None] * by_column(dR[k], rate) for k in (0, 1)]
    block = jnp.stack(rows, -2)  # NP × 3 rows × 2 columns
    return jnp.concatenate([jnp.zeros_like(block[..., :1]), block], -1)


def _admittance(d, state, ions_grid, Vint, g_frac, laps, B, ions, drift=False, dt=None,
                response=False):  # fmt: skip
    """One device's `δJ/δV_int` at each `iω` of `laps` (complex), at its steady state `state`
    (NP + 2 × 3: V, ln n, ln p) with its ions' densities `ions_grid` (NP + 2 × 2). `drift`: ions
    on sites with the modified drift. `dt`: implicit Euler's steps (`laps` theirs, the traps'
    `_beyond_steady`). `response`: also `δy` (frequencies × NP × unknowns)."""
    if drift:  # a key, not a value (`solver._drift_factor`)
        d = dict(d, ion=dict(d["ion"], drift=jnp.asarray(1.0)))
    s, ion, Vt = d["s"], d["ion"], d["Vt"]
    y3 = state[1:-1]
    NP = y3.shape[0]
    U = W = held = dg = None
    if ions:  # the ions unknowns of their own (`solver._assemble_ions`), a step of `dti` = 1

        def log_of(dens, mask):
            return jnp.where(mask > 0, jnp.log(jnp.where(mask > 0, dens, 1.0)), 0.0)

        y = jnp.concatenate([y3, jnp.stack([log_of(ions_grid[1:-1, 0], s["a_mask"]),
                                            log_of(ions_grid[1:-1, 1], s["c_mask"])], -1)],
                            -1)  # fmt: skip
        s_k = _time_terms(s, _full(_bounds(d, Vint), y), 1.0)
        logs = jnp.zeros(_slots(ion)[0].shape[0])

        def blocks(Vint_):
            return _assemble_ions(_full(_bounds(d, Vint_), y), s_k, Vt, g_frac, logs, ion, B,
                                  mass=True)  # fmt: skip

        _, A, Bb, C, U, M = blocks(Vint)
        shift = -1.0  # the step's `dti` = 1 is in `Bb` already

        # Each slot's ions keep their number: at low frequencies the ions' rows fix their
        # electrochemical potential in a region only up to a constant, which the time term
        # `iω·n` alone pins (the system's condition ~1/(ωτ)). As in a transient (ZimT's
        # `Rescale_Ion_Density`), a scale per slot (`U`, the rows' derivative in it) and the
        # slot's number, trapezoidal over its region, as a constraint (`_ion_constraints_time`).
        def numbers(y_, Vint_):
            full = _ion_full(_full(_bounds(d, Vint_), y_), ion, Vt, B, 1.0)
            return _ion_constraints_time(full, logs, ion)

        W = jax.jacrev(numbers)(y, Vint)  # K × NP × 5
        dg = jax.jvp(lambda v: numbers(y, v), (Vint,), (jnp.ones_like(Vint),))[1]
        totals = _slots(ion)[1]
        held = jnp.where(totals > 0, 0.0, 1.0)
    else:
        y = y3
        zero = jnp.zeros(NP)
        fac = s["fac"]
        M = jnp.stack([zero, -fac * jnp.exp(y3[:, 1]) * s["sC"],
                       -fac * jnp.exp(y3[:, 2]) * s["sC"]], -1)  # fmt: skip

        def blocks(Vint_):
            return _assemble(_full(_bounds(d, Vint_), y3), s, Vt, g_frac, zero, zero, B)

        _, A, Bb, C, _ = blocks(Vint)
        shift = 0.0
    # The voltage's drive: the residual's derivative in the internal voltage (the contacts).
    _, dF = jax.jvp(lambda v: blocks(v)[0], (Vint,), (jnp.ones_like(Vint),))
    n, p = jnp.exp(y3[:, 1]), jnp.exp(y3[:, 2])
    n_full, p_full = jnp.exp(state[:, 1]), jnp.exp(state[:, 2])
    eye = jnp.eye(y.shape[-1])

    def solve(lap):
        Bl = Bb + ((lap + shift) * M)[..., None] * eye
        Bl = Bl.at[:, :3, :3].add(_trap_blocks(s, n, p, lap, dt))  # 0 without traps
        Al, Cl = A.astype(Bl.dtype), C.astype(Bl.dtype)
        if "ti" in s:  # interface traps: each edge's occupation couples its two points
            dA, dB, dC = _interface_blocks(s, n_full, p_full, lap, dt)
            Al, Bl, Cl = (Al.at[:, :3, :3].add(dA), Bl.at[:, :3, :3].add(dB),
                          Cl.at[:, :3, :3].add(dC))  # fmt: skip
        rhs = (-dF).astype(Bl.dtype)[..., None]
        if U is not None:  # bordered by the slots (`solver._newton_step`'s Schur complement)
            rhs = jnp.concatenate([rhs, U.astype(Bl.dtype)], -1)
        X = _block_thomas(Al, Bl, Cl, rhs)
        if U is None:
            return X[..., 0]
        XF, XU = X[..., 0], X[..., 1:]
        schur = jnp.diag(held) - jnp.einsum("kir,irl->kl", W, XU)
        z = jnp.linalg.solve(schur, -dg - jnp.einsum("kir,ir->k", W, XF))
        return XF - XU @ z

    dy = jax.vmap(solve)(laps)  # frequencies × NP × unknowns

    # The current as a transient takes it (`batch._current`), affine in the step's `dti`:
    # `δJ = J_y·δy + iω·(J_y(1) − J_y(0))·δy`; the traps' occupations beyond their steady
    # response enter as each carrier's rate (`tr["dU"]`).
    full0 = _full(_bounds(d, Vint), y)
    zU = (jnp.zeros(NP + 2), jnp.zeros(NP + 2))
    tr0 = {"n": jnp.exp(full0[:, 1]), "p": jnp.exp(full0[:, 2]), "V": full0[:, 0],
           "width": _sum_widths(d["h"][: NP + 1], False)}  # fmt: skip
    if ions:
        ma, mc = _ion_membership(ion)
        tr0["ions"] = {"a": jnp.exp(full0[:, 3]) * ma, "c": jnp.exp(full0[:, 4]) * mc}
        tr0["ion_last"] = jnp.asarray(0.0)

    zRi = (jnp.zeros(NP + 2), jnp.zeros(NP + 2))

    def total(y_, Vint_, dti, dU, dRi=zRi):
        tr = dict(tr0, dti=jnp.asarray(dti), dU=dU, dRi=dRi)
        return _current(dict(d, tr=tr), _full(_bounds(d, Vint_), y_), g_frac, B)

    def linear(dy_real, dV, dti):  # `J_y·δy + J_V·δV` at `dti`
        return jax.jvp(lambda y_, v_: total(y_, v_, dti, zU), (y, Vint), (dy_real, dV))[1]

    def rates(dU_real, dRi_real):  # the current's response to the carriers' rates
        def of(u, r):
            return total(y, Vint, 0.0, u, r)

        return jax.jvp(of, (zU, zRi), (dU_real, dRi_real))[1]

    def beyond(lap, dy_):  # each carrier's rate beyond the steady traps', on every point
        excess, _, dR, _ = _beyond_steady(s, n, p, lap, dt)
        df = sum(excess[..., k] * _per_level(s["Nt"], dy_[:, k + 1])[0] for k in (0, 1))
        pad = jnp.zeros(1, dtype=df.dtype)
        return tuple(jnp.concatenate([pad, _summed(s["Nt"], dR[k] * df, n), pad]) for k in (0, 1))

    one, none = jnp.ones_like(Vint), jnp.zeros_like(Vint)

    def current(lap, dy_):
        J0 = linear(dy_.real, one, 0.0) + 1j * linear(dy_.imag, none, 0.0)
        J1 = linear(dy_.real, one, 1.0) + 1j * linear(dy_.imag, none, 1.0)
        dUn, dUp = beyond(lap, dy_)
        if "ti" in s:
            dRn, dRp = _interface_rates(s, n_full, p_full, lap, dy_[:, :3], dt)
        else:
            dRn = dRp = jnp.zeros(NP + 2, dtype=dUn.dtype)
        Ju = (rates((dUn.real, dUp.real), (dRn.real, dRp.real))
              + 1j * rates((dUn.imag, dUp.imag), (dRn.imag, dRp.imag)))  # fmt: skip
        return J0 + lap * (J1 - J0) + Ju

    out = jax.vmap(current)(laps, dy)  # frequencies
    return (out, dy) if response else out


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "ions", "drift", "dt"))
def _batch_admittance(batch, states, ions_grid, Vint, g_frac, laps, bernoulli_name, ions,
                      drift, dt):  # fmt: skip
    one = functools.partial(_admittance, laps=laps, B=BERNOULLI[bernoulli_name], ions=ions,
                            drift=drift, dt=dt)  # fmt: skip
    return jax.vmap(one)(batch, states, ions_grid, Vint, g_frac)


def solve_impedance(batch, frequencies, voltage=0.0, tol=1e-10, max_iter=60,
                    bernoulli_name="exact", ion_flux="electrochemical",
                    dt=None) -> ImpedanceResult:  # fmt: skip
    """The impedance of a batch (`batch.prepare`) at the internal voltage `voltage` (V; the
    devices' `G_frac`) at each of `frequencies` (Hz), from the steady state there
    (`solve_states`, its `tol` and `max_iter`) by one linear solve per frequency (docs/model.md,
    "Impedance"). Mobile ions move with the voltage (their steady state the free ions'), with
    sites at the flux `ion_flux` names (`transient.solve_transient`'s); the bulk traps'
    occupations follow at their own rates. `dt` (s): instead of the continuous response, what
    `transient.solve_transient` sees through implicit Euler at steps of `dt` (its time terms at
    `(1 − e^(−iωΔt))/Δt` for `iω`, the traps' occupations over each step as it solves them):
    what ZimT's impedance from a sine would measure at that step, but for the sine's amplitude
    and settling. Refused: a free end (no circuit), ions frozen at a pre-bias."""
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if ion_flux not in ION_FLUXES:
        raise ValueError(f"ion_flux {ion_flux!r}: one of {ION_FLUXES}")
    s = batch["s"]
    if "free" in s:
        raise NotImplementedError("impedance with a free end: there is no circuit through it")
    ion = batch["ion"]
    ions = bool(np.any(np.asarray(ion["a_totals"]) > 0) or np.any(np.asarray(ion["c_totals"]) > 0))
    if ions and np.any((np.asarray(batch["pre_cond"]) > 0) & (np.asarray(batch["fix_ions"]) > 0)):
        raise NotImplementedError("impedance with ions frozen at a pre-bias")
    st = solve_states(batch, [float(voltage)], tol=tol, max_iter=max_iter,
                      bernoulli_name=bernoulli_name)  # fmt: skip
    state = jnp.stack([st["V"][:, 0], jnp.log(st["n"][:, 0]), jnp.log(st["p"][:, 0])], -1)
    ions_grid = jnp.stack([st["anion"][:, 0], st["cation"][:, 0]], -1)
    Vint = jnp.full(state.shape[0], float(voltage))
    omega = 2 * np.pi * frequencies
    laps = 1j * omega if dt is None else -np.expm1(-1j * omega * dt) / dt
    with _quiet():
        Y = _batch_admittance(batch, state, ions_grid, Vint, batch["g_frac"],
                              jnp.asarray(np.asarray(laps, complex)), bernoulli_name, ions,
                              ion_flux == "modified-drift",
                              None if dt is None else float(dt))  # fmt: skip
    Y = np.asarray(Y)
    Z = 1.0 / (Y + np.asarray(batch["G_shunt"])[:, None]) + np.asarray(batch["R_series"])[:, None]
    return ImpedanceResult(frequencies=frequencies, Z=Z, Y=Y, J=np.asarray(st["J"][:, 0]),
                           converged=np.asarray(st["converged"][:, 0]))  # fmt: skip
