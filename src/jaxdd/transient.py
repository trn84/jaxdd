"""Transients as ZimT 5.36 solves them (Stage C, PLAN.md section 4.1): a batch of devices
through a list of time steps `(t, Vext, G_frac)` (`simss_input.read_tvg`).

Each step is one Newton solve of the whole system with implicit Euler in time (ZimT's
`Main_Solver` with `dti = 1/Δt`): the same residual as the steady state with
`−fac·dti·(n − n_prev)` in each continuity equation (`solver._time_terms`), from the state of the
step before; `dti = 0` is the steady state, so the first step (`t = 0`) and any return to it are
steady states. The traps' occupations are state carried from step to step, each level's
solved exactly over the step from the densities at its end (`solver._trap_step`, ZimT's
`Calc_f_tb`); each carrier's capture takes the occupation's mean over the step, so that
electrons and holes recombine at different rates while traps fill or empty. The current adds
each density's change to the integral form's integrand and the displacement current
(`batch._current`). The steps are a `lax.scan`, the devices `vmap`.

The internal voltage: given (`Vint = Vext`), or, at the open circuit (`Vext = oc`) and with a
series resistance, an unknown of the step's Newton solve with one more bordered row
(`batch._vint_row`: `Jext = 0`, or `Vint + Jext·R_series = Vext`), where ZimT brackets it and runs
Brent on whole solves.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.batch import _bounds, _caps, _full, _initial, _newton_solve, _poisson_solve, _quiet
from jaxdd.simss_input import TVG
from jaxdd.solver import (
    BERNOULLI,
    ION_LOG_CAP,
    STEP_CAPS,
    _bulk_occupation,
    _interface_occupation,
    _ion_full,
    _ion_logs,
    _ion_membership,
    _point_logs,
    _time_terms,
    _with_contacts,
)

__all__ = ["TransientResult", "solve_transient"]


@dataclass(frozen=True)
class TransientResult:
    """Per device and step (batch × steps): `J` the internal current density (A/m², ZimT's
    `Jint`), `J_ext` with the shunt (`Jext`), the internal and external voltages (V, `Vint`,
    `Vext = Vint + Jext·R_series`), `converged` and the Newton steps."""

    J: np.ndarray
    J_ext: np.ndarray
    V_int: np.ndarray
    V_ext: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray


#: The internal voltage's first guess at an open circuit in the steady state (`t = 0`), from
#: which the step's Newton finds it (ZimT brackets from 0 V).
OC_GUESS = 0.0


def _solve_one(d, dti, V, oc, G_frac, B, tol, max_iter, ions, vint):
    """One device through its steps: the start as SIMsalabim's (`Init_Pot_Dens_Ions_Traps`,
    Poisson at the first voltage), then each step by Newton from the one before. With `ions` the
    mobile ions are unknowns of their own (`V, ln n, ln p, ln a, ln c` per point,
    `solver._assemble_ions`), from the steady state's Boltzmann profiles. With `vint` the internal
    voltage is one more unknown of each step (`batch._vint_row`), the external voltage `V` its
    target (where `oc` is 0)."""
    s, ion = d["s"], d["ion"]
    V_first = jnp.where(oc[0] > 0, OC_GUESS, V[0])
    d0, s0, ion0, V0 = jax.lax.stop_gradient((d, s, ion, V_first))
    y = _poisson_solve(_initial(d0, V0)[1:-1], d0, s0, ion0, V0, max_iter)
    bounds = _bounds(d0, V0)
    logs = _ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion0)
    # The steady state at the first step (t = 0), as `batch._solve_one` finds it: with charged
    # bulk traps from the solution with them neutral. The first step then starts from it.
    if "tq" in s0:
        neutral = dict(s0, tq=jnp.zeros_like(s0["tq"]))
        y_n, logs_n, _, ok_n, _ = _newton_solve(y, logs, d0, neutral, ion0, V0, G_frac[0], B,
                                                tol, max_iter)  # fmt: skip
        y, logs = jnp.where(ok_n, y_n, y), jnp.where(ok_n, logs_n, logs)
    y_s, logs_s, J_s, ok_s, _ = _newton_solve(y, logs, d0, s0, ion0, V0, G_frac[0], B, tol,
                                              max_iter)  # fmt: skip
    y, logs = jnp.where(ok_s, y_s, y), jnp.where(ok_s, logs_s, logs)
    Vt = d["Vt"]
    if ions:  # the ions' densities as unknowns: their steady profiles, 0 outside their regions
        la, lc = _point_logs(logs, ion)
        Vm = y[:, 0]
        y = jnp.concatenate([y, jnp.stack([jnp.where(s["a_mask"] > 0, la + Vm / Vt, 0.0),
                                           jnp.where(s["c_mask"] > 0, lc - Vm / Vt, 0.0)], -1)],
                            -1)  # fmt: skip
    caps = _caps(Vt, y.shape[-1])
    v_cap = STEP_CAPS[0] * Vt
    traps = "ti" in s

    def full_of(y, Vint, dti_):  # the contacts as the current takes them (`_with_contacts`)
        full = _with_contacts(_full(_bounds(d, Vint), y), s, Vt, B)
        return _ion_full(full, ion, Vt, B, dti_) if ions else full

    def stencils(before, dti_k, fb, fi):
        """The step's stencils and device: the state before it, the traps' occupations."""
        s_k = _time_terms(s, before, dti_k, fb, fi if traps else None)
        pad = jnp.zeros_like(fb[:1])
        tr = {"dti": dti_k, "n": jnp.exp(before[:, 1]), "p": jnp.exp(before[:, 2]),
              "V": before[:, 0], "fb": jnp.concatenate([pad, fb, pad])}  # fmt: skip
        if ions:
            ma, mc = _ion_membership(ion)
            tr["ions"] = {"a": jnp.exp(before[:, 3]) * ma, "c": jnp.exp(before[:, 4]) * mc}
        return s_k, dict(d, s=s_k, tr=tr)

    def step(carry, x):
        y, logs, y_prev, logs_prev, V_now, V_before, J_ext_now, dti_now, fb, fi, first = carry
        dti_k, V_k, oc_k, G_k = x
        before = full_of(y, V_now, dti_now)
        s_k, d_k = stencils(before, dti_k, fb, fi)
        # The start: the last two states extrapolated linearly in time (ZimT's
        # `Extrapolate_Solution`, in V and ln n), held to one Newton step's caps; none from or
        # into a steady state.
        moving = (dti_k > 0) & (dti_now > 0) & ~first
        rt = jnp.where(moving, dti_now / jnp.where(moving, dti_k, 1.0), 0.0)  # Δt_k / Δt_before
        guess = y + jnp.clip(rt * (y - y_prev), -caps, caps)
        logs_guess = logs + jnp.clip(rt * (logs - logs_prev), -ION_LOG_CAP, ION_LOG_CAP)
        # The internal voltage's start: at the open circuit the last one, extrapolated as the
        # state; with a series resistance the external voltage less the last current's drop.
        V_g = jnp.where(oc_k > 0, V_now + jnp.clip(rt * (V_now - V_before), -v_cap, v_cap),
                        V_k - d["R_series"] * J_ext_now)  # fmt: skip
        if ions:  # over a step the slots are the ions' scales (1 at the start), in the steady
            # state Boltzmann's normalisations
            steady = _ion_logs(full_of(guess, V_g, 0.0)[:, 0], ion)
            logs_guess = jnp.where(dti_k > 0, jnp.where(moving, logs_guess, 0.0), steady)
        if vint:
            y_new, logs_v, J, ok, it = _newton_solve(
                guess, jnp.concatenate([logs_guess, V_g[None]]), d_k, s_k, ion, (oc_k, V_k),
                G_k, B, tol, max_iter, False, True,
            )  # fmt: skip
            logs_new, V_new = logs_v[:-1], logs_v[-1]
        else:
            y_new, logs_new, J, ok, it = _newton_solve(
                guess, logs_guess, d_k, s_k, ion, V_k, G_k, B, tol, max_iter, False
            )
            V_new = V_k
        J_ext = J + d["G_shunt"] * V_new
        # The traps' occupations at the step's end, from its solution.
        after = full_of(y_new, V_new, dti_k)
        fb_new = _bulk_occupation(jnp.exp(after[1:-1, 1]), jnp.exp(after[1:-1, 2]), s_k)
        fi_new = (_interface_occupation(jnp.exp(after[:, 1]), jnp.exp(after[:, 2]), s_k["ti"])
                  if traps else fi)  # fmt: skip
        # ZimT's failureMode 1 (its default): a step not converged is kept if it is finite.
        keep = (jnp.all(jnp.isfinite(y_new)) & jnp.all(jnp.isfinite(logs_new))
                & jnp.isfinite(V_new))  # fmt: skip
        new = (y_new, logs_new, y, logs, V_new, V_now, J_ext, dti_k, fb_new, fi_new)
        old = (y, logs, y_prev, logs_prev, V_now, V_before, J_ext_now, dti_now, fb, fi)
        new = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), new, old)
        return (*new, jnp.array(False)), (J, J_ext, V_new, ok, it)

    # Before the first step (t = 0, a steady state) the occupations do not matter.
    fb0 = jnp.zeros_like(s["Nt"])
    fi0 = jnp.zeros_like(s["ti"]["Nt"]) if traps else jnp.zeros(())
    J0 = jnp.where(ok_s, J_s, 0.0) + d["G_shunt"] * V0
    start = (y, logs, y, logs, V0, V0, J0, jnp.asarray(0.0), fb0, fi0, jnp.array(True))
    _, out = jax.lax.scan(step, start, (dti, V, oc, G_frac))
    return out


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "max_iter", "ions", "vint"))
def _solve_batch(batch, dti, V, oc, G_frac, tol, max_iter, bernoulli_name, ions, vint):
    one = functools.partial(_solve_one, B=BERNOULLI[bernoulli_name], tol=tol, max_iter=max_iter,
                            ions=ions, vint=vint)  # fmt: skip
    J, J_ext, V_int, ok, its = jax.vmap(one)(batch, dti, V, oc, G_frac)
    return J, J_ext, V_int, V_int + J_ext * batch["R_series"][:, None], ok, its


def solve_transient(batch, tvg: TVG | list[TVG], tol=1e-10, max_iter=200,
                    bernoulli_name="exact") -> TransientResult:  # fmt: skip
    """A batch (`batch.prepare`) through ZimT's time steps `tvg`: one for every device, or one
    per device (all of one length). Each step's Newton to `tol` or `max_iter`. Where any device
    of the batch has an open circuit or a series resistance, the internal voltage is an unknown
    of every device's steps (with the voltage given, its row `Vint = Vext`)."""
    n = jax.tree_util.tree_leaves(batch)[0].shape[0]
    tvgs = list(tvg) if isinstance(tvg, list | tuple) else [tvg] * n
    if len(tvgs) != n or len({len(t.t) for t in tvgs}) != 1:
        raise ValueError("one tVG for the batch, or one per device, all of one length")
    dti, G = (jnp.asarray(np.stack([getattr(t, k) for t in tvgs])) for k in ("dti", "G_frac"))
    oc = np.stack([t.oc for t in tvgs])
    V = jnp.asarray(np.where(oc, 0.0, np.stack([t.Vext for t in tvgs])))
    ion = batch["ion"]
    ions = bool(np.any(np.asarray(ion["a_totals"]) > 0) or np.any(np.asarray(ion["c_totals"]) > 0))
    vint = bool(oc.any() or np.any(np.asarray(batch["R_series"]) > 0))
    with _quiet():
        out = _solve_batch(batch, dti, V, jnp.asarray(oc, dtype=float), G, tol, max_iter,
                           bernoulli_name, ions, vint)  # fmt: skip
    return TransientResult(*(np.asarray(a) for a in out))
