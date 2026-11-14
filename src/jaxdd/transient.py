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

So far: electrons, holes and traps, the voltage given (`Vint = Vext`). Mobile ions, the open
circuit and series resistance are the next steps of Stage C and refused here by name.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.batch import _bounds, _initial, _newton_solve, _poisson_solve, _quiet
from jaxdd.device import Device
from jaxdd.simss_input import TVG
from jaxdd.solver import (
    BERNOULLI,
    ION_LOG_CAP,
    _bulk_occupation,
    _interface_occupation,
    _ion_logs,
    _time_terms,
    _with_contacts,
    step_caps,
)

__all__ = ["TransientResult", "solve_transient", "transient_problems"]


@dataclass(frozen=True)
class TransientResult:
    """Per device and step (batch × steps): `J` the internal current density (A/m², ZimT's
    `Jint`), `J_ext` with the shunt (`Jext`), `converged` and the Newton steps."""

    J: np.ndarray
    J_ext: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray


def transient_problems(dev: Device, tvg: TVG | None = None) -> list[str]:
    """What `solve_transient` does not take yet, by name: mobile ions, the open circuit and
    series resistance (Stage C's next steps)."""
    problems = []
    if np.any(dev.anion_mask) or np.any(dev.cation_mask):
        problems.append("mobile ions")
    if dev.R_series > 0:
        problems.append("R_series > 0")
    if tvg is not None and np.any(tvg.oc):
        problems.append("the open circuit (Vext = oc)")
    return problems


def _solve_one(d, dti, V, G_frac, B, tol, max_iter):
    """One device through its steps: the start as SIMsalabim's (`Init_Pot_Dens_Ions_Traps`,
    Poisson at the first voltage), then each step by Newton from the one before."""
    s, ion = d["s"], d["ion"]
    d0, s0, ion0 = jax.lax.stop_gradient((d, s, ion))
    y = _poisson_solve(_initial(d0, V[0])[1:-1], d0, s0, ion0, V[0], max_iter)
    bounds = _bounds(d0, V[0])
    logs = _ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion0)
    # The steady state at the first step (t = 0), as `batch._solve_one` finds it: with charged
    # bulk traps from the solution with them neutral. The first step then starts from it.
    if "tq" in s0:
        neutral = dict(s0, tq=jnp.zeros_like(s0["tq"]))
        y_n, logs_n, _, ok_n, _ = _newton_solve(y, logs, d0, neutral, ion0, V[0], G_frac[0], B,
                                                tol, max_iter)  # fmt: skip
        y, logs = jnp.where(ok_n, y_n, y), jnp.where(ok_n, logs_n, logs)
    y_s, logs_s, _, ok_s, _ = _newton_solve(y, logs, d0, s0, ion0, V[0], G_frac[0], B, tol,
                                            max_iter)  # fmt: skip
    y, logs = jnp.where(ok_s, y_s, y), jnp.where(ok_s, logs_s, logs)
    caps = step_caps(d["Vt"])
    traps = "ti" in s

    def full_of(y, Vint):  # the contacts' densities as the current takes them (`_with_contacts`)
        b = _bounds(d, Vint)
        return _with_contacts(jnp.concatenate([b[0][None], y, b[1][None]]), s, d["Vt"], B)

    def stencils(before, dti_k, fb, fi):
        """The step's stencils and device: the state before it, the traps' occupations."""
        s_k = _time_terms(s, before, dti_k, fb, fi if traps else None)
        pad = jnp.zeros_like(fb[:1])
        tr = {"dti": dti_k, "n": jnp.exp(before[:, 1]), "p": jnp.exp(before[:, 2]),
              "V": before[:, 0], "fb": jnp.concatenate([pad, fb, pad])}  # fmt: skip
        return s_k, dict(d, s=s_k, tr=tr)

    def step(carry, x):
        y, logs, y_prev, logs_prev, V_now, dti_now, fb, fi, first = carry
        dti_k, V_k, G_k = x
        before = full_of(y, V_now)
        s_k, d_k = stencils(before, dti_k, fb, fi)
        # The start: the last two states extrapolated linearly in time (ZimT's
        # `Extrapolate_Solution`, in V and ln n), held to one Newton step's caps; none from or
        # into a steady state.
        moving = (dti_k > 0) & (dti_now > 0) & ~first
        rt = jnp.where(moving, dti_now / jnp.where(moving, dti_k, 1.0), 0.0)  # Δt_k / Δt_before
        guess = y + jnp.clip(rt * (y - y_prev), -caps, caps)
        logs_guess = logs + jnp.clip(rt * (logs - logs_prev), -ION_LOG_CAP, ION_LOG_CAP)
        y_new, logs_new, J, ok, it = _newton_solve(
            guess, logs_guess, d_k, s_k, ion, V_k, G_k, B, tol, max_iter, False
        )
        # The traps' occupations at the step's end, from its solution.
        after = full_of(y_new, V_k)
        fb_new = _bulk_occupation(jnp.exp(after[1:-1, 1]), jnp.exp(after[1:-1, 2]), s_k)
        fi_new = (_interface_occupation(jnp.exp(after[:, 1]), jnp.exp(after[:, 2]), s_k["ti"])
                  if traps else fi)  # fmt: skip
        # ZimT's failureMode 1 (its default): a step not converged is kept if it is finite.
        keep = jnp.all(jnp.isfinite(y_new)) & jnp.all(jnp.isfinite(logs_new))
        new = (y_new, logs_new, y, logs, V_k, dti_k, fb_new, fi_new)
        old = (y, logs, y_prev, logs_prev, V_now, dti_now, fb, fi)
        new = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), new, old)
        return (*new, jnp.array(False)), (J, ok, it)

    # Before the first step (t = 0, a steady state) the occupations do not matter.
    fb0 = jnp.zeros_like(s["Nt"])
    fi0 = jnp.zeros_like(s["ti"]["Nt"]) if traps else jnp.zeros(())
    start = (y, logs, y, logs, V[0], jnp.asarray(0.0), fb0, fi0, jnp.array(True))
    _, (J, ok, its) = jax.lax.scan(step, start, (dti, V, G_frac))
    return J, ok, its


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "max_iter"))
def _solve_batch(batch, dti, V, G_frac, tol, max_iter, bernoulli_name):
    one = functools.partial(_solve_one, B=BERNOULLI[bernoulli_name], tol=tol, max_iter=max_iter)
    J, ok, its = jax.vmap(one)(batch, dti, V, G_frac)
    return J, J + batch["G_shunt"][:, None] * V, ok, its


def solve_transient(batch, tvg: TVG | list[TVG], tol=1e-10, max_iter=200,
                    bernoulli_name="exact") -> TransientResult:  # fmt: skip
    """A batch (`batch.prepare`) through ZimT's time steps `tvg`: one for every device, or one
    per device (all of one length). Each step's Newton to `tol` or `max_iter`."""
    n = jax.tree_util.tree_leaves(batch)[0].shape[0]
    tvgs = list(tvg) if isinstance(tvg, list | tuple) else [tvg] * n
    if len(tvgs) != n or len({len(t.t) for t in tvgs}) != 1:
        raise ValueError("one tVG for the batch, or one per device, all of one length")
    if any(np.any(t.oc) for t in tvgs):
        raise NotImplementedError("the open circuit (Vext = oc) is not in jaxdd's transients yet")
    dti, V, G = (jnp.asarray(np.stack([getattr(t, k) for t in tvgs]))
                 for k in ("dti", "Vext", "G_frac"))  # fmt: skip
    with _quiet():
        J, J_ext, ok, its = _solve_batch(batch, dti, V, G, tol, max_iter, bernoulli_name)
    return TransientResult(*(np.asarray(a) for a in (J, J_ext, ok, its)))
