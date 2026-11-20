"""Transients as ZimT 5.36 solves them: a batch of devices
through a list of time steps `(t, Vext, G_frac)` (`simss_input.read_tvg`).

Each step is one Newton solve of the whole system with implicit Euler in time (ZimT's
`Main_Solver` with `dti = 1/Δt`): the same residual as the steady state with
`−fac·dti·(n − n_prev)` in each continuity equation (`solver._time_terms`), from the state of the
step before; `dti = 0` is the steady state, so the first step (`t = 0`) and any return to it are
steady states. The traps' occupations are state carried from step to step, each level's
solved exactly over the step from the densities at its end (`solver._trap_step`, ZimT's
`Calc_f_tb`); each carrier's capture takes the occupation's mean over the step, so that
electrons and holes recombine at different rates while traps fill or empty. The current adds
each density's change to the integral form's integrand, summed over each point's control
volume (ZimT's sum with `zimt_current`, docs/differences.md §13), and the displacement current
(`batch._current`). The steps are a `lax.scan`, the devices `vmap`.

The internal voltage: given (`Vint = Vext`), or, at the open circuit (`Vext = oc`) and with a
series resistance, an unknown of the step's Newton solve with one more bordered row
(`batch._vint_row`: `Jext = 0`, or `Vint + Jext·R_series = Vext`), where ZimT brackets it and runs
Brent on whole solves.

ZimT's main loop per device: its `failureMode` and automatic stop (`Track`, `States_Are_Equal`),
each step's `Δt` from the last one kept; and every column of its `tJFile` (`_columns`,
`tj_columns`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.batch import (
    _bounds,
    _caps,
    _current_parts,
    _full,
    _newton_solve,
    _quiet,
    _start,
    _sum_widths,
    _traced,
)
from jaxdd.constants import Q
from jaxdd.simss_input import TVG
from jaxdd.solver import (
    BERNOULLI,
    ION_FLUXES,
    ION_LOG_CAP,
    STEP_CAPS,
    _bulk_occupation,
    _carrier_excess,
    _degenerate,
    _free_ends,
    _interface_occupation,
    _ion_full,
    _ion_logs,
    _ion_membership,
    _point_logs,
    _time_terms,
    _volumes,
    _with_contacts,
    occupancy,
)

__all__ = ["TransientResult", "solve_transient", "tj_columns"]


@dataclass(frozen=True)
class TransientResult:
    """Per device and step (batch × steps): `J` the internal current density (A/m², ZimT's
    `Jint`), `J_ext` with the shunt (`Jext`), the internal and external voltages (V, `Vint`,
    `Vext = Vint + Jext·R_series`), `converged` and the Newton steps; `columns` the rest of
    ZimT's `tJFile` (`_columns`, per layer on a last axis padded to the batch's most layers),
    `n_layers` each device's; `tj_columns` takes them as ZimT names them. `accepted`: the steps
    ZimT keeps, the rows of its `tJFile` (not a step skipped by `failureMode = 2`, nor one at or
    after a stop); `stop` per device: 0 through every step, `STOP_STATIC` where the state has
    stabilised (`Track`, `MinCountStatic`), `STOP_FAILED` at a failed step (`failureMode` 0, or 2
    with the reference state the last one kept)."""

    J: np.ndarray
    J_ext: np.ndarray
    V_int: np.ndarray
    V_ext: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    columns: dict
    n_layers: np.ndarray
    accepted: np.ndarray
    stop: np.ndarray
    profiles: dict | None = None


def _columns(d, parts, points, J, J_ext):
    """ZimT's `tJFile` columns beyond the voltages and currents at one step (`Write_To_tJV_File`
    with `transient`), from the step's currents on each interval and its points
    (`batch._current_parts`): `errJ`, the current's rms deviation over the grid from its average
    (`Calc_RMS_Error`, the interval past the last point the last's); per layer the QFLS averaged
    on the grid, the photocurrent and each carrier's direct and bulk recombination as currents
    (`EquiCurr`); at each layer's right end the interface traps' recombination of each carrier
    (`q·L·h·(R[i1] + R[i1 + 1])`); the minority
    carriers' currents at the contacts, the shunt's, the ions' and the displacement current
    averaged; with `specialOutput = spv` each layer's voltage drop."""
    lay = d["layers"]
    cells, first, last = lay["cells"], lay["first"], lay["last"]
    x, h, L = d["x"], d["h"], d["L"]
    NP = x.shape[0] - 2
    hh = h[: NP + 1]
    total = sum(parts.values())
    dev = jnp.concatenate([total, total[-1:]]) - J
    err = jnp.sqrt(jnp.sum(dev**2) / (NP + 1))
    weight = cells * hh  # layers × intervals
    norm = jnp.sum(weight, axis=1)
    norm = jnp.where(norm > 0, norm, 1.0)

    def interior(v):  # the contacts' points take none (SIMsalabim's arrays are 0 there)
        return v.at[0].set(0.0).at[-1].set(0.0)

    def average(v):  # SIMsalabim's `Average` over each layer, a value per interval
        return weight @ v[: NP + 1] / norm

    def equi(v):  # `EquiCurr`: q times the layer's average times its thickness
        return Q * average(interior(v)) * ((last - first) @ x)

    V, n, p = points["V"], points["n"], points["p"]
    qfls = jnp.log(n) + jnp.log(p) - 2 * jnp.log(d["ni"])
    w = _carrier_excess(jnp.log(n), jnp.log(p), d["rec"])
    if w is not None:  # another statistics: the excess potentials' part (`solve_states`)
        qfls = qfls + w[0] + w[1]
    qfls = d["Vt"] * qfls
    h_end = last @ jnp.concatenate([h, jnp.zeros(x.shape[0] - h.shape[0])])
    next_ = jnp.roll(last, 1, axis=1)

    def through(R):  # at each layer's last point and the one after it
        return Q * L * h_end * (last @ R + next_ @ R)

    Jn, Jp = parts["Jn"], parts["Jp"]
    zero = jnp.zeros_like(Jn)

    def mean(v):
        return jnp.sum(hh * v) / jnp.sum(hh)

    return {
        "errJ": err,
        "QFLS": average(0.5 * (qfls[:-1] + qfls[1:])),
        "Jphoto": equi(points["gen"]),
        "Jdir": equi(points["direct"]),
        "JbulkElec": equi(points["bulk_n"]),
        "JbulkHoles": equi(points["bulk_p"]),
        "JintElec": through(points["int_n"]),
        "JintHoles": through(points["int_p"]),
        "JminLeft": jnp.where(n[0] < p[0], Jn[0], Jp[0]),
        "JminRight": jnp.where(n[-1] < p[-1], Jn[-1], Jp[-1]),
        "JShunt": J_ext - J,
        "Jnion": mean(parts.get("Ja", zero)),
        "Jpion": mean(parts.get("Jc", zero)),
        "JD": mean(parts.get("JD", zero)),
        "VL": (last - first) @ V,
    }


#: ZimT's per-layer columns (`L1`, `L2`, …) and per-interface ones (`L1L2`, …), in its order.
_LAYER_COLUMNS = ("QFLS", "Jphoto", "Jdir", "JbulkElec", "JbulkHoles")
_INTERFACE_COLUMNS = ("JintElec", "JintHoles")


def tj_columns(result: TransientResult, k: int = 0, spv: bool = False) -> dict[str, np.ndarray]:
    """Device `k`'s steps as ZimT 5.36's `tJFile` names its columns, in its order (`t` is the
    caller's): `Vext Jext errJ Vint Jint G_frac` (`G_frac` the caller's too, left out), then
    per layer and interface, then `JminLeft JminRight JShunt Jnion Jpion JD`; with `spv`
    (`specialOutput = spv`) each layer's voltage drop `VL1`, `VL2`, …"""
    c = {k_: np.asarray(v)[k] for k_, v in result.columns.items()}
    nl = int(result.n_layers[k])
    out = {"Vext": result.V_ext[k], "Jext": result.J_ext[k], "errJ": c["errJ"],
           "Vint": result.V_int[k], "Jint": result.J[k]}  # fmt: skip
    for name in _LAYER_COLUMNS:
        out.update({f"{name}L{j + 1}": c[name][:, j] for j in range(nl)})
    for name in _INTERFACE_COLUMNS:
        out.update({f"{name}L{j + 1}L{j + 2}": c[name][:, j] for j in range(nl - 1)})
    out.update({name: c[name] for name in ("JminLeft", "JminRight", "JShunt", "Jnion", "Jpion",
                                           "JD")})  # fmt: skip
    if spv:
        out.update({f"VL{j + 1}": c["VL"][:, j] for j in range(nl)})
    return out


#: ZimT's `MinCountStatic`: this many equal states in a row (`Track` 2 or 3) stop a transient.
MIN_COUNT_STATIC = 5
#: `TransientResult.stop`: stopped as ZimT's automatic stop has it, or at a failed step.
STOP_STATIC, STOP_FAILED = 1, 2


def _states_equal(A, B, ctl):
    """ZimT's `States_Are_Equal` of two state summaries (`_summary`) under the setup's `convVar`
    and tolerances (`ctl`): the same `G_frac`, the potentials within `tolPois` (V, the Euclidean
    norm over the grid), then by `convVar` the densities (each relatively within `tolDens`), the
    current (within `tolCurr` or the smaller `errJ`), either or both."""

    def norm(v):
        return jnp.sqrt(jnp.sum(v**2))

    def rel(a, b):
        scale = 0.5 * (norm(a) + norm(b))
        return jnp.where(scale > 0, norm(a - b) / jnp.where(scale > 0, scale, 1.0), 0.0)

    dens = jnp.all(jnp.array([rel(A[k], B[k]) for k in ("n", "p", "a", "c")]) < ctl["tol_dens"])
    curr = jnp.abs(A["J"] - B["J"]) <= jnp.maximum(ctl["tol_curr"], jnp.minimum(A["err"], B["err"]))
    cv = ctl["conv_var"]
    by = jnp.where(cv == 1, dens, jnp.where(cv == 2, curr, jnp.where(cv == 3, curr | dens,
                                                                      curr & dens)))  # fmt: skip
    same_v = norm(A["V"] - B["V"]) < ctl["tol_pois"]
    return (A["G"] == B["G"]) & same_v & by


#: The internal voltage's first guess at an open circuit in the steady state (`t = 0`), from
#: which the step's Newton finds it (ZimT brackets from 0 V).
OC_GUESS = 0.0


def _solve_one(d, t, V, oc, G_frac, track, ctl, recycle, B, tol, max_iter, ions, vint, ion_last,
               zimt_current, profiles=False, drift=False):  # fmt: skip
    """One device through its steps: the start as SIMsalabim's (`Init_Pot_Dens_Ions_Traps`,
    Poisson at the first voltage), then each step by Newton from the one before. With `ions` the
    mobile ions are unknowns of their own (`V, ln n, ln p, ln a, ln c` per point,
    `solver._assemble_ions`), from the steady state's Boltzmann profiles. With `vint` the internal
    voltage is one more unknown of each step (`batch._vint_row`), the external voltage `V` its
    target (where `oc` is 0). `ion_last` and `zimt_current`: ZimT's currents (`solve_transient`'s
    `zimt_ion_currents` and `zimt_current`). The steps' `track` and the setup's control `ctl`
    (`failureMode`, `convVar` and its tolerances) decide, as in ZimT's main loop, which steps
    are kept, where the run stops, and so each step's `Δt`: from the last step kept. `recycle`
    (NP × NP, or None): each step's generation adds the photons of the state before it that
    it reabsorbs (`solve_transient`). `profiles`: each step's state (`V`, `ln n`, `ln p` on every
    point, the contacts' included) as one more output. `drift`: the ions bounded by their sites
    flow with the modified drift (`solve_transient`'s `ion_flux`)."""
    if drift:  # a key, not a value: the flux chosen when tracing (`solver._drift_factor`)
        d = dict(d, ion=dict(d["ion"], drift=jnp.asarray(1.0)))
    s, ion = d["s"], d["ion"]
    V_first = jnp.where(oc[0] > 0, OC_GUESS, V[0])
    d0, s0, ion0, V0 = jax.lax.stop_gradient((d, s, ion, V_first))
    # The steady state at the first step (t = 0), from `batch._solve_one`'s start (`_start`).
    # The first step then starts from it.
    y, logs, _ = _start(d0, s0, ion0, V0, G_frac[0], B, tol, max_iter)  # solved again: its current
    y_s, logs_s, J_s, ok_s, _ = _newton_solve(y, logs, d0, s0, ion0, V0, G_frac[0], B, tol,
                                              max_iter)  # fmt: skip
    y, logs = jnp.where(ok_s, y_s, y), jnp.where(ok_s, logs_s, logs)
    Vt = d["Vt"]
    if ions:  # the ions' densities as unknowns: their steady profiles, 0 outside their regions
        la, lc = _point_logs(logs, ion)
        Vm = y[:, 0]
        za, zc = la + Vm / Vt, lc - Vm / Vt
        if "a_inv" in s:  # bounded by their sites: the occupancy's logarithm
            za = jnp.log(occupancy(za, s["a_inv"])[0])
            zc = jnp.log(occupancy(zc, s["c_inv"])[0])
        y = jnp.concatenate([y, jnp.stack([jnp.where(s["a_mask"] > 0, za, 0.0),
                                           jnp.where(s["c_mask"] > 0, zc, 0.0)], -1)],
                            -1)  # fmt: skip
    caps = _caps(Vt, y.shape[-1])
    v_cap = STEP_CAPS[0] * Vt
    traps = "ti" in s
    # The integral form's widths, once per device: not in every Newton step's current.
    width = _sum_widths(d["h"][: d["x"].shape[0] - 1], zimt_current)

    def full_of(y, Vint, dti_):  # the contacts as the current takes them (`_with_contacts`)
        full = _with_contacts(_full(_bounds(d, Vint), y), s, Vt, B)
        return _ion_full(full, ion, Vt, B, dti_) if ions else full

    vol = _volumes(s, d["L"])

    def stencils(before, dti_k, fb, fi):
        """The step's stencils and device: the state before it, the traps' occupations, and
        with `recycle` the photons of its radiative recombination reabsorbed, a generation."""
        s_k = _time_terms(s, before, dti_k, fb, fi if traps else None)
        if recycle is not None:
            n_b, p_b = jnp.exp(before[1:-1, 1]), jnp.exp(before[1:-1, 2])
            ni2 = _degenerate(s, n_b, p_b)["ni2"]  # another statistics' equilibrium
            emitted = s["kd"] * (n_b * p_b - ni2) * vol  # per area of each volume
            # none into a steady state (the first step, a return to t = 0)
            s_k = dict(s_k, G_rec=jnp.where(dti_k > 0, (recycle @ emitted) / vol, 0.0))
        pad = jnp.zeros_like(fb[:1])
        tr = {"dti": dti_k, "n": jnp.exp(before[:, 1]), "p": jnp.exp(before[:, 2]),
              "V": before[:, 0], "fb": jnp.concatenate([pad, fb, pad]), "width": width}  # fmt: skip
        if zimt_current:  # a key, not a value: the current's form chosen when tracing
            tr["zimt_current"] = True
        if ions:
            ma, mc = _ion_membership(ion)
            tr["ions"] = {"a": jnp.exp(before[:, 3]) * ma, "c": jnp.exp(before[:, 4]) * mc}
            tr["ion_last"] = jnp.asarray(float(ion_last))
        return s_k, dict(d, s=s_k, tr=tr)

    fm = ctl["failure_mode"]

    def summary(full, J, err, G):  # what two states compare by (`_states_equal`)
        if ions:
            ma, mc = _ion_membership(ion)
            a, c = jnp.exp(full[:, 3]) * ma, jnp.exp(full[:, 4]) * mc
        else:
            a = c = jnp.zeros_like(full[:, 0])
        return {"V": full[:, 0], "n": jnp.exp(full[:, 1]), "p": jnp.exp(full[:, 2]), "a": a,
                "c": c, "J": J, "err": err, "G": G}  # fmt: skip

    def step(carry, x):
        (y, logs, y_prev, logs_prev, V_now, V_before, J_ext_now, dti_now, fb, fi, first,
         t_last, track_curr, cur, ref, count, stop) = carry  # fmt: skip
        t_k, V_k, oc_k, G_k, track_k = x
        # Δt from the last step kept (ZimT's `Read_tVG` from `curr.tijd`); t = 0 the steady state.
        dti_k = jnp.where(t_k == 0, 0.0, 1.0 / jnp.where(t_k > t_last, t_k - t_last, 1.0))
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
        parts, points = _current_parts(d_k, _full(_bounds(d, V_new), y_new), G_k, B)
        columns = _columns(d, parts, points, J, J_ext)
        # The traps' occupations at the step's end, from its solution.
        after = full_of(y_new, V_new, dti_k)
        fb_new = _bulk_occupation(jnp.exp(after[1:-1, 1]), jnp.exp(after[1:-1, 2]), s_k)
        fi_new = (_interface_occupation(jnp.exp(after[:, 1]), jnp.exp(after[:, 2]), s_k["ti"])
                  if traps else fi)  # fmt: skip
        # ZimT's main loop. A step not converged: `failureMode` 0 stops the run, 1 keeps the step,
        # 2 skips it (keeps it in the steady state), but stops where the last step kept is the
        # reference (`Track` 1). A state that is not finite is never kept.
        running = stop == 0
        finite = (jnp.all(jnp.isfinite(y_new)) & jnp.all(jnp.isfinite(logs_new))
                  & jnp.isfinite(V_new))  # fmt: skip
        failed = running & ~ok & ((fm == 0) | ((fm == 2) & (track_curr == 1)))
        keep = running & ~failed & finite & (ok | (fm == 1) | ((fm == 2) & (dti_k == 0)))
        new = (y_new, logs_new, y, logs, V_new, V_now, J_ext, dti_k, fb_new, fi_new)
        old = (y, logs, y_prev, logs_prev, V_now, V_before, J_ext_now, dti_now, fb, fi)
        new = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b), new, old)
        cur = jax.tree_util.tree_map(lambda a, b: jnp.where(keep, a, b),
                                     summary(after, J, columns["errJ"], G_k), cur)  # fmt: skip
        track_curr = jnp.where(keep, track_k, track_curr)
        t_last = jnp.where(keep, t_k, t_last)
        # The automatic stop, by the last state kept (ZimT's `curr.Track`): 1 makes it the
        # reference where the step converged; 2 and 3 count equal states in a row (where the
        # step converged or failureMode is 1), 3 then makes it the reference.
        counted = running & ~failed & ((track_curr == 2) | (track_curr == 3)) & (ok | (fm == 1))
        equal = _states_equal(ref, cur, ctl)
        count = jnp.where(counted, jnp.where(equal, count + 1, 0), count)
        becomes_ref = running & ~failed & (((track_curr == 1) & ok)
                                           | ((track_curr == 3) & (ok | (fm == 1))))  # fmt: skip
        ref = jax.tree_util.tree_map(lambda a, b: jnp.where(becomes_ref, a, b), cur, ref)
        static = running & (count >= MIN_COUNT_STATIC)
        stop = jnp.where(failed, STOP_FAILED, jnp.where(static, STOP_STATIC, stop))
        carry = (*new, jnp.array(False), t_last, track_curr, cur, ref, count, stop)
        # a free end's boundary point its neighbour's state, as the solver reads it
        state = (
            ((_free_ends(after[:, :3], s) if "free" in s else after[:, :3]),) if profiles else ()
        )
        return carry, (J, J_ext, V_new, ok, it, columns, keep, *state)

    # Before the first step (t = 0, a steady state) the occupations do not matter.
    fb0 = jnp.zeros_like(s["Nt"])
    fi0 = jnp.zeros_like(s["ti"]["Nt"]) if traps else jnp.zeros(())
    J0 = jnp.where(ok_s, J_s, 0.0) + d["G_shunt"] * V0
    blank = summary(full_of(y, V0, 0.0), 0.0, 0.0, -1.0)
    zero = jnp.asarray(0)
    start = (y, logs, y, logs, V0, V0, J0, jnp.asarray(0.0), fb0, fi0, jnp.array(True),
             jnp.asarray(0.0), zero, blank, blank, zero, zero)  # fmt: skip
    end, out = jax.lax.scan(step, start, (t, V, oc, G_frac, track))
    return (*out, end[-1])


@functools.partial(jax.jit, static_argnames=("bernoulli_name", "max_iter", "ions", "vint",
                                             "ion_last", "zimt_current", "profiles",
                                             "drift"))  # fmt: skip
def _solve_batch(batch, t, V, oc, G_frac, track, ctl, recycle, tol, max_iter, bernoulli_name,
                 ions, vint, ion_last, zimt_current, profiles=False, drift=False):  # fmt: skip
    one = functools.partial(_solve_one, B=BERNOULLI[bernoulli_name], tol=tol, max_iter=max_iter,
                            ions=ions, vint=vint, ion_last=ion_last, zimt_current=zimt_current,
                            profiles=profiles, drift=drift)  # fmt: skip
    out = jax.vmap(one)(batch, t, V, oc, G_frac, track, ctl, recycle)
    J, J_ext, V_int, ok, its, columns, kept = out[:7]
    V_ext = V_int + J_ext * batch["R_series"][:, None]
    return J, J_ext, V_int, V_ext, ok, its, columns, kept, out[-1], (out[7] if profiles else None)


def solve_transient(batch, tvg: TVG | list[TVG], tol=1e-10, max_iter=200, bernoulli_name="exact",
                    zimt_ion_currents=False, ions=None, vint=None, zimt_current=False,
                    recycle=None, profiles=False,
                    ion_flux="electrochemical") -> TransientResult:  # fmt: skip
    """A batch (`batch.prepare`) through ZimT's time steps `tvg`: one for every device, or one
    per device (all of one length). Each step's Newton to `tol` or `max_iter`. Where any device
    of the batch has an open circuit or a series resistance, the internal voltage is an unknown
    of every device's steps (with the voltage given, its row `Vint = Vext`). Each device's
    `failureMode` and `Track` as ZimT's main loop has them (`TransientResult.accepted`, `stop`);
    the steps after a stop are solved too (the batch runs in lockstep) but not kept.
    `zimt_ion_currents`: with several regions of one species only the last one's current, as
    ZimT 5.36 has it (docs/differences.md §11); by default every region's. `zimt_current`: the
    integral form's sums as ZimT 5.36 has them (each point's change over the interval after
    it, the left contact's too), which converge at first order in the grid; by default over
    each point's control volume, the current the equations balance (docs/differences.md §13).
    `recycle` (jaxdd's): photon recycling, a matrix over the interior points per device
    (batch × NP × NP) or one for all (NP × NP), `recycle[i, j]` the share of the photons of point
    `j`'s radiative recombination absorbed in point `i`'s control volume
    (`recycling.device_kernel`); each step's generation adds them, from the state before the
    step (explicit in time, first order like the steps). A steady state (`t = 0`, a return to
    it) has none: the dark's `n·p = n_i²` emits nothing net, and under light it would need the
    recycled generation's own solution. `profiles`: each step's state on every point (the
    contacts' included), `TransientResult.profiles` `{"V", "n", "p"}` (batch × steps × points;
    V in volts, the densities in m⁻³), for band diagrams in time. `ion_flux` (`solver.ION_FLUXES`):
    how mobile ions bounded by their sites (`N_max_anion`, `N_max_cation`) flow, down their
    electrochemical potential at a constant mobility (`"electrochemical"`, ChargeTransport.jl's
    default) or with the drift modified by the free sites' share at a constant diffusivity
    (`"modified-drift"`, Abdel, Courtier, Farrell 2023; docs/model.md): the same steady states,
    other transients. Without sites the two are the same.

    Differentiable in the batch (`batch.device_tree`): each step's derivatives those of its
    converged solution by the implicit function (`batch._converged_jvp`), carried from step to
    step by the state, in forward mode (`jax.jvp`, `jax.jacfwd`). Where `batch` is traced the
    result holds JAX arrays, and `ions` (any mobile ions: 5×5 blocks) and `vint` (the internal
    voltage an unknown), which set the program's shape, must be given; else they are read from
    the batch."""
    n = jax.tree_util.tree_leaves(batch)[0].shape[0]
    tvgs = list(tvg) if isinstance(tvg, list | tuple) else [tvg] * n
    if len(tvgs) != n or len({len(t.t) for t in tvgs}) != 1:
        raise ValueError("one tVG for the batch, or one per device, all of one length")
    t, G, track = (jnp.asarray(np.stack([getattr(x, k) for x in tvgs]))
                   for k in ("t", "G_frac", "track"))  # fmt: skip
    ctl = {k: jnp.asarray([getattr(x, k) for x in tvgs])
           for k in ("failure_mode", "conv_var", "tol_pois", "tol_dens", "tol_curr")}  # fmt: skip
    oc = np.stack([t.oc for t in tvgs])
    V = jnp.asarray(np.where(oc, 0.0, np.stack([t.Vext for t in tvgs])))
    traced = _traced(batch)
    if traced and (ions is None or vint is None):
        raise ValueError("a traced batch needs `ions` and `vint` given")
    if ions is None:
        ion = batch["ion"]
        ions = bool(np.any(np.asarray(ion["a_totals"]) > 0)
                    or np.any(np.asarray(ion["c_totals"]) > 0))  # fmt: skip
    if ion_flux not in ION_FLUXES:
        raise ValueError(f"ion_flux {ion_flux!r}: one of {ION_FLUXES}")
    if vint is None:
        vint = bool(oc.any() or np.any(np.asarray(batch["R_series"]) > 0))
    if recycle is not None:
        NP = batch["s"]["G"].shape[-1]
        recycle = jnp.asarray(recycle)
        if recycle.ndim == 2:
            recycle = jnp.broadcast_to(recycle, (n, *recycle.shape))
        if recycle.shape != (n, NP, NP):
            raise ValueError(f"recycle: one {NP} × {NP} matrix, or one per device ({n})")
    with _quiet():
        J, J_ext, V_int, V_ext, ok, its, columns, kept, stop, prof = _solve_batch(
            batch, t, V, jnp.asarray(oc, dtype=float), G, track, ctl, recycle, tol, max_iter,
            bernoulli_name, bool(ions), bool(vint), bool(zimt_ion_currents),
            bool(zimt_current), bool(profiles), ion_flux == "modified-drift",
        )  # fmt: skip
    out = (lambda a: a) if traced else np.asarray
    xp = jnp if traced else np
    n_layers = xp.asarray(batch["layers"]["cells"]).any(axis=2).sum(axis=1)
    return TransientResult(*(out(a) for a in (J, J_ext, V_int, V_ext, ok, its)),
                           columns={k: out(v) for k, v in columns.items()},
                           n_layers=n_layers, accepted=out(kept), stop=out(stop),
                           profiles=None if prof is None else {
                               "V": out(prof[..., 0]), "n": out(xp.exp(prof[..., 1])),
                               "p": out(xp.exp(prof[..., 2]))})  # fmt: skip
