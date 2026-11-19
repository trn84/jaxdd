"""Many devices at once: the solver of `solver.py` as one pure function, `vmap`ped.

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
    BRANCH_FLIP_SIZE,
    ION_LOG_CAP,
    ROUND_OFF_FLOOR,
    STALL_STEPS,
    STEP_CAPS,
    _assemble,
    _assemble_ions,
    _balanced_logs,
    _bulk_rates,
    _contact_branch,
    _field_mobilities,
    _free_ends,
    _held,
    _interface_points,
    _ion_constraints,
    _ion_constraints_time,
    _ion_densities,
    _ion_full,
    _ion_logs,
    _ion_setup,
    _ions_on_grid,
    _on_points,
    _point_logs,
    _point_rates,
    _poisson_jac,
    _poisson_val,
    _recombination,
    _slot_columns,
    _slots,
    _stencils,
    _unbounded,
    _volumes,
    _with_contacts,
    bernoulli,
    density_steps,
    step_caps,
)
from jaxdd.sweep import block_sweep

__all__ = ["device_tree", "external_voltages", "prepare", "solve_batch", "solve_states"]

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
    tree = {
        "s": _stencils(dev, xp),
        "ion": _ion_setup(dev),
        "Vt": dev.Vt, "g_frac": dev.G_frac, "L": dev.L,
        "x": dev.x, "h": dev.h, "eps": dev.eps, "N_c": dev.N_c, "ni": dev.ni, "E_c": dev.E_c,
        "V0": dev.V0, "VL": dev.VL, "W_L": dev.W_L, "W_R": dev.W_R,
        "ln_n_left": xp.log(dev.n_left), "ln_n_right": xp.log(dev.n_right),
        "ln_p_left": xp.log(dev.p_left), "ln_p_right": xp.log(dev.p_right),
        "offset_n": dev.offset_n, "offset_p": dev.offset_p,
        "mu_n": dev.mu_n, "mu_p": dev.mu_p, "G": dev.G,
        "mu_a": xp.zeros(NP + 2) if dev.mu_anion is None else dev.mu_anion,
        "mu_c": xp.zeros(NP + 2) if dev.mu_cation is None else dev.mu_cation,
        "rec": {"kd": dev.k_direct, "ni2": dev.ni**2, "Nt": dev.srh_Nt, "Cn": dev.srh_Cn,
                "Cp": dev.srh_Cp, "nt0": dev.srh_nt0, "pt0": dev.srh_pt0},
        "iface": iface,
        "J_scale": xp.maximum(generated, 1.0),
        "R_series": dev.R_series,
        "left_elec": float(dev.left_elec),
        "G_shunt": xp.where(dev.R_shunt > 0, 1 / xp.where(dev.R_shunt > 0, dev.R_shunt, 1), 0.0),
        "pre_cond": float(dev.pre_cond), "V_pre": dev.V_pre, "fix_ions": float(dev.fix_ions),
        # (left, right): 1 where the device ends at a free surface (`Device.free`)
        "free": np.array([float(f) for f in dev.free]),
    }  # fmt: skip
    if dev.curr_diff:  # `currDiffInt = 1`; without the key the integral form alone
        tree["curr_diff"] = 1.0
    tree["layers"] = _layer_masks(dev)
    return tree


def _layer_masks(dev: Device) -> dict:
    """Each layer's intervals (`i0 ≤ i < i1`, layers × NP + 1) and its first and last points
    (layers × NP + 2), for the output's averages per layer (`transient._columns`)."""
    NP = len(dev.x) - 2
    nl = len(dev.i0)
    cells, first, last = np.zeros((nl, NP + 1)), np.zeros((nl, NP + 2)), np.zeros((nl, NP + 2))
    for j, (a, b) in enumerate(zip(dev.i0, dev.i1, strict=True)):
        cells[j, a:b] = 1.0
        first[j, a] = 1.0
        last[j, b] = 1.0
    return {"cells": cells, "first": first, "last": last}


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
    if any("fs_k" in t["s"] for t in trees):  # a free end (`solver._free_stencils`); the others
        for t in trees:  # none: no surface, no gauge
            for k in ("fs_n", "fs_p", "fs_k", "fs_ni"):
                t["s"].setdefault(k, np.zeros_like(t["s"]["hm"]))
            t["s"].setdefault("gauge", np.zeros(()))
            t["s"].setdefault("gauge_on", np.zeros(()))
            t["s"].setdefault("gauge_col", np.full((), 2.0))
            t["ion"].setdefault("vol", np.zeros(len(t["x"])))
            t["ion"].setdefault("by_volume", np.zeros(()))
            t["s"].setdefault("free", np.zeros(2))
    with_traps = [t["s"]["ti"] for t in trees if "ti" in t["s"]]
    if with_traps:  # the others with none: zero on every edge
        for t in trees:
            t["s"].setdefault("ti", {k: np.zeros_like(v) for k, v in with_traps[0].items()})
    if any("ct" in t["s"] for t in trees):  # finite surface recombination; the others infinite
        for t in trees:
            t["s"].setdefault("ct", {"S": -np.ones(4), "a": np.ones(4), "both": np.asarray(0.0)})
    if any("fn_m" in t["s"] for t in trees):  # a field-dependent mobility; the others constant
        for t in trees:
            for k in ("fn_m", "fn_p", "fp_m", "fp_p"):
                t["s"].setdefault(k, np.zeros_like(t["s"]["hm"]))
    if any("kln" in t["s"] for t in trees):  # Langevin in the field; the others' rate constant
        for t in trees:
            for k in ("kln", "klp", "klb"):
                t["s"].setdefault(k, np.zeros_like(t["s"]["hm"]))
    braun = {"br_romberg" in t["s"] for t in trees if "br_on" in t["s"]}
    if len(braun) > 1:
        raise ValueError("a batch takes one Braun rule (build_device's braun)")
    if braun:  # field-dependent generation; the others none
        for t in trees:
            for k in ("br_on", "br_P0", "br_lpre", "br_bq", "br_cF", "br_lkf"):
                t["s"].setdefault(k, np.zeros_like(t["s"]["hm"]))
            for k in ("br_kind", "br_cE"):
                t["s"].setdefault(k, np.ones_like(t["s"]["hm"]))
            if True in braun:
                t["s"].setdefault("br_romberg", np.zeros_like(t["s"]["hm"]))
    if any("a_inv" in t["s"] for t in trees):  # ions bounded by their sites; the others none
        for t in trees:
            t["s"].setdefault("a_inv", np.zeros_like(t["s"]["hm"]))
            t["s"].setdefault("c_inv", np.zeros_like(t["s"]["hm"]))
            t["ion"].setdefault("a_inv", np.zeros(len(t["x"])))
            t["ion"].setdefault("c_inv", np.zeros(len(t["x"])))
    if any("tq" in t["s"] for t in trees):  # charged bulk traps; the others none
        for t in trees:
            t["s"].setdefault("tq", np.zeros_like(t["s"]["Nt"]))
            t["s"].setdefault("tcwe", np.zeros_like(t["s"]["Nt"]))
    for sp in ("a", "c"):  # each species' slots (`solver._slots`) to the most of any device
        most = max(t["ion"][f"{sp}_slots"].shape[0] for t in trees)
        for t in trees:
            extra = most - t["ion"][f"{sp}_slots"].shape[0]
            if extra:  # empty slots: no ions, their normalisation held
                slots, totals = t["ion"][f"{sp}_slots"], t["ion"][f"{sp}_totals"]
                t["ion"][f"{sp}_slots"] = np.concatenate([slots, np.zeros((extra, slots.shape[1]))])
                t["ion"][f"{sp}_totals"] = np.concatenate([totals, np.zeros(extra)])
    if any("curr_diff" in t for t in trees):  # the differential current; the others integral
        for t in trees:
            t.setdefault("curr_diff", 0.0)
    most = max(t["layers"]["cells"].shape[0] for t in trees)  # layers: padded with empty ones
    for t in trees:
        t["layers"] = {k: np.pad(v, ((0, most - len(v)), (0, 0))) for k, v in t["layers"].items()}
    _pad_levels(trees)
    return jax.tree_util.tree_map(lambda *leaves: jnp.asarray(np.stack(leaves)), *trees)


def _pad_levels(trees: list[dict]) -> None:
    """Trap levels to one count across a batch: an array over points (one level) or points ×
    levels (a trap file), padded with levels of density 0, which do nothing. The capture
    coefficients carry the levels only where a device gives its levels their own
    (`device.TRAP_FILE_CAPTURE`); then every device's do, a coefficient over points being its
    every level's, and padded with the last level's."""
    groups = (("s", ("Nt", "nt0", "pt0", "tq", "tcwe")), ("rec", ("Nt", "nt0", "pt0")),
              ("ti", ("Nt", "nt0_lo", "pt0_lo", "nt0_hi", "pt0_hi")))  # fmt: skip
    for group, keys in groups:
        dicts = [t["s"].get("ti") if group == "ti" else t[group] for t in trees]
        if dicts[0] is None:
            continue
        K = max(np.shape(d[k])[1] if np.ndim(d[k]) == 2 else 1 for d in dicts for k in keys
                if k in d)  # fmt: skip
        if K > 1:
            for d in dicts:
                for k in keys:
                    if k in d:
                        a = np.asarray(d[k])
                        a = a[:, None] if a.ndim == 1 else a
                        d[k] = np.pad(a, ((0, 0), (0, K - a.shape[1])))
        if any(np.ndim(d["Cn"]) == 2 for d in dicts):  # the levels' own capture coefficients
            for d in dicts:
                for k in ("Cn", "Cp"):
                    a = np.asarray(d[k])
                    a = np.repeat(a[:, None], K, axis=1) if a.ndim == 1 else a
                    d[k] = np.pad(a, ((0, 0), (0, K - a.shape[1])), mode="edge")


def _bounds(d, Vint):
    return jnp.array([
        [d["V0"] + 0.5 * d["left_elec"] * Vint, d["ln_n_left"], d["ln_p_left"]],
        [d["VL"] - 0.5 * d["left_elec"] * Vint, d["ln_n_right"], d["ln_p_right"]],
    ])  # fmt: skip


def _full(bounds, y):
    """`y` with the contacts' rows from `bounds`, padded to `y`'s columns (with mobile ions in time
    the ions' two, filled where they matter by `solver._ion_full`)."""
    pad = y.shape[-1] - bounds.shape[-1]
    b = jnp.pad(bounds, ((0, 0), (0, pad))) if pad else bounds
    return jnp.concatenate([b[0][None], y, b[1][None]])


def _caps(Vt, width):
    """Newton's step caps for `width` unknowns per point: the ions' logarithms as the densities'."""
    caps = step_caps(Vt)
    return jnp.concatenate([caps, jnp.full(width - 3, STEP_CAPS[1])]) if width > 3 else caps


def _initial(d, Vint):
    """SIMsalabim's `Init_Pot_Dens_Ions_Traps`: linear potential, linear Fermi level. With a
    free end `solver._initial_free`'s start: with one contact its potential and Fermi level
    everywhere, with none the potential 0 and the balanced densities (`_balanced_logs`)."""
    x = d["x"]
    e = d["left_elec"]  # −1: the left electrode the cathode, 1 the anode
    V = d["V0"] + 0.5 * e * Vint + x * (d["VL"] - d["V0"] - e * Vint) / d["L"]
    EF = d["W_L"] - x * (d["W_L"] - d["W_R"]) / d["L"]
    ln_n = jnp.log(d["N_c"]) - (EF - d["E_c"]) / d["Vt"]
    ln_p = 2 * jnp.log(d["ni"]) - ln_n
    linear = jnp.stack([V, ln_n, ln_p], -1)
    if "fs_k" not in d["s"]:
        return linear
    left, right = d["free"][0] > 0, d["free"][1] > 0
    V_c = jnp.where(left, d["VL"] - 0.5 * e * Vint, d["V0"] + 0.5 * e * Vint)
    W = jnp.where(left, d["W_R"], d["W_L"])
    ln_n = jnp.log(d["N_c"]) - (W - d["E_c"]) / d["Vt"]
    flat = jnp.stack([jnp.full_like(x, V_c), ln_n, 2 * jnp.log(d["ni"]) - ln_n], -1)
    s = d["s"]
    ln_n, ln_p = _balanced_logs(s["dop"], d["ni"][1:-1], d["N_c"][1:-1], d["Vt"],
                                d["g_frac"] * d["G"][1:-1], s, _volumes(s, d["L"]))  # fmt: skip
    edge = functools.partial(jnp.pad, pad_width=1, mode="edge")
    balanced = jnp.stack([jnp.zeros_like(x), edge(ln_n), edge(ln_p)], -1)
    return jnp.where(left & right, balanced, jnp.where(left | right, flat, linear))


def _current(d, full, g_frac, B=bernoulli):
    """SIMsalabim's `Calc_Curr_Int` for both carriers, or `Calc_Curr_Diff` where `curr_diff`
    is 1, averaged (`Calc_All_Currents`); the contacts' densities of finite surface
    recombination from `full`'s neighbours (Bernoulli function `B`). With `d["tr"]`, a time
    step from the state before it (`transient`): the change of each density in the integral
    form's integrand (summed as `_sum_widths` has it), the mobile ions' currents and the
    displacement current (`Calc_Displacement_Curr`). Its parts: `_current_parts`."""
    parts, _ = _current_parts(d, full, g_frac, B)
    hh = d["h"][: d["x"].shape[0] - 1]
    J = jnp.sum(hh * sum(parts.values())) / jnp.sum(hh)
    # A free end: in the steady state no current anywhere (`solver._current`).
    return jnp.where(jnp.any(d["free"] > 0), 0.0, J) if "free" in d else J


def _current_parts(d, full, g_frac, B=bernoulli):
    """`_current`'s parts on each interval (NP + 1), with SIMsalabim's sign (turned with the
    anode on the left): `Jn`, `Jp`, and over a time step `Ja`, `Jc` (the anions' and cations',
    with mobile ions) and `JD`; and on the points what the output's columns take: the densities,
    the generation and each carrier's direct, bulk and interface recombination."""
    full = _with_contacts(full, d["s"], d["Vt"], B)
    if "free" in d["s"]:  # a free end's boundary point is its neighbour (`solver._free_ends`)
        full = full.at[:, :3].set(_free_ends(full[:, :3], d["s"]))
    V, n, p = full[:, 0], jnp.exp(full[:, 1]), jnp.exp(full[:, 2])
    Vt, x, h, L = d["Vt"], d["x"], d["h"], d["L"]
    NP = x.shape[0] - 2
    rec, G = d["rec"], d["G"]
    if "kln" in d["s"] or "br_on" in d["s"]:  # the rates in the field at `V`
        kd, g_fac = _point_rates(V[:-2], V[1:-1], V[2:], d["s"])
        rec = dict(rec, kd=rec["kd"].at[1:-1].set(kd))
        G = G.at[1:-1].multiply(g_fac)
    tr = d.get("tr")
    if tr is not None and "fb" in tr:  # over a time step: each carrier's capture (`_bulk_rates`)
        Rn, Rp = _bulk_rates(n, p, dict(rec, tr={"dti": tr["dti"], "fb": tr["fb"]}))
    else:
        Rn = Rp = _recombination(n, p, rec)
    R_int = (jnp.zeros_like(Rn), jnp.zeros_like(Rn))
    if "ti" in d["s"]:
        R_int = _interface_points(n, p, d["s"]["ti"])[:2]
    dx = jnp.diff(x[: NP + 1])
    width = h[: NP + 1] if tr is None else tr["width"]
    zimt = tr is not None and "zimt_current" in tr

    def carrier(sn, Vg, dens, mu, R, R_i, before):
        U = R + R_i - G * g_frac
        U = U.at[0].set(0.0).at[-1].set(0.0)
        if tr is not None:
            # ZimT's takes the left contact's change too (a finite surface recombination moves
            # it), in half an interval; no equation stores anything there.
            change = tr["dti"] * (dens - before)
            U = U + (change if zimt else change.at[0].set(0.0))
        ev = jnp.exp(sn * Vg[: NP + 1] / Vt)
        int_u = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * dx * (U[:NP] + U[1 : NP + 1]))])
        single = jnp.sum(0.5 * dx * (ev[:-1] / mu[:NP] + ev[1:] / mu[1 : NP + 1]))
        double = jnp.sum(
            0.5 * dx * (ev[:-1] * int_u[:-1] / mu[:NP] + ev[1:] * int_u[1:] / mu[1 : NP + 1])
        )
        K = (Vt * (sn * dens[NP + 1] * jnp.exp(sn * Vg[NP + 1] / Vt)
                   - sn * dens[0] * jnp.exp(sn * Vg[0] / Vt)) - sn * double) / single  # fmt: skip
        J = Q * K + jnp.concatenate(
            [jnp.zeros(1), jnp.cumsum(sn * Q * L * width[1 : NP + 1] * U[1 : NP + 1])]
        )
        if "curr_diff" in d:  # each interval's Scharfetter–Gummel flux (`Calc_Curr_Diff`)
            dV = sn * jnp.diff(Vg) / Vt
            flux = sn * Q * mu[: NP + 1] * Vt * (dens[1:] * B(-dV) - dens[:-1] * B(dV))
            J = jnp.where(d["curr_diff"] > 0, flux / (L * h[: NP + 1]), J)
        # Across an interface SIMsalabim takes the current before it plus the current through
        # the interface traps.
        through = sn * 0.5 * Q * L * h[: NP + 1] * (R_i[: NP + 1] + R_i[1:])
        return jnp.where(d["iface"], jnp.roll(J, 1) + through, J)

    mu_n, mu_p = _field_mobilities(d["mu_n"], d["mu_p"], V, d["s"])
    before_n, before_p = (None, None) if tr is None else (tr["n"], tr["p"])
    # With the anode on the left SIMsalabim turns every current's sign.
    sign = -d["left_elec"]
    parts = {
        "Jn": sign * carrier(-1.0, V + d["offset_n"], n, mu_n, Rn, R_int[0], before_n),
        "Jp": sign * carrier(1.0, V + d["offset_p"], p, mu_p, Rp, R_int[1], before_p),
    }
    if tr is not None and "ions" in tr:  # mobile ions in time (ZimT's Jnion, Jpion)
        Ja, Jc = _ion_currents(d, _ion_full(full, d["ion"], Vt, B, tr["dti"]), tr, B)
        parts["Ja"], parts["Jc"] = sign * Ja, sign * Jc
    if tr is not None:  # the displacement current on each interval
        dV, dV_before = jnp.diff(V), jnp.diff(tr["V"])
        parts["JD"] = sign * d["eps"][: NP + 1] * (dV - dV_before) * tr["dti"] / (L * h[: NP + 1])
    direct = rec["kd"] * (n * p - rec["ni2"])
    points = {"V": V, "n": n, "p": p, "gen": G * g_frac, "direct": direct, "bulk_n": Rn - direct,
              "bulk_p": Rp - direct, "int_n": R_int[0], "int_p": R_int[1]}  # fmt: skip
    return parts, points


def _sum_widths(hh, zimt):
    """The width each point's `U` takes in the integral form's sum along the device
    (`J[i] = J[i − 1] ± q·L·width[i]·U[i]`, over the intervals' `hh`). SIMsalabim's is the
    interval after the point, `h[i]`: the steady state's (`_current_parts` without `tr`), where
    the two carriers' sums cancel in the total, and ZimT's in time (`zimt`). Over a time step it
    is by default the point's control volume, `(h[i − 1] + h[i])/2`, in which its continuity
    equation and Poisson's hold: the densities' change is then the displacement current's
    change from interval to interval, and the total is the same on every interval
    (docs/differences.md §13). Computed once per device (`tr["width"]`), not in every Newton
    step's current: on the A100 that cost the transients with ions 13 %."""
    if zimt:
        return hh
    return jnp.concatenate([hh[:1], 0.5 * (hh[:-1] + hh[1:])])


def _ion_currents(d, full, tr, B):
    """The mobile ions' currents on each interval (NP + 1) over a time step, each region on its own
    (ZimT's `Calc_All_Currents`: `Calc_Curr_Int` of the region with the densities' change as its
    integrand, or `Calc_Curr_Diff` with `curr_diff`), as the carriers' (`_current`): no
    recombination, so the current across a layer interface is the one before it; 0 outside.
    The anions' and the cations' apart. With `tr["ion_last"]` 1 each species' last region's
    alone, as ZimT 5.36 has it with several (each region's `Calc_Curr_Int` zeroes the current
    outside it, the ones before it among it; docs/differences.md §11)."""
    Vt, x, h, L, ion = d["Vt"], d["x"], d["h"], d["L"], d["ion"]
    NP = x.shape[0] - 2
    V = full[:, 0]
    dx, hh = jnp.diff(x), h[: NP + 1]
    width = tr["width"]
    masks, totals, _ = _slots(ion)
    ka = ion["a_slots"].shape[0]
    out = [jnp.zeros(NP + 1), jnp.zeros(NP + 1)]
    idx = jnp.arange(masks.shape[0])
    last_a = jnp.max(jnp.where((totals > 0) & (idx < ka), idx, -1))
    last_c = jnp.max(jnp.where((totals > 0) & (idx >= ka), idx, -1))
    only_last = tr.get("ion_last", 0.0) > 0
    for k in range(masks.shape[0]):
        anion = k < ka
        sn = -1.0 if anion else 1.0
        dens = jnp.exp(full[:, 3] if anion else full[:, 4])
        before = tr["ions"]["a"] if anion else tr["ions"]["c"]
        mu = d["mu_a"] if anion else d["mu_c"]
        mask = masks[k]
        pair = mask[1:] * mask[:-1]
        U = tr["dti"] * (dens - before)
        ev = jnp.exp(sn * V / Vt)
        mu_ = jnp.where(mu > 0, mu, 1.0)
        int_u = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * dx * (U[:-1] + U[1:]) * pair)])
        single = jnp.sum(0.5 * dx * (ev[:-1] / mu_[:-1] + ev[1:] / mu_[1:]) * pair)
        double = jnp.sum(0.5 * dx * (ev[:-1] * int_u[:-1] / mu_[:-1] + ev[1:] * int_u[1:] / mu_[1:])
                         * pair)  # fmt: skip
        first = jnp.argmax(mask > 0)
        last = NP + 1 - jnp.argmax(mask[::-1] > 0)
        single = jnp.where(single > 0, single, 1.0)
        ends = sn * (dens[last] * ev[last] - dens[first] * ev[first])
        K = (Vt * ends - sn * double) / single
        e = jnp.arange(NP + 1)
        step = jnp.where(e > first, sn * Q * L * width * U[: NP + 1], 0.0) * pair
        J = Q * K + jnp.cumsum(step)
        if "curr_diff" in d:
            dV = sn * jnp.diff(V) / Vt
            flux = sn * Q * mu[: NP + 1] * Vt * (dens[1:] * B(-dV) - dens[:-1] * B(dV)) / (L * hh)
            J = jnp.where(d["curr_diff"] > 0, flux, J)
        J = jnp.where(d["iface"], jnp.roll(J, 1), J)
        counted = (totals[k] > 0) & (~only_last | (k == (last_a if anion else last_c)))
        out[0 if anion else 1] = out[0 if anion else 1] + jnp.where((pair > 0) & counted, J, 0.0)
    return tuple(out)


def _poisson_solve(y, d, s, ion, Vint, max_iter):
    """Poisson's equation alone, the densities following the potential (`solver._poisson_step`
    in a while loop)."""
    bounds = _bounds(d, Vint)
    Vt = d["Vt"]

    def body(carry):
        y, _, it = carry
        full = _full(bounds, y)
        V = full[:, 0]
        la, lc = _point_logs(_ion_logs(V, ion), ion)
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


def _residual(y, logs, bounds, s, ion, Vt, g_frac, B):
    """Newton's system at `(y, logs)`: the rows' residual (NP × 3) and the ions' constraints,
    a held normalisation's row its own value (an identity row, as in `_system`)."""
    full = _full(bounds, y)
    if y.shape[-1] == 5:  # mobile ions in time (`solver._assemble_ions`)
        F = _assemble_ions(full, s, Vt, g_frac, logs, ion, B)[0]
        g = _ion_constraints_time(_ion_full(full, ion, Vt, B, s["tr"]["dti"][0]), logs, ion)
        return F, g + _held(ion) * logs
    F = _assemble(full, s, Vt, g_frac, *_point_logs(logs, ion), B)[0]
    return F, _ion_constraints(full[:, 0], logs, ion) + _held(ion) * logs


def _system(y, logs, bounds, s, ion, Vt, g_frac, B):
    """`_residual` and its Jacobian in `(y, logs)`: `(F, g, blocks)`, the blocks
    `(A, B, C, U, W, D)` of the bordered system `[[T, U], [Wᵀ, D]]` — `T` block tridiagonal
    (`A, B, C`), `U` the rows' derivatives in the normalisations, `W` the constraints' in the
    potentials (column 0 of each point), `D` theirs in the normalisations."""
    if y.shape[-1] == 5:  # mobile ions in time: the ions' constraints in their own densities
        dti = s["tr"]["dti"][0]
        F, A, Bb, C, U = _assemble_ions(_full(bounds, y), s, Vt, g_frac, logs, ion, B)

        def in_time(y_, logs_):
            return _ion_constraints_time(_ion_full(_full(bounds, y_), ion, Vt, B, dti), logs_, ion)

        g = in_time(y, logs)
        dg_dy, D = jax.jacrev(in_time, argnums=(0, 1))(y, logs)  # K × NP × 5, K × K
        D = D + jnp.diag(_held(ion))
        return F, g, (A, Bb, C, U, jnp.moveaxis(dg_dy, 0, -1), D)
    la, lc = _point_logs(logs, ion)
    full = _full(bounds, y)
    F, A, Bb, C, U = _assemble(full, s, Vt, g_frac, la, lc, B)
    U = _slot_columns(U, ion)

    def constraints(V_interior, logs_):
        V_full = jnp.concatenate([bounds[0][:1], V_interior, bounds[1][:1]])
        return _ion_constraints(V_full, logs_, ion)

    g = constraints(full[1:-1, 0], logs)
    # K outputs (one per slot, `solver._slots`) of NP inputs: reverse mode, K passes (forward
    # mode took NP passes and 90 % of a Newton step).
    dg_dV, D = jax.jacrev(constraints, argnums=(0, 1))(full[1:-1, 0], logs)
    D = D + jnp.diag(_held(ion))
    W = jnp.zeros((y.shape[0], 3, logs.shape[0])).at[:, 0, :].set(dg_dV.T)
    return F, g, (A, Bb, C, U, W, D)


def _vint_row(y, Vint, target, d, g_frac, B):
    """The internal voltage's row, where it is an unknown (ZimT's `SimType` 2–4): with
    `target = (oc, Vext)`, the external current `Jext = Jint + Vint/R_shunt` at the open circuit
    (`oc` 1), over the device's current scale; else `Vext = Vint + Jext·R_series` (ZimT's
    `Residual_Current_Voltage`), in V_T, which with `R_series = 0` is `Vint = Vext`. With a free
    end the latter (`Vext` 0 at the open circuit): its current is 0 at every voltage."""
    oc, Vext = target
    if "free" in d:  # a free end: no current whatever the voltage, which only moves the gauge
        oc = jnp.where(jnp.any(d["free"] > 0), 0.0, oc)
    J_ext = _current(d, _full(_bounds(d, Vint), y), g_frac, B) + d["G_shunt"] * Vint
    return jnp.where(oc > 0, J_ext / d["J_scale"],
                     (Vint + d["R_series"] * J_ext - Vext) / d["Vt"])  # fmt: skip


def _residual_vint(y, logs, d, s, ion, target, g_frac, B):
    """`_residual` with the internal voltage the last of `logs` and its row (`_vint_row`)."""
    Vint = logs[-1]
    F, g = _residual(y, logs[:-1], _bounds(d, Vint), s, ion, d["Vt"], g_frac, B)
    return F, jnp.concatenate([g, _vint_row(y, Vint, target, dict(d, s=s), g_frac, B)[None]])


def _system_vint(y, logs, d, s, ion, target, g_frac, B):
    """`_system` with the internal voltage an unknown, the last of `logs`: its column (the rows'
    and constraints' derivatives in it, through the contacts) and its row (`_vint_row`, dense
    in the grid) one more border of the bordered system."""
    Vint, Vt = logs[-1], d["Vt"]
    F, g, (A, Bb, C, U, W, D) = _system(y, logs[:-1], _bounds(d, Vint), s, ion, Vt, g_frac, B)

    def rows(V):
        return _residual(y, logs[:-1], _bounds(d, V), s, ion, Vt, g_frac, B)

    _, (dF, dg) = jax.jvp(rows, (Vint,), (jnp.ones_like(Vint),))
    d_s = dict(d, s=s)
    h, (h_y, h_V) = jax.value_and_grad(_vint_row, argnums=(0, 1))(y, Vint, target, d_s, g_frac, B)
    K = D.shape[0]
    D = jnp.concatenate([jnp.concatenate([D, dg[:, None]], 1),
                         jnp.concatenate([jnp.zeros(K), h_V[None]])[None]])  # fmt: skip
    return F, jnp.concatenate([g, h[None]]), (
        A, Bb, C, jnp.concatenate([U, dF[..., None]], -1),
        jnp.concatenate([W, h_y[..., None]], -1), D,
    )  # fmt: skip


def _bordered_solve(blocks, r, rg):
    """`[[T, U], [Wᵀ, D]]·(x, z) = (r, rg)` (`_system`'s blocks): the block sweep on 1 + K
    right-hand sides and a K×K Schur complement (Sherman–Morrison–Woodbury, docs/solver.md)."""
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


def _newton_solve(y, logs, d, s, ion, Vint, g_frac, B, tol, max_iter, strict=False, vint=False):
    """Newton at one voltage until converged (module docstring) or `max_iter`:
    `(y, logs, J, converged, iterations)`. Differentiable in everything but the start: the
    derivatives are those of the converged solution (`_converged_jvp`), not of the iterations.
    With `vint` the internal voltage is an unknown, the last of `logs` (its start), and `Vint`
    the target `(oc, Vext)` of its row (`_vint_row`)."""
    guess = jax.lax.stop_gradient((y, logs))
    y, logs, J, done, it = _converged(B, max_iter, strict, vint, guess,
                                      (d, s, ion, Vint, g_frac), tol)  # fmt: skip
    return y, logs, J, done > 0, it.astype(int)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2, 3))
def _converged(B, max_iter, strict, vint, guess, params, tol):
    """`_newton_solve` from `guess`, without derivatives of its own; the flag and the count as
    floats, so that their tangents are zeros in every JAX (an integer output's tangent is
    `float0` in newer JAX and of the output's own type in older)."""
    d, s, ion, Vint, g_frac = params
    y, logs, J, done, it = _newton_loop(*guess, d, s, ion, Vint, g_frac, B, tol, max_iter,
                                        strict, vint)  # fmt: skip
    return y, logs, J, done.astype(float), jnp.asarray(it, dtype=float)


@_converged.defjvp
def _converged_jvp(B, max_iter, strict, vint, primals, tangents):
    """Implicit differentiation: at a root of `_residual`, `∂r/∂y·ẏ = −∂r/∂θ·θ̇`. One linear
    solve with Newton's own Jacobian, whatever the iterations were; its transpose, for reverse
    mode, the same sweep on the transposed blocks (`custom_linear_solve`). A voltage not
    converged (or not finite) gets zero derivatives, so that it cannot spoil a gradient summed
    over the curve."""
    guess, params, tol = primals
    out = _converged(B, max_iter, strict, vint, guess, params, tol)
    y, logs, _, done, it = out
    valid = (done > 0) & jnp.all(jnp.isfinite(y)) & jnp.all(jnp.isfinite(logs))
    y_dot, logs_dot, J_dot = _tangent(B, vint, valid, y, logs, params, tangents[1])
    return out, (y_dot, logs_dot, J_dot, jnp.zeros_like(done), jnp.zeros_like(it))


@functools.partial(jax.checkpoint, static_argnums=(0, 1))
def _tangent(B, vint, valid, y, logs, params, params_dot):
    """The converged solution's and current's tangents. Under `jax.checkpoint`, so that reverse
    mode keeps the solution per voltage, not its Jacobian blocks and the residual's
    linearisation: these are computed again in the backward pass."""
    y = jnp.where(valid, y, 0.0)
    logs = jnp.where(valid, logs, 0.0)
    # The tangents of integer and boolean leaves (masks, counts) as `float0`, which `jax.jvp`
    # takes for them: JAX before 0.4.36 or so hands a custom JVP rule their tangents as bools.
    params_dot = jax.tree_util.tree_map(
        lambda p, t: t if jnp.issubdtype(jnp.result_type(p), jnp.inexact)
        else np.zeros(np.shape(p), dtype=jax.dtypes.float0),
        params, params_dot,
    )  # fmt: skip

    def residual(params):
        d, s, ion, Vint, g_frac = params
        if vint:
            return _residual_vint(y, logs, d, s, ion, Vint, g_frac, B)
        return _residual(y, logs, _bounds(d, Vint), s, ion, d["Vt"], g_frac, B)

    _, (r, rg) = jax.jvp(residual, (params,), (params_dot,))
    d, s, ion, Vint, g_frac = params
    if vint:
        _, _, blocks = _system_vint(y, logs, d, s, ion, Vint, g_frac, B)
    else:
        _, _, blocks = _system(y, logs, _bounds(d, Vint), s, ion, d["Vt"], g_frac, B)
    # An invalid voltage: the identity, whose solve of a zero right-hand side is zero even
    # where its own blocks are not finite.
    eye = jnp.eye(y.shape[-1])
    K = logs.shape[0]
    identity = (0.0, eye, 0.0, 0.0, 0.0, jnp.eye(K))
    blocks = tuple(jnp.where(valid, b, jnp.broadcast_to(i, b.shape))
                   for b, i in zip(blocks, identity, strict=True))  # fmt: skip

    # One flat right-hand side: JAX transposes a linear solve only where every part of it is
    # linear in the tangents, and the constraints' part is a constant zero where no tangent
    # reaches the ions.
    def flat(f, blocks):
        def g(v):
            x, z = f(blocks, v[:-K].reshape(r.shape), v[-K:])
            return jnp.concatenate([x.ravel(), z])

        return g

    sol = jax.lax.custom_linear_solve(
        flat(_apply, blocks),
        -jnp.concatenate([r.ravel(), rg]),
        solve=lambda _, b: flat(_bordered_solve, blocks)(b),
        transpose_solve=lambda _, b: flat(_bordered_solve, _transposed(blocks))(b),
    )
    y_dot, logs_dot = sol[:-K].reshape(r.shape), sol[-K:]
    y_dot = jnp.where(valid, y_dot, 0.0)
    logs_dot = jnp.where(valid, logs_dot, 0.0)

    def current(y, logs, params):
        d, _, _, Vint, g_frac = params
        bounds = _bounds(d, logs[-1] if vint else Vint)
        return _current(d, _full(bounds, y), g_frac, B)

    _, J_dot = jax.jvp(current, (y, logs, params), (y_dot, logs_dot, params_dot))
    return y_dot, logs_dot, jnp.where(valid, J_dot, 0.0)


def _newton_loop(y, logs, d, s, ion, Vint, g_frac, B, tol, max_iter, strict, vint=False):
    """Newton's iterations of `_newton_solve`. With `vint` the internal voltage the last of
    `logs`, its step held to the potential's cap; the current's settling is then no stop (the
    voltage's last digits are what the open circuit wants)."""
    Vt = d["Vt"]

    def bounds_of(logs):
        return _bounds(d, logs[-1] if vint else Vint)

    contacts = "ct" in s

    def held(branch):  # the stencils with each contact's branch held (`solver._contact_ends`)
        return dict(s, ct=dict(s["ct"], branch=branch)) if contacts else s

    def full_of(y, logs):
        return _full(bounds_of(logs), y)

    def step(y, logs, branch):
        if vint:
            F, g, blocks = _system_vint(y, logs, d, held(branch), ion, Vint, g_frac, B)
        else:
            F, g, blocks = _system(y, logs, bounds_of(logs), held(branch), ion, Vt, g_frac, B)
        delta, d_logs = _bordered_solve(blocks, -F, -g)
        caps = _caps(Vt, y.shape[-1])
        stepv = density_steps(y, delta, jnp.clip(delta, -caps, caps), s["n_floor"])
        if vint:
            d_logs = jnp.concatenate([
                jnp.clip(d_logs[:-1], -ION_LOG_CAP, ION_LOG_CAP),
                jnp.clip(d_logs[-1:], -STEP_CAPS[0] * Vt, STEP_CAPS[0] * Vt),
            ])  # fmt: skip
            log_size = jnp.maximum(jnp.max(jnp.abs(d_logs[:-1])), jnp.abs(d_logs[-1]) / Vt)
        else:
            d_logs = jnp.clip(d_logs, -ION_LOG_CAP, ION_LOG_CAP)
            log_size = jnp.max(jnp.abs(d_logs))
        new = y + stepv
        dens = jnp.exp(new[:, 1:])
        weight = dens / (dens + s["n_floor"][:, None])
        size = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(stepv[:, 0])) / Vt, jnp.max(jnp.abs(stepv[:, 1:]) * weight)
            ),
            log_size,
        )
        return new, logs + d_logs, size

    def current(y, logs):
        return _current(d, full_of(y, logs), g_frac, B)

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
        if "free" in d:  # with a free end the current is 0 whatever the state: no stop
            settled = settled & ~jnp.any(d["free"] > 0)
        if vint:
            return (size < tol) | stalled | quadratic
        return (size < tol) | stalled | settled | quadratic

    def body(carry):
        y, logs, size_hist, J_hist, it, _, branch = carry
        y, logs, size = step(y, logs, branch)
        size_hist = jnp.roll(size_hist, -1).at[-1].set(size)
        J_hist = jnp.roll(J_hist, -1).at[-1].set(current(y, logs))
        it = it + 1
        done = converged(size_hist, J_hist, it)
        if contacts:  # a branch that does not hold near the solution: changed, histories anew
            now = _contact_branch(full_of(y, logs), s, Vt, B, branch)
            flip = (done | (size < BRANCH_FLIP_SIZE)) & jnp.any(now != branch)
            branch = jnp.where(flip, now, branch)
            size_hist = jnp.where(flip, hist0, size_hist)
            J_hist = jnp.where(flip, J0, J_hist)
            done = done & ~flip
        return y, logs, size_hist, J_hist, it, done, branch

    def cond(carry):
        *_, it, done, _ = carry
        return (~done) & (it < max_iter)

    branch0 = _contact_branch(full_of(y, logs), s, Vt, B) if contacts else jnp.zeros(4)
    y, logs, _, J_hist, it, done, _ = jax.lax.while_loop(
        cond, body, (y, logs, hist0, J0, 0, jnp.array(False), branch0)
    )
    return y, logs, J_hist[-1], done, it


def _freeze(s, ion, y, logs, Vt, flag):
    """With `flag`, the mobile ions of `y` as fixed densities (`solver._freeze`)."""
    V = y[:, 0]
    la, lc = _point_logs(logs, ion)
    anion, cation = _ion_densities(V, la, lc, s, Vt)
    s = dict(s)
    s["a_fix"] = jnp.where(flag, anion, s["a_fix"])
    s["c_fix"] = jnp.where(flag, cation, s["c_fix"])
    s["a_mask"] = jnp.where(flag, 0.0, s["a_mask"])
    s["c_mask"] = jnp.where(flag, 0.0, s["c_mask"])
    ion = dict(ion)
    for k in ("a_slots", "a_totals", "c_slots", "c_totals"):
        ion[k] = jnp.where(flag, 0.0, ion[k])
    return s, ion


def _solve_one(d, voltages, B, tol, max_iter, start_iter, strict, states=False):
    """One device's J-V curve: the pre-bias stage, then the scan outwards from it. `voltages`
    ascending; `max_iter` caps Newton per voltage of the scan, `start_iter` Poisson and Newton
    at the pre-bias (the final pass's cap in both passes: a first pass changes no device's
    start, and a low first cap does not fail the pre-bias, which a batch pays once). With
    `states` each voltage's state and what photoluminescence measures of it too
    (`_observed`)."""
    s, ion, Vt, g_frac = d["s"], d["ion"], d["Vt"], d["g_frac"]
    first = jnp.where(d["pre_cond"] > 0, d["V_pre"], voltages[0])
    # The start: no derivatives (`_newton_solve` has them, of its converged solution; reverse
    # mode has none through a while loop).
    d0, s0, ion0, first0 = jax.lax.stop_gradient((d, s, ion, first))
    bounds = _bounds(d0, first0)

    def poisson_start(s_, ion_):
        y = _poisson_solve(_initial(d0, first0)[1:-1], d0, s_, ion_, first0, start_iter)
        return y, _ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion_)

    # Ions bounded by their sites: from the solution with Boltzmann's, else from Poisson's
    # with the bounded ions (`solver.solve_jv`).
    sb, ionb = s0, ion0
    s0, ion0 = _unbounded(s0, ion0) if "a_inv" in s0 else (s0, ion0)
    y, logs = poisson_start(s0, ion0)
    if "tq" in s0:  # charged bulk traps: from the solution with them neutral (`solver.solve_jv`)
        neutral = dict(s0, tq=jnp.zeros_like(s0["tq"]))
        y_n, logs_n, _, ok_n, _ = _newton_solve(
            y, logs, d0, neutral, ion0, first0, d0["g_frac"], B, tol, start_iter, strict
        )
        y, logs = jnp.where(ok_n, y_n, y), jnp.where(ok_n, logs_n, logs)
    if "a_inv" in s:
        y_b, logs_b, _, ok_b, _ = _newton_solve(
            y, logs, d0, s0, ion0, first0, d0["g_frac"], B, tol, start_iter, strict
        )
        y_b, logs_b, _, ok_s, _ = _newton_solve(
            y_b, logs_b, d0, sb, ionb, first0, d0["g_frac"], B, tol, start_iter, strict
        )
        y_d, logs_d = poisson_start(sb, ionb)
        ok_b = ok_b & ok_s
        y, logs = jnp.where(ok_b, y_b, y_d), jnp.where(ok_b, logs_b, logs_d)
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
        if states:
            return carry, (J, ok, it, *_observed(d, s, ion, y_new, logs_new, Vint, g_frac, B))
        return carry, (J, ok, it)

    # Outwards from the pre-bias (docs/solver.md): up from the first voltage at or above it,
    # then, from the pre-bias solution again, down from the last below it. Without a pre-bias
    # `first` is the lowest voltage and the scan runs upwards only.
    n = voltages.shape[0]
    below = jnp.searchsorted(voltages, first)
    j = jnp.arange(n)
    order = jnp.where(j < n - below, below + j, n - 1 - j)
    start = jax.lax.stop_gradient((y, logs, y, logs, first, first))
    _, out = jax.lax.scan(scan, start, (voltages[order], j == n - below))
    J, ok, its, *rest = (jnp.zeros_like(a).at[order].set(a) for a in out)
    return (J, ok & pre_ok, its, *rest)


def _observed(d, s, ion, y, logs, Vint, g_frac, B):
    """A solution `y`'s state on all `NP + 2` points (`(V, ln n, ln p)`: the contacts' densities
    of finite surface recombination from their neighbours, a free end's boundary point its
    neighbour's), as `solver.solve_state` its implied `V_oc` and PLQY of direct
    recombination, and its ions (`solver._ions_on_grid`, NP + 2 × 2)."""
    full = _with_contacts(_full(_bounds(d, Vint), y), s, d["Vt"], B)
    if "free" in d:
        full = full.at[0].set(jnp.where(d["free"][0] > 0, full[1], full[0]))
        full = full.at[-1].set(jnp.where(d["free"][1] > 0, full[-2], full[-1]))
    n, p = jnp.exp(full[1:-1, 1]), jnp.exp(full[1:-1, 2])
    vol = _volumes(s, d["L"])
    k = d["rec"]["kd"][1:-1]
    if "kln" in s or "br_on" in s:  # the rate in the field
        V = full[:, 0]
        k = _point_rates(V[:-2], V[1:-1], V[2:], s)[0]
    rad = jnp.sum(vol * k * n * p)
    rad0 = jnp.sum(vol * k * d["rec"]["ni2"][1:-1])
    gen = g_frac * jnp.sum(vol * d["G"][1:-1])
    implied_voc = jnp.where(rad0 > 0, d["Vt"] * jnp.log(rad / jnp.where(rad0 > 0, rad0, 1.0)),
                            jnp.nan)  # fmt: skip
    plqy = jnp.where(gen > 0, (rad - rad0) / jnp.where(gen > 0, gen, 1.0), jnp.nan)
    return full, implied_voc, plqy, jnp.stack(_ions_on_grid(y, logs, s, ion, d["Vt"]), -1)


@functools.partial(
    jax.jit, static_argnames=("bernoulli_name", "max_iter", "start_iter", "strict", "states")
)
def _solve_batch(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict, states=False):
    """One pass. `_solve_one` wants the voltages ascending: sorted here, inside the jit (they
    may be traced), and the results put back in the caller's order: `(J, ok, its)`, with
    `states` followed by `_observed`'s per voltage."""
    B = BERNOULLI[bernoulli_name]
    one = functools.partial(_solve_one, B=B, tol=tol, max_iter=max_iter, start_iter=start_iter,
                            strict=strict, states=states)  # fmt: skip
    order = jnp.argsort(voltages)
    J, *rest = (a[:, jnp.argsort(order)]
                for a in jax.vmap(one, in_axes=(0, None))(batch, voltages[order]))  # fmt: skip
    # The shunt, at the internal voltage as SimSS has it.
    return (J + batch["G_shunt"][:, None] * voltages[None, :], *rest)


def _shard_map(f, mesh, in_specs, out_specs):
    """`shard_map` of JAX 0.12 (`jax.shard_map`) or of JAX 0.4.31 (`jax.experimental`), without
    the replication check: each device solves its own devices, nothing is replicated."""
    if hasattr(jax, "shard_map"):
        return jax.shard_map(f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False)
    from jax.experimental.shard_map import shard_map

    return shard_map(f, mesh, in_specs, out_specs, check_rep=False)


@functools.partial(
    jax.jit,
    static_argnames=("bernoulli_name", "max_iter", "start_iter", "strict", "mesh", "states"),
)
def _solve_sharded(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict, mesh,
                   states=False):  # fmt: skip
    """`_solve_batch` with the batch split along its first axis across `mesh`'s devices, each
    solving its share (no communication: the devices are independent)."""
    P = jax.sharding.PartitionSpec

    def local(b, v, tol):
        return _solve_batch(b, v, tol, max_iter, start_iter, bernoulli_name, strict, states)

    return _shard_map(local, mesh, (P("b"), P(), P()), P("b"))(batch, voltages, tol)


def _mesh(devices):
    """The devices a batch is split across (`solve_batch`'s `devices`), as a one-axis mesh, or
    None for one device."""
    devices = jax.local_devices() if devices is None else list(devices)
    return jax.sharding.Mesh(np.array(devices), ("b",)) if len(devices) > 1 else None


def _run(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict, mesh, states=False):
    """One pass on one device, or split across `mesh`: the batch padded to a whole number of
    devices per accelerator by repeating its last device, the padding dropped again, the results
    on one device."""
    if mesh is None:
        return _solve_batch(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict,
                            states)  # fmt: skip
    n = jax.tree_util.tree_leaves(batch)[0].shape[0]
    k = mesh.devices.size
    m = -(-n // k) * k
    if m > n:
        idx = np.concatenate([np.arange(n), np.full(m - n, n - 1)])
        batch = jax.tree_util.tree_map(lambda leaf: leaf[idx], batch)
    out = _solve_sharded(batch, voltages, tol, max_iter, start_iter, bernoulli_name, strict, mesh,
                         states)  # fmt: skip
    # The results through the host onto one device: what follows (the second pass's scatter,
    # the caller's sums) is then never an operation across the GPUs, which XLA would run with
    # collectives (NCCL); on a host with an old driver such a scatter hung both GPUs.
    return tuple(jnp.asarray(np.asarray(o)[:n]) for o in out)


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
                bernoulli_name="exact", strict=False, devices=None):  # fmt: skip
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
    current; for checking the shortcuts, at about one more step per voltage.
    `devices`: the accelerators the batch is split across, each solving its share in one
    process (docs/hardware.md); by default every local device of JAX's backend, so all the GPUs
    a process sees (`CUDA_VISIBLE_DEVICES` chooses). Not split where `batch` or `voltages` are
    traced."""
    voltages = jnp.asarray(voltages, dtype=float)
    mesh = None if _traced(batch, voltages) else _mesh(devices)
    with _quiet():
        return _passes(batch, voltages, tol, max_iter, first_max_iter, bernoulli_name, strict,
                       mesh)  # fmt: skip


def solve_states(batch, voltages=0.0, tol=1e-10, max_iter=60, first_max_iter=6,
                 bernoulli_name="exact", strict=False, devices=None):  # fmt: skip
    """The steady states of a batch (`prepare`) at `voltages` (one or several, internal) and
    what photoluminescence measures of them, as `solver.solve_state` per device: a dict of
    `V`, `n`, `p`, `qfls`, `anion`, `cation` (batch × voltages × NP + 2; `V` in its gauge),
    `implied_voc`, `plqy`,
    `J`, `converged`, `iterations` (batch × voltages). Solved as `solve_batch` (its arguments
    too), each device's voltages one after another; with a free end the voltage only moves the
    potential's gauge (one contact) or is not used (none), so one voltage is what such a batch
    needs. Differentiable as `solve_batch`: `implied_voc` and `plqy` in the parameters of
    `device_tree`'s devices."""
    voltages = jnp.atleast_1d(jnp.asarray(voltages, dtype=float))
    mesh = None if _traced(batch, voltages) else _mesh(devices)
    with _quiet():
        J, ok, its, full, voc, plqy, ions = _passes(
            batch, voltages, tol, max_iter, first_max_iter, bernoulli_name, strict, mesh, True
        )
    n, p = jnp.exp(full[..., 1]), jnp.exp(full[..., 2])
    ni2 = batch["rec"]["ni2"][:, None, :]
    return {"V": full[..., 0], "n": n, "p": p, "qfls": batch["Vt"][:, None, None]
            * jnp.log(n * p / ni2), "implied_voc": voc, "plqy": plqy, "J": J, "converged": ok,
            "iterations": its, "anion": ions[..., 0], "cation": ions[..., 1]}  # fmt: skip


def _passes(batch, voltages, tol, max_iter, first_max_iter, bernoulli_name, strict, mesh=None,
            states=False):  # fmt: skip
    """`solve_batch`'s one or two passes: `_solve_batch`'s outputs."""
    run = functools.partial(_run, voltages=voltages, tol=tol, bernoulli_name=bernoulli_name,
                            strict=strict, mesh=mesh, states=states)  # fmt: skip
    if first_max_iter is None or first_max_iter >= max_iter or _traced(batch, voltages):
        return run(batch, max_iter=max_iter, start_iter=max_iter)
    out = run(batch, max_iter=first_max_iter, start_iter=max_iter)
    idx, padded = _stragglers(out[1])
    if idx.size == 0:
        return out
    part = jax.tree_util.tree_map(lambda leaf: leaf[padded], batch)
    again = run(part, max_iter=max_iter, start_iter=max_iter)
    n = idx.size
    return tuple(a.at[idx].set(b[:n]) for a, b in zip(out, again, strict=True))
