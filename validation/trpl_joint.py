"""The eight measured decays of Kober-Czerny et al. (2025) fitted together with jaxdd, every end
free: the FAPbI3 film on glass (four: either side, two fluences), glass/SnO2/FAPbI3 (three) and
glass/FAPbI3/Spiro-OMeTAD (one).

One perovskite for all three samples: k_rad, mobility, photodoping (as acceptors), a deep trap
level and a shallow one that takes no holes (each level its own capture coefficients,
`trpl_common.TRAPS_OWN`), their capture coefficient and the shallow level's depth. One
glass/perovskite surface (the film's and the Spiro stack's) and one perovskite/air surface (the
film's and the SnO2 stack's). Each transport layer its own interface SRV, band edge and doping,
Spiro its mobility; one transfer velocity over both interfaces.

Least squares on log10(model/data) in log-spaced time bins, the model averaged over the 2 ns
bins as the measurement; the Jacobian by forward differences, all perturbed sets and all eight
decays in one batch; from the authors' values and from `--starts` more, spread about them. A set
whose decays do not all converge costs `PENALTY` per bin, so that no step goes there.

`--bottom-zone NM` tests a hypothesis: a zone of NM nm at each perovskite's bottom with its own
deep-trap density, one more parameter; it writes validation/trpl_joint_zoneNMnm.json.

    .venv/bin/python validation/trpl_joint.py [--starts N] [--seed S] [--max-nfev M]
        [--from-best] [--start-mu LOG10] [--bottom-zone NM [--start-zone LOG10]]

writes validation/trpl_joint.json. docs/validation.md, "TRPL without electrodes".
"""

import argparse
import json
import time

import numpy as np
import trpl_common
from lm_batch import least_squares_many
from scipy.optimize import least_squares
from trpl_common import (
    FLUENCE,
    REPO,
    T_PULSE,
    TRAPS_OWN,
    C,
    as_measured,
    binned,
    detected,
    film_data,
    fine_times,
    log_bins,
    pl_of,
    pulse,
    setup,
    sno2_data,
    spiro_data,
)  # fmt: skip

from jaxdd.batch import prepare
from jaxdd.device import device_arrays, device_parameters
from jaxdd.transient import solve_transient, tj_columns

#: (stack, lit through the glass, fluence) of each decay, in the data's order.
DECAYS = (
    ("film", True, FLUENCE), ("film", True, FLUENCE / 10),
    ("film", False, FLUENCE), ("film", False, FLUENCE / 10),
    ("SnO2", True, FLUENCE), ("SnO2", True, FLUENCE / 10**0.5), ("SnO2", False, FLUENCE),
    ("Spiro", False, FLUENCE),
)  # fmt: skip
#: Each stack's perovskite layers (with `--bottom-zone` two: the zone and the rest).
PEROVSKITE_LAYERS = {"film": (0,), "SnO2": (1,), "Spiro": (0,)}
ZONE = None  # with `--bottom-zone`: each stack's zone, its layer index (`use_bottom_zone`)
DETECT = None  # with `--detect`: "same" (the lit side) or "opposite" (`use_detection`)
#: name, start (the authors' values where they have them), lower and upper bound
PARAMETERS = (
    ("log10 k_rad (m³/s)", np.log10(7.5e-17), -19, -14),
    ("log10 μ FAPbI3 (m²/Vs)", np.log10(4.1e-5), -8, -2),
    ("log10 photodoping (m⁻³)", np.log10(2.07e21), 18, 23),
    ("log10 C (m³/s)", np.log10(C), -18, -10),
    ("log10 N_deep (m⁻³)", np.log10(1.6e4 / C), 14, 22),
    ("log10 N_shallow (m⁻³)", np.log10(1.3e6 / C), 14, 24),
    ("shallow level below E_c (eV)", 0.261, 0.02, 0.6),
    ("log10 S glass/FAPbI3 (m/s)", np.log10(0.072), -5, 3),
    ("log10 S FAPbI3/air (m/s)", np.log10(0.036), -5, 3),
    ("log10 S SnO2/FAPbI3 (m/s)", np.log10(0.09), -5, 3),
    ("E_c SnO2 (eV)", 4.1, 4.0, 4.6),
    ("log10 N_D SnO2 (m⁻³)", 24.0, 20, 25.5),
    ("log10 S FAPbI3/Spiro (m/s)", np.log10(1.0), -5, 3),
    ("E_v Spiro (eV)", 5.2, 5.0, 5.5),
    ("log10 N_A Spiro (m⁻³)", 24.0, 20, 25.5),
    ("log10 μ Spiro (m²/Vs)", -7.0, -10, -4),
    ("log10 ν_int (m/s)", 5.0, 0, 7),
)
NAMES = tuple(p[0] for p in PARAMETERS)
START = np.array([p[1] for p in PARAMETERS])
LO = np.array([p[2] for p in PARAMETERS])
HI = np.array([p[3] for p in PARAMETERS])
FLOOR = 3e-3  # the data below this are noise


def use_detection(side, sublayers=12):
    """The PL as a detector at 806 nm sees it (`trpl_common.detected`): the perovskite in
    `sublayers` layers, the detector on the lit side ("same") or the other ("opposite"); two more
    parameters, the absorption at 806 nm (start: the authors' ellipsometry, 4.4e4 cm⁻¹ at the
    top, 6.2e4 at the bottom) and the far side's reflection."""
    global NAMES, START, LO, HI, DETECT, PEROVSKITE_LAYERS
    stacks = trpl_common.with_sublayers(sublayers)
    trpl_common.STACKS.update(stacks)
    PEROVSKITE_LAYERS = {name: stacks[name][1] for name in stacks}
    _BASE.clear()
    DETECT = side
    NAMES = (*NAMES, "log10 α at 806 nm (1/m)", "reflection at the far side")
    START, LO, HI = np.r_[START, np.log10(5.3e6), 0.3], np.r_[LO, 5.5, 0.0], np.r_[HI, 8.0, 1.0]


def use_bottom_zone(thickness):
    """The hypothesis of a zone of `thickness` (m) at each perovskite's bottom, with its own deep
    traps (`trpl_common.with_bottom_zone`): one more parameter, the zone's deep-level density."""
    global NAMES, START, LO, HI, ZONE, PEROVSKITE_LAYERS
    stacks, ZONE = trpl_common.with_bottom_zone(thickness)
    trpl_common.STACKS.update(stacks)
    PEROVSKITE_LAYERS = {name: stacks[name][1] for name in stacks}
    _BASE.clear()
    NAMES = (*NAMES, "log10 N_deep bottom zone (m⁻³)")
    START, LO, HI = np.r_[START, 21.0], np.r_[LO, 14.0], np.r_[HI, 24.0]


PENALTY = 3.0  # decades: a decay not converged, or not finite, in every bin

_BASE = {}


def _base(stack, glass):
    """The stack's structure and parameters, lit from that side, at a fluence of 1 photon/m²."""
    if (stack, glass) not in _BASE:
        _BASE[stack, glass] = device_parameters(
            setup(stack, lit_from_glass=glass, fluence=1.0, traps=TRAPS_OWN)
        )
    return _BASE[stack, glass]


def device(theta, stack, glass, fluence):
    st, base = _base(stack, glass)
    (k_rad, mu, N_A, Cn, N_deep, N_sh, dE, S_glass, S_air, S_sno2, Ec_sno2, ND_sno2, S_spiro,
     Ev_spiro, NA_spiro, mu_spiro, nu) = theta[:17]  # fmt: skip
    p = {k: np.array(v, dtype=float, copy=True) if isinstance(v, np.ndarray) else v
         for k, v in base.items()}  # fmt: skip
    for k in PEROVSKITE_LAYERS[stack]:
        p["k_direct"][k] = 10.0**k_rad
        p["mu_n"][k] = p["mu_p"][k] = 10.0**mu
        p["N_A"][k] = 10.0**N_A
        p["bulk_trap_N"][k] = [10.0**N_deep, 10.0**N_sh]
        p["bulk_trap_E"][k, 1] = p["E_c"][k] + dE
        p["bulk_trap_Cn"][k] = [10.0**Cn, 10.0**Cn]
        p["bulk_trap_Cp"][k] = [10.0**Cn, 0.0]
    if ZONE is not None:  # the bottom zone's own deep level
        p["bulk_trap_N"][ZONE[stack], 0] = 10.0 ** theta[17]
    last = PEROVSKITE_LAYERS[stack][-1]
    p["gen_G"] = p["gen_G"] * fluence
    S = {"S_n_L": 0.0, "S_p_L": 0.0, "S_n_R": 0.0, "S_p_R": 0.0}
    if stack == "film":
        S.update(S_n_L=10.0**S_glass, S_p_L=10.0**S_glass, S_n_R=10.0**S_air, S_p_R=10.0**S_air)
    elif stack == "SnO2":  # SnO2 on the glass; the perovskite's top in air
        S.update(S_n_R=10.0**S_air, S_p_R=10.0**S_air)
        p["N_t_int"][0] = 10.0**S_sno2 / 1e-14  # the interface traps' C (trpl_common.interface)
        p["E_c"][0], p["N_D"][0] = Ec_sno2, 10.0**ND_sno2
        p["nu_int_n"][0] = p["nu_int_p"][0] = 10.0**nu
    else:  # the perovskite on the glass; Spiro's top in air
        S.update(S_n_L=10.0**S_glass, S_p_L=10.0**S_glass)
        p["N_t_int"][last] = 10.0**S_spiro / 1e-14
        p["E_v"][-1], p["N_A"][-1] = Ev_spiro, 10.0**NA_spiro
        p["mu_n"][-1] = p["mu_p"][-1] = 10.0**mu_spiro
        p["nu_int_n"][last] = p["nu_int_p"][last] = 10.0**nu
    p.update(S)
    return device_arrays(st, p)


class Problem:
    """The eight decays binned (log-spaced, the model the same way), and the model's residuals."""

    def __init__(self):
        self.t = fine_times()
        self.tm = self.t[1:] - T_PULSE
        self.data = []  # per decay: (t, its points, which bin each point is in, binned data)
        for td, y in (film_data(), sno2_data(), spiro_data()):
            which = log_bins(td, td.max(), 60)
            for row in y:
                yb = binned(row[None], which)[0]
                self.data.append((td, which, np.where(yb > FLOOR, yb, np.nan)))
        self.keep = np.isfinite(np.concatenate([d[2] for d in self.data]))
        self.solves = 0

    def residuals_many(self, thetas):
        """Each set's residuals and whether all its decays converged and are finite: a decay
        that is not has `PENALTY` in every bin, so that a step to it is never taken."""
        devices = [device(th, *dec) for th in thetas for dec in DECAYS]
        r = solve_transient(prepare(devices), pulse(self.t), ions=False, vint=False)
        self.solves += len(devices)
        nl = [int(n) for n in np.asarray(r.n_layers)]
        conv = np.asarray(r.converged).reshape(len(thetas), len(DECAYS), -1).all(axis=2)
        out, good = [], []
        for i in range(len(thetas)):
            res, fine = [], True
            for j, (td, which, yb) in enumerate(self.data):
                k = i * len(DECAYS) + j
                with np.errstate(invalid="ignore", divide="ignore"):
                    if DETECT is None:
                        pl = pl_of(tj_columns(r, k), nl[k])
                    else:  # as the detector sees it
                        stack, glass, _ = DECAYS[j]
                        side = glass if DETECT == "same" else not glass
                        th = thetas[i]
                        seen = detected(tj_columns(r, k), PEROVSKITE_LAYERS[stack], 10.0 ** th[-2],
                                        th[-1], side)  # fmt: skip
                        pl = seen[1:] / seen[1]
                    model = as_measured(self.tm, pl, td)
                    rj = np.log10(binned(model[None], which)[0]) - np.log10(yb)
                bad = ~conv[i, j] or not np.all(np.isfinite(rj[np.isfinite(yb)]))
                fine &= not bad
                res.append(np.where(np.isfinite(yb), PENALTY, np.nan) if bad else rj)
            out.append(np.concatenate(res)[self.keep])
            good.append(fine)
        return np.array(out), np.array(good)

    def fun(self, theta):
        return self.residuals_many([theta])[0][0]

    def jac(self, theta, h=0.02):
        """Forward differences; a column whose step fails takes the backward one, or is 0."""
        E = np.eye(len(theta))
        R, ok = self.residuals_many([theta] + [theta + h * e for e in E])
        J = (R[1:] - R[0]) / h
        failed = np.flatnonzero(~ok[1:])
        if failed.size:
            Rb, okb = self.residuals_many([theta - h * E[k] for k in failed])
            for m, k in enumerate(failed):
                J[k] = (R[0] - Rb[m]) / h if okb[m] else 0.0
        return J.T

    def per_decay_rms(self, r):
        sizes = [int(np.isfinite(d[2]).sum()) for d in self.data]
        parts = np.split(r, np.cumsum(sizes)[:-1])
        return [float(np.sqrt(np.mean(p**2))) for p in parts]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--starts", type=int, default=3, help="starts besides the authors' values")
    ap.add_argument("--spread", type=float, default=0.5, help="the starts' spread (decades)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-nfev", type=int, default=60, help="evaluations per start")
    ap.add_argument(
        "--optimizer",
        choices=("batched", "scipy"),
        default="batched",
        help="scipy's least_squares from all starts at once, their evaluations batched "
        "(lm_batch), or one start after another",
    )
    ap.add_argument(
        "--bottom-zone",
        type=float,
        metavar="NM",
        help="a zone of NM nm at each perovskite's bottom with its own deep traps",
    )
    ap.add_argument(
        "--start-mu",
        type=float,
        metavar="LOG10",
        help="the first start's perovskite mobility, log10 m²/Vs",
    )
    ap.add_argument(
        "--start-zone",
        type=float,
        default=21.0,
        metavar="LOG10",
        help="with --bottom-zone: the first start's zone trap density, log10 m⁻³",
    )
    ap.add_argument(
        "--detect",
        choices=("same", "opposite"),
        help="the PL as a detector at 806 nm on that side sees it (reabsorption)",
    )
    ap.add_argument("--sublayers", type=int, default=12, help="with --detect: perovskite layers")
    ap.add_argument("--from-best", action="store_true",
                    help="start from the best fit in validation/trpl_joint.json")  # fmt: skip
    a = ap.parse_args()
    out = "trpl_joint.json"
    first = START
    if a.from_best:
        fits = json.loads((REPO / "validation" / "trpl_joint.json").read_text())["fits"]
        first = np.array(min(fits, key=lambda f: f["rms"])["theta"])
    if a.bottom_zone:
        use_bottom_zone(a.bottom_zone * 1e-9)
        first = np.r_[first[:17], a.start_zone]
        out = f"trpl_joint_zone{a.bottom_zone:g}nm.json"
    if a.detect:
        use_detection(a.detect, a.sublayers)
        first = np.r_[first[:17], START[17:]]
        out = f"trpl_joint_detect_{a.detect}.json"
    if a.start_mu is not None:
        first = first.copy()
        first[1] = a.start_mu
    prob = Problem()
    rng = np.random.default_rng(a.seed)
    starts = [np.clip(first, LO, HI)] + [
        np.clip(first + rng.uniform(-a.spread, a.spread, len(first)), LO, HI)
        for _ in range(a.starts)]  # fmt: skip
    t_all = time.time()
    if a.optimizer == "batched":  # scipy from every start at once, its evaluations batched
        R0, _ = prob.residuals_many(starts)
        fits, batches = least_squares_many(prob.residuals_many, starts, LO, HI,
                                           max_nfev=a.max_nfev)  # fmt: skip
        R, _ = prob.residuals_many([np.array(f["theta"]) for f in fits])
        for f, r, r0 in zip(fits, R, R0, strict=True):
            f["per_decay"] = prob.per_decay_rms(r)
            f["rms_start"] = float(np.sqrt(np.mean(r0**2)))
        sizes = [b[0] for b in batches]
        print(f"{len(batches)} batches of {min(sizes)}–{max(sizes)} parameter sets, "
              f"{sum(b[1] for b in batches):.0f} s in the solver")  # fmt: skip
        for i, f in enumerate(fits):
            print(f"start {i}: rms {f['rms_start']:.3f} -> {f['rms']:.4f} decades; per decay "
                  + " ".join(f"{v:.3f}" for v in f["per_decay"]), flush=True)  # fmt: skip
    else:
        fits = sequential(prob, starts, a.max_nfev)
    print(f"{len(starts)} starts in {time.time() - t_all:.0f} s")
    best = min(fits, key=lambda f: f["rms"])
    print("the best fit:")
    for n, v, lo, hi in zip(NAMES, best["theta"], LO, HI, strict=True):
        print(f"   {n:32s} {v:9.3f}" + ("   (at its bound)" if min(v - lo, hi - v) < 1e-3 else ""))
    path = REPO / "validation" / out
    path.write_text(json.dumps({"names": NAMES, "fits": fits, "solves": prob.solves}))
    print("wrote", path.relative_to(REPO), f"({prob.solves} transients)")


def sequential(prob, starts, max_nfev):
    """scipy's least_squares from each start in turn."""
    fits = []
    for i, x0 in enumerate(starts):
        t0 = time.time()
        r0 = prob.fun(x0)
        res = least_squares(prob.fun, x0, jac=prob.jac, bounds=(LO, HI), max_nfev=max_nfev)
        rms = float(np.sqrt(np.mean(res.fun**2)))
        per = prob.per_decay_rms(res.fun)
        fits.append({"start": x0.tolist(), "theta": res.x.tolist(), "rms": rms,
                     "rms_start": float(np.sqrt(np.mean(r0**2))), "per_decay": per,
                     "seconds": time.time() - t0})  # fmt: skip
        print(f"start {i}: rms {fits[-1]['rms_start']:.3f} -> {rms:.4f} decades in "
              f"{time.time() - t0:.0f} s; per decay " + " ".join(f"{v:.3f}" for v in per),
              flush=True)  # fmt: skip
    return fits


if __name__ == "__main__":
    main()
