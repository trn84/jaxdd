"""The posterior of `trpl_joint.py`'s parameters, by `sample_measured.py`'s ensemble sampler over
the batch: each half-step proposes for half the walkers and solves the eight decays of those
proposals that lie within the bounds at once.

The likelihood is Gaussian and independent per bin, its width `--sigma` times the best joint
fit's rms misfit (decades; by default 1, the reduced χ² one at the fit), the prior uniform in
the sampled coordinates within the bounds. The misfit is systematic (docs/validation.md), so the
widths are a lower bound. The walkers start from the linearised posterior around the best fit;
the first third of the steps is discarded.

Two models. By default the seventeen parameters of the joint fit (from the best of
`validation/trpl_joint.json`). With `--outcoupling`, the fit as the detector sees it, the light
from near the glass leaving the film better (`trpl_joint.py --detect same --sublayers 24
--outcoupling film`, 21 parameters, from `trpl_joint_detect_same_out_film.json` or
`trpl_robust.json`, whichever fits best, else the best fit written below); its boost and reach
are sampled as log10(boost × reach) and log10 reach, along the valley the decays leave them
(docs/validation.md), the boost kept within its bounds. With `--outcoupling --neutral`, that fit
with the pulse's absorbed share by side and the SnO2 interface's states with their own density
and capture, neutral (saturable), at physical bounds but for SnO2's conduction band, from 3.9 eV
(`trpl_joint.py ... --outcoupling film --absorbed --neutral-interface --physical --ec-min 3.9`,
26 parameters, from the fit at 4.0 eV,
`trpl_joint_detect_same_out_film_absorbed_neutral_physical_ec4.json`, else the best fit written
below). With `--neutral --background`, that fit with the SnO2 stack's
decays without their background (`trpl_joint.py ... --background`), from
`..._ec3.9_bg.json`; where there is none, refitted first from the fit without the background (one
start, 200 evaluations, as `trpl_joint.py --background --max-nfev 200 --start-from` it) and
written there, so that the next start and the other chains take the same one (one outside the
bounds is refused).
`--fit-only` makes sure of the best fit and stops (benchmarks/apptainer.sh does so once before
its chains run side by side).

    .venv/bin/python validation/trpl_sample.py [--outcoupling [--neutral [--background]]]
        [--walkers W] [--steps S] [--sigma F] [--seed N] [--resume] [--fit-only]

writes the summary (validation/trpl_sample.json, with `--outcoupling`
validation/trpl_sample_out_s<seed>.json, with `--neutral` trpl_sample_neutral_s<seed>.json,
with `--background` trpl_sample_bg_ec3.9_s<seed>.json) and the chain with its bounds
(benchmarks/samples/trpl_chain.npz, trpl_out_chain_s<seed>.npz, trpl_neutral_chain_s<seed>.npz,
trpl_bg_ec3.9_chain_s<seed>.npz), the chain also every `--every` steps; `--resume` goes on from
the chain saved, if its bounds are these, up to `--steps` in all. Prints its progress as it goes.
"""

import argparse
import json
import time

import numpy as np
import trpl_joint as tj
from sample_measured import autocorr_time, de_move, sample, start
from trpl_common import REPO

#: The authors' medians where a parameter of ours has one (Kober-Czerny et al. 2025), in our
#: coordinates.
AUTHORS = {
    "log10 k_rad (m³/s)": np.log10(7.5e-17),
    "log10 μ FAPbI3 (m²/Vs)": np.log10(4.1e-5),
    "log10 photodoping (m⁻³)": np.log10(2.07e21),
    "log10 S glass/FAPbI3 (m/s)": np.log10(0.072),
    "log10 S FAPbI3/air (m/s)": np.log10(0.036),
    "log10 S SnO2/FAPbI3 (m/s)": np.log10(0.09),
    "log10 S FAPbI3/Spiro (m/s)": np.log10(1.0),
}
#: The best fit with outcoupling found (`trpl_robust.py scatter`, rms 0.1193 decades), where no
#: result file is at hand (a fresh clone).
OUTCOUPLING_BEST = [
    -15.9635, -2.5458, 21.0988, -13.1167, 14.0000, 20.6153, 0.4485, -4.8367, -5.0000, 2.9363,
    4.0012, 25.0799, -0.4414, 5.0820, 24.9613, -7.6404, 6.7800, 6.5491, 0.8955, 23.7511,
    -7.2314,
]  # fmt: skip
#: The same for `--neutral` (`trpl_joint.py`, the fit of the module docstring).
NEUTRAL_BEST = [
    -15.8740, -2.3468, 21.3363, -13.6032, 14.0497, 21.2660, 0.1924, -0.9734, -4.8911, 2.9993,
    4.0703, 25.4594, -0.3895, 5.0259, 25.4601, -8.1955, 3.2936, 6.5265, 0.9999, 53.8015, -7.3891,
    -0.0201, -0.2473, 14.4918, 4.6753, -17.8121,
]  # fmt: skip
NEUTRAL_RMS = 0.0912
NEUTRAL_FILE = "trpl_joint_detect_same_out_film_absorbed_neutral_physical_ec4.json"
NEUTRAL_BG_FILE = NEUTRAL_FILE.replace("_ec4.json", "_ec3.9_bg.json")
BOOST, REACH = 19, 20


def refit(prob, theta0):
    """The fit of `prob` from `theta0` (one start, 200 evaluations, as `trpl_joint.py` with
    `--max-nfev 200 --start-from`), written to validation/NEUTRAL_BG_FILE in `trpl_joint.py`'s
    form."""
    from lm_batch import least_squares_many

    t0 = time.time()
    print(f"no validation/{NEUTRAL_BG_FILE}: refitting with the background taken off", flush=True)
    fits, _ = least_squares_many(prob.residuals_many, [np.clip(theta0, tj.LO, tj.HI)], tj.LO,
                                 tj.HI, max_nfev=200)  # fmt: skip
    R, _ = prob.residuals_many([np.array(f["theta"]) for f in fits])
    for f, r in zip(fits, R, strict=True):
        f["per_decay"] = prob.per_decay_rms(r)
    path = REPO / "validation" / NEUTRAL_BG_FILE
    path.write_text(json.dumps({"names": tj.NAMES, "fits": fits, "solves": prob.solves}))
    f = fits[0]
    print(f"rms {f['rms']:.4f} decades in {time.time() - t0:.0f} s; per decay "
          + " ".join(f"{v:.3f}" for v in f["per_decay"]))  # fmt: skip
    print("theta", [round(v, 4) for v in f["theta"]])
    print("wrote", path.relative_to(REPO), flush=True)
    return np.array(f["theta"]), f["rms"]


def best_fit(outcoupling, neutral=False, background=False, prob=None):
    """The best fit's parameters and rms misfit; with `background` refitted by `prob` from the
    fit without it where no result file is at hand."""
    if background:
        path = REPO / "validation" / NEUTRAL_BG_FILE
        if path.exists():
            f = min(json.loads(path.read_text())["fits"], key=lambda f: f["rms"])
            th = np.array(f["theta"])
            if np.any((th < tj.LO) | (th > tj.HI)):  # another run's bounds, and so its sigma
                raise SystemExit(f"{path.name} lies outside these bounds: move it away")
            return th, f["rms"]
        return refit(prob, best_fit(outcoupling, neutral)[0])
    found = []
    if neutral:
        names = [NEUTRAL_FILE]
    elif outcoupling:
        names = ["trpl_joint_detect_same_out_film.json", "trpl_robust.json"]
    else:
        names = ["trpl_joint.json"]
    for name in names:
        path = REPO / "validation" / name
        if path.exists():
            d = json.loads(path.read_text())
            found += d.get("fits", []) + d.get("scatter", [])
    if found:
        f = min(found, key=lambda f: f["rms"])
        return np.array(f["theta"]), f["rms"]
    if neutral:
        return np.array(NEUTRAL_BEST), NEUTRAL_RMS
    if outcoupling:
        return np.array(OUTCOUPLING_BEST), 0.1193
    raise SystemExit("no validation/trpl_joint.json: run validation/trpl_joint.py first")


def coordinates(outcoupling):
    """The sampled coordinates: names, bounds, and the maps to `trpl_joint`'s parameters and
    back. With outcoupling the boost and reach as log10(boost × reach) and log10 reach."""
    if not outcoupling:
        same = lambda u: np.array(u, dtype=float)  # noqa: E731
        return tj.NAMES, tj.LO, tj.HI, same, same
    names = (*tj.NAMES[:BOOST], "log10 boost × reach (m)", *tj.NAMES[REACH:])
    lo = np.r_[tj.LO[:BOOST], -9.0, tj.LO[REACH:]]
    hi = np.r_[tj.HI[:BOOST], np.log10(tj.HI[BOOST]) + tj.HI[REACH], tj.HI[REACH:]]

    def to_theta(u):
        th = np.array(u, dtype=float)
        th[..., BOOST] = 10.0 ** (th[..., BOOST] - th[..., REACH])
        return th

    def from_theta(th):
        u = np.array(th, dtype=float)
        u[..., BOOST] = np.log10(np.maximum(u[..., BOOST], 1e-6)) + u[..., REACH]
        return u

    return names, lo, hi, to_theta, from_theta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--outcoupling", action="store_true",
                    help="the fit with outcoupling near the glass (21 parameters)")  # fmt: skip
    ap.add_argument("--neutral", action="store_true",
                    help="with --outcoupling: the absorbed share by side and saturable SnO2 "
                         "interface states, at physical bounds (26 parameters)")  # fmt: skip
    ap.add_argument("--background", action="store_true",
                    help="with --neutral: the SnO2 decays without their background")  # fmt: skip
    ap.add_argument("--fit-only", action="store_true",
                    help="make sure of the best fit (refit it where needed) and stop")  # fmt: skip
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=300, help="steps in all, with --resume too")
    ap.add_argument("--sigma", type=float, default=1.0, help="× the fit's rms misfit")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=20, help="save and report every N steps")
    ap.add_argument("--resume", action="store_true", help="go on from the chain saved")
    a = ap.parse_args()
    if a.neutral and not a.outcoupling:
        raise SystemExit("--neutral goes with --outcoupling")
    if a.background and not a.neutral:
        raise SystemExit("--background goes with --outcoupling --neutral")
    out_npz = REPO / "benchmarks" / "samples" / "trpl_chain.npz"
    out_json = REPO / "validation" / "trpl_sample.json"
    if a.outcoupling:
        tj.use_detection("same", 24)
        tj.use_outcoupling(["film"])
        kind = "out"
    if a.neutral:  # in trpl_joint.py's order of the parameters
        tj.use_absorbed()
        tj.use_charged_interface(0)
        tj.use_physical_bounds(3.9)
        kind = "bg_ec3.9" if a.background else "neutral"
    if a.outcoupling:
        out_npz = REPO / "benchmarks" / "samples" / f"trpl_{kind}_chain_s{a.seed}.npz"
        out_json = REPO / "validation" / f"trpl_sample_{kind}_s{a.seed}.json"
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    NAMES, LO, HI, to_theta, from_theta = coordinates(a.outcoupling)
    prob = tj.Problem(background=a.background)
    if a.fit_only:
        th_best, rms = best_fit(a.outcoupling, a.neutral, a.background, prob)
        print(f"the best fit: rms {rms:.4f} decades")
        return

    def allowed(points):  # the sampled box, and the boost within its own bounds
        th = to_theta(points)
        return np.all((points > LO) & (points < HI) & (th >= tj.LO) & (th <= tj.HI), axis=1)

    def log_prob(points):
        inside = allowed(points)
        lp = np.full(len(points), -np.inf)
        n = int(inside.sum())
        if n:
            th = to_theta(points[inside])
            # padded to a multiple of 8 sets: each batch size compiles the solver once
            th = np.concatenate([th, np.repeat(th[-1:], -n % 8, axis=0)])
            R, _ = prob.residuals_many(list(th))
            lp[inside] = -0.5 * np.sum(R[:n] ** 2, axis=1) / sigma**2
        return lp

    def save(chain, lps):
        np.savez(out_npz, chain=chain, log_prob=lps, names=np.array(NAMES), theta=theta,
                 sigma=sigma, lo=LO, hi=HI)  # fmt: skip

    if a.resume and out_npz.exists():
        old = np.load(out_npz)
        if "lo" not in old or not np.array_equal([old["lo"], old["hi"]], [LO, HI]):
            raise SystemExit(f"{out_npz.name} has other bounds or none saved: move it away")
        before, lps_before = old["chain"], old["log_prob"]
        theta, sigma = old["theta"], float(old["sigma"])
        x0 = before[-1]
        rng = np.random.default_rng([a.seed, len(before)])
        print(f"going on from step {len(before)} of {out_npz.relative_to(REPO)}")
    else:
        th_best, rms = best_fit(a.outcoupling, a.neutral, a.background, prob)
        theta = np.clip(from_theta(th_best), LO + 1e-9, HI - 1e-9)
        sigma = a.sigma * rms
        rng = np.random.default_rng(a.seed)
        h = 0.02  # the Jacobian in the sampled coordinates, forward differences
        R, _ = prob.residuals_many([to_theta(theta + h * e) for e in np.eye(theta.size)]
                                   + [to_theta(theta)])  # fmt: skip
        J = ((R[:-1] - R[-1]) / h).T
        x0 = start(theta, J, sigma, LO, HI, a.walkers, rng, allowed)
        before, lps_before = np.empty((0, a.walkers, theta.size)), np.empty((0, a.walkers))
    steps = a.steps - len(before)
    t0 = time.time()

    def progress(done, chain, lps, acc):
        save(np.concatenate([before, chain]), np.concatenate([lps_before, lps]))
        print(f"step {len(before) + done}/{a.steps}: {time.time() - t0:.0f} s, acceptance "
              f"{acc:.2f}, best log-probability {lps.max():.1f}", flush=True)  # fmt: skip

    if steps > 0:
        chain, lps, acc = sample(log_prob, x0, steps, rng, move=de_move, every=a.every,
                                 progress=progress)  # fmt: skip
        chain, lps = np.concatenate([before, chain]), np.concatenate([lps_before, lps])
    else:  # done before: the summary only
        chain, lps, acc = before, lps_before, float("nan")
    save(chain, lps)
    kept = chain[len(chain) // 3 :]
    flat = kept.reshape(-1, kept.shape[2])
    tau = autocorr_time(kept)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    width = HI - LO
    at_bound = np.mean((flat - LO < 0.02 * width) | (HI - flat < 0.02 * width), axis=0)
    print(f"{len(chain)} steps of {chain.shape[1]} walkers ({steps} in {time.time() - t0:.0f} s), "
          f"acceptance {acc:.2f}, σ {sigma:.3f} decades; the last {len(kept)} steps:")  # fmt: skip
    heads = ("fit", "median", "16 %", "84 %")
    print(f"   {'':32s}" + "".join(f" {h:>8s}" for h in heads) + f" {'τ':>6s} {'at bound':>9s}"
          f" {'authors':>8s}")  # fmt: skip
    rows = []
    for i, n in enumerate(NAMES):
        auth = AUTHORS.get(n, np.nan)
        rows.append({"name": n, "fit": theta[i], "median": q50[i], "q16": q16[i], "q84": q84[i],
                     "tau": tau[i], "at_bound": at_bound[i], "authors": auth})  # fmt: skip
        print(f"   {n:32s} {theta[i]:8.3f} {q50[i]:8.3f} {q16[i]:8.3f} {q84[i]:8.3f} {tau[i]:6.1f}"
              f" {at_bound[i]:9.2f} {auth:8.3f}")  # fmt: skip
    derived = {}
    if a.outcoupling:  # the boost itself, and the mobility in cm²/Vs
        boost = 10.0 ** (flat[:, BOOST] - flat[:, REACH])
        mu = 10.0 ** flat[:, 1] * 1e4
        for n, v in (("boost", boost), ("μ FAPbI3 (cm²/Vs)", mu)):
            derived[n] = [float(q) for q in np.percentile(v, [16, 50, 84])]
            print(f"   {n:32s} median {derived[n][1]:.3g}, 16–84 % {derived[n][0]:.3g}–"
                  f"{derived[n][2]:.3g}")  # fmt: skip
    corr = np.corrcoef(flat.T)
    pairs = sorted(
        ((abs(corr[i, j]), i, j) for i in range(len(NAMES)) for j in range(i)), reverse=True
    )
    print("strongest correlations:")
    for _, i, j in pairs[:6]:
        print(f"   {corr[i, j]:+.2f}  {NAMES[i]}  ×  {NAMES[j]}")

    def plain(r):
        return {k: (float(v) if isinstance(v, np.floating | float) else v) for k, v in r.items()}

    strongest = [[NAMES[i], NAMES[j], float(corr[i, j])] for _, i, j in pairs[:10]]
    summary = {"walkers": chain.shape[1], "steps": len(chain), "sigma": sigma,
               "acceptance": acc, "rows": [plain(r) for r in rows], "derived": derived,
               "correlations": strongest}  # fmt: skip
    out_json.write_text(json.dumps(summary, indent=1))
    print("wrote", out_json.relative_to(REPO), "and", out_npz.relative_to(REPO))


if __name__ == "__main__":
    main()
