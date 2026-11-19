"""The posterior of `trpl_joint.py`'s seventeen parameters, by `sample_measured.py`'s ensemble
sampler over the batch: each half-step solves half the walkers' eight decays at once.

The likelihood is Gaussian and independent per bin, its width `--sigma` times the best joint
fit's rms misfit (decades; by default 1, the reduced χ² one at the fit), the prior uniform in
`trpl_joint`'s coordinates within its bounds. The misfit is systematic (docs/validation.md), so the
widths are a lower bound. The walkers start from the linearised posterior around the best of
`validation/trpl_joint.json`; the first third of the steps is discarded.

    .venv/bin/python validation/trpl_sample.py [--walkers W] [--steps S] [--sigma F] [--seed N]

writes validation/trpl_sample.json (the summary) and benchmarks/samples/trpl_chain.npz (the chain,
also every `--every` steps); prints its progress as it goes.
"""

import argparse
import json
import time

import numpy as np
from sample_measured import autocorr_time, de_move, sample, start
from trpl_common import REPO
from trpl_joint import HI, LO, NAMES, Problem

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--walkers", type=int, default=48)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sigma", type=float, default=1.0, help="× the fit's rms misfit")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--every", type=int, default=20, help="save and report every N steps")
    a = ap.parse_args()
    fits = json.loads((REPO / "validation" / "trpl_joint.json").read_text())["fits"]
    best = min(fits, key=lambda f: f["rms"])
    theta = np.array(best["theta"])
    prob = Problem()
    sigma = a.sigma * best["rms"]
    rng = np.random.default_rng(a.seed)
    out_npz = REPO / "benchmarks" / "samples" / "trpl_chain.npz"
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    def log_prob(points):
        inside = np.all((points > LO) & (points < HI), axis=1)
        lp = np.full(len(points), -np.inf)
        if inside.any():
            R, _ = prob.residuals_many(list(points[inside]))
            lp[inside] = -0.5 * np.sum(R**2, axis=1) / sigma**2
        return lp

    J = prob.jac(theta)
    x0 = start(theta, J, sigma, LO, HI, a.walkers, rng)
    t0 = time.time()

    def progress(done, chain, lps, acc):
        np.savez(
            out_npz, chain=chain, log_prob=lps, names=np.array(NAMES), theta=theta, sigma=sigma
        )
        print(f"step {done}/{a.steps}: {time.time() - t0:.0f} s, acceptance {acc:.2f}, "
              f"best log-probability {lps.max():.1f}", flush=True)  # fmt: skip

    chain, lps, acc = sample(log_prob, x0, a.steps, rng, move=de_move, every=a.every,
                             progress=progress)  # fmt: skip
    np.savez(out_npz, chain=chain, log_prob=lps, names=np.array(NAMES), theta=theta, sigma=sigma)
    kept = chain[a.steps // 3 :]
    flat = kept.reshape(-1, kept.shape[2])
    tau = autocorr_time(kept)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    width = HI - LO
    at_bound = np.mean((flat - LO < 0.02 * width) | (HI - flat < 0.02 * width), axis=0)
    print(f"{a.steps} steps of {a.walkers} walkers in {time.time() - t0:.0f} s, acceptance "
          f"{acc:.2f}, σ {sigma:.3f} decades; the last {len(kept)} steps:")  # fmt: skip
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
    summary = {"walkers": a.walkers, "steps": a.steps, "sigma": sigma, "acceptance": acc,
               "rows": [plain(r) for r in rows], "correlations": strongest}  # fmt: skip
    (REPO / "validation" / "trpl_sample.json").write_text(json.dumps(summary, indent=1))
    print("wrote validation/trpl_sample.json and", out_npz.relative_to(REPO))


if __name__ == "__main__":
    main()
