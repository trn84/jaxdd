"""How firmly the joint TRPL fit with outcoupling near the glass (trpl_joint.py `--detect same
--sublayers 24 --outcoupling film`) holds its values: refits from starts scattered about its best
fit (`scatter`), and profiles (`profiles`): the boost and the perovskite's mobility each held at
values of its own, out of the fitted vector, while every other parameter is refitted. The
mobility's profile steps down from the best fit, each value from the one before. Each part one
gathered least_squares (lm_batch) per step.

    python validation/trpl_robust.py scatter    # about 25 min on one GPU
    python validation/trpl_robust.py profiles   # about an hour; both into trpl_robust.json
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import trpl_joint as tj
from lm_batch import least_squares_many

REPO = tj.REPO
SOURCE = "trpl_joint_detect_same_out_film.json"
OUT = REPO / "validation" / "trpl_robust.json"
BOOST, REACH, MU = 19, 20, 1  # their places in the parameters
BOOSTS = (0.0, 3.0, 10.0, 100.0)  # the boost's profile
# the mobility's profile, log10 m²/Vs: up to 100 cm²/Vs from the best fit, and down to the
# authors' 0.41 in steps, each from the one before
MOBILITY_UP = (-2.0,)
MOBILITY_DOWN = (-3.0, np.log10(3e-4), -4.0, np.log10(0.41e-4))
CHUNK = 44  # parameter sets per solve (8 transients each)


def setup():
    tj.use_detection("same", 24)
    tj.use_outcoupling(["film"])
    fits = json.loads((REPO / "validation" / SOURCE).read_text())["fits"]
    best = np.array(min(fits, key=lambda f: f["rms"])["theta"])
    prob = tj.Problem()

    def residuals_many(thetas):
        parts = [prob.residuals_many(thetas[i : i + CHUNK]) for i in range(0, len(thetas), CHUNK)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])

    return best, prob, residuals_many


def finish(prob, residuals_many, fits, labels):
    R, _ = residuals_many([np.array(f["theta"]) for f in fits])
    for f, r, label in zip(fits, R, labels, strict=True):
        f["what"] = label
        f["per_decay"] = prob.per_decay_rms(r)
    return fits


def show(fits):
    print(f"{'':24s} {'rms':>6s} {'boost':>6s} {'reach nm':>8s} {'μ cm²/Vs':>8s}   per decay")
    for f in fits:
        th = f["theta"]
        per = " ".join(f"{v:.3f}" for v in f["per_decay"])
        print(f"{f['what']:24s} {f['rms']:6.4f} {th[BOOST]:6.2f} {10 ** th[REACH] * 1e9:8.1f} "
              f"{10 ** th[MU] * 1e4:8.2f}   {per}", flush=True)  # fmt: skip


def save(best, prob, part, fits):
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old.update({"names": tj.NAMES, "best": best.tolist(), part: fits})
    old[f"solves_{part}"] = prob.solves
    OUT.write_text(json.dumps(old))


def scatter():
    best, prob, residuals_many = setup()
    rng = np.random.default_rng(1)
    starts, labels, tries = [], [], 0
    while len(starts) < 3:  # starts that solve
        x = best + rng.uniform(-0.15, 0.15, best.size)  # decades, and the boost by a factor
        x[BOOST] = best[BOOST] * 10.0 ** rng.uniform(-0.15, 0.15)
        x = np.clip(x, tj.LO, tj.HI)
        tries += 1
        if residuals_many([x])[1][0]:
            starts.append(x)
            labels.append(f"scatter {tries}")
    t0 = time.time()
    fits, batches = least_squares_many(residuals_many, starts, tj.LO, tj.HI, max_nfev=60)
    print(f"{len(starts)} fits in {time.time() - t0:.0f} s, {len(batches)} batches")
    fits = finish(prob, residuals_many, fits, labels)
    show(fits)
    save(best, prob, "scatter", fits)


def held(theta, fixed):
    """The fitted vector without the parameters in `fixed`, its bounds, and the function that
    puts them back."""
    free = np.setdiff1d(np.arange(theta.size), fixed)
    base = theta.copy()

    def full(x):
        y = base.copy()
        y[free] = x
        return y

    return theta[free], tj.LO[free], tj.HI[free], full


def profiles():
    best, prob, residuals_many = setup()
    out = []

    def step(points):  # [(label, theta, the indices held)]
        starts, lo, hi, expand = [], [], [], []
        for _, theta, fixed in points:
            x, lo_k, hi_k, full = held(theta, fixed)
            starts.append(x)
            lo.append(lo_k)
            hi.append(hi_k)
            expand.append(full)
        t0 = time.time()
        fits, batches = least_squares_many(residuals_many, starts, lo, hi, max_nfev=60,
                                           expand=expand)  # fmt: skip
        print(f"{len(points)} fits in {time.time() - t0:.0f} s, {len(batches)} batches")
        fits = finish(prob, residuals_many, fits, [p[0] for p in points])
        show(fits)
        out.extend(fits)
        save(best, prob, "profiles", out)
        return fits

    def with_value(theta, i, v):
        y = theta.copy()
        y[i] = v
        return y

    first = [(f"boost {b:g}", with_value(best, BOOST, b), [BOOST, REACH] if b == 0 else [BOOST])
             for b in BOOSTS]  # fmt: skip
    first += [(f"mobility {10.0 ** m * 1e4:.3g} cm²/Vs", with_value(best, MU, m), [MU])
              for m in MOBILITY_UP + MOBILITY_DOWN[:1]]  # fmt: skip
    theta = np.array(step(first)[-1]["theta"])
    for m in MOBILITY_DOWN[1:]:
        theta = np.array(step([(f"mobility {10.0**m * 1e4:.3g} cm²/Vs",
                                with_value(theta, MU, m), [MU])])[0]["theta"])  # fmt: skip


if __name__ == "__main__":
    {"scatter": scatter, "profiles": profiles}[sys.argv[1] if len(sys.argv) > 1 else "scatter"]()
