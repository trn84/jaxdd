"""The uncertainty of a fit to measured curves (M7), by sampling: a posterior of the parameters
`validation/fit_measured.py` fits, by an affine-invariant ensemble sampler over the batch.

    .venv/bin/python validation/sample_measured.py [SAMPLE] [--device mirrored|anode]
        [--move de|stretch] [--walkers W] [--steps S] [--sigma F] [--seed N] [--out FILE.npz]

The walkers are split into two halves, and each half moves relative to the other, so one step is
two calls of the model, each solving half the walkers' five intensities as one batch
(`fit_measured.problem`'s `residuals_many`). The move is differential evolution (ter Braak 2006:
along the difference of two walkers of the other half) or Goodman and Weare's stretch (2010;
emcee's default). Both are invariant under affine maps of the parameters, so the Jacobian's
spread of singular values over seven to nine decades, which ruins a random walk, costs them
nothing where the posterior is close to Gaussian. On a 14-dimensional Gaussian whose axes span
nine decades, with 64 walkers, both recover every variance within 7 %; the autocorrelation time
is about 50 steps with differential evolution and 125 with the stretch, hence the default.

The model: `fit_measured`'s device and its fourteen parameters, in the same coordinates
(logarithms where `FREE` says so). The prior is uniform in those coordinates within the paper's
bounds (`FREE`); the likelihood Gaussian and independent per point with a standard deviation
`sigma` times the 1-sun short-circuit current, by default the least-squares fit's own rms misfit
(the reduced χ² one at the fit). A voltage not converged costs a whole scale, as in the fit, so
such a point is never accepted. The misfit is systematic, not noise (the residuals are smooth in
the voltage), so this likelihood is a lower bound on the widths; they scale with `sigma` where the
posterior is close to Gaussian.

The walkers start around the fit (it is run first, from the middle of the bounds), drawn from the
linearised posterior `σ²(JᵀJ + P)⁻¹`, `P` a weak Gaussian prior of the bounds' width so that the
directions the curves do not settle start spread over the box, not beyond it. The first third of
the steps is discarded. Printed: the acceptance fraction, each parameter's integrated
autocorrelation time (Sokal's window, as emcee), and per parameter the fit, the posterior's median
and its 16–84 % interval, the linearised 1σ interval `σ²(JᵀJ)⁻¹` for comparison, and the share of
the samples within 2 % of the box's width from a bound; then the strongest correlations.
"""

import argparse
import time

import jax.numpy as jnp
import numpy as np
from fit_measured import DEVICES, FREE, GAP_eV, fit, problem


def stretch_move(x, other, rng, a=2.0):
    """Goodman and Weare's stretch: along the line through a walker of the other half. Returns
    the proposals and their log Hastings factors."""
    z = ((a - 1) * rng.random(len(x)) + 1) ** 2 / a  # density ∝ 1/√z on [1/a, a]
    partner = other[rng.integers(len(other), size=len(x))]
    return partner + z[:, None] * (x - partner), (x.shape[1] - 1) * np.log(z)


def de_move(x, other, rng, jump=0.1):
    """Differential evolution (ter Braak, Stat. Comput. 16, 239 (2006)): along the difference of
    two walkers of the other half, scaled by 2.38/√(2d), by one in a tenth of the moves (a jump
    between modes). Symmetric, so no Hastings factor."""
    n, d = other.shape
    j = rng.integers(n, size=len(x))
    k = (j + rng.integers(1, n, size=len(x))) % n  # never j
    gamma = 2.38 / np.sqrt(2 * d) * (1 + 0.1 * rng.normal(size=len(x)))
    gamma = np.where(rng.random(len(x)) < jump, 1.0, gamma)
    return x + gamma[:, None] * (other[j] - other[k]), np.zeros(len(x))


def sample(log_prob, x0, steps, rng, move=de_move, every=0, progress=None):
    """An ensemble sampler, the two halves of the walkers in turn, each moved by `move` relative
    to the other: `log_prob(points) -> (points,)` is called twice a step, on half the walkers.
    Returns the chain (steps, walkers, dim), its log-probabilities and the acceptance fraction.
    With `every`, `progress(steps done, chain so far, log-probabilities so far, acceptance)` is
    called every `every` steps."""
    W, d = x0.shape
    halves = (np.arange(W // 2), np.arange(W // 2, W))
    x, lp = x0.copy(), log_prob(x0)
    chain, lps, accepted = np.empty((steps, W, d)), np.empty((steps, W)), 0
    for s in range(steps):
        for k in (0, 1):
            mine, other = halves[k], halves[1 - k]
            y, log_factor = move(x[mine], x[other], rng)
            lp_y = log_prob(y)
            take = np.log(rng.random(len(mine))) < log_factor + lp_y - lp[mine]
            x[mine[take]], lp[mine[take]] = y[take], lp_y[take]
            accepted += take.sum()
        chain[s], lps[s] = x, lp
        if every and progress and (s + 1) % every == 0 and s + 1 < steps:
            progress(s + 1, chain[: s + 1], lps[: s + 1], accepted / ((s + 1) * W))
    return chain, lps, accepted / (steps * W)


def autocorr_time(chain, c=5.0):
    """The integrated autocorrelation time of each parameter, from the walkers' mean
    autocorrelation function, with Sokal's automatic window (as emcee's `integrated_time`)."""
    steps = chain.shape[0]
    n = 1 << (2 * steps - 1).bit_length()
    f = chain - chain.mean(axis=0)  # (steps, walkers, dim)
    acf = np.fft.irfft(np.abs(np.fft.rfft(f, n=n, axis=0)) ** 2, axis=0)[:steps]
    acf = acf.mean(axis=1)
    acf /= acf[0]
    taus = 2.0 * np.cumsum(acf, axis=0) - 1.0
    out = np.empty(chain.shape[2])
    for i in range(chain.shape[2]):
        m = np.arange(steps) < c * taus[:, i]
        window = np.argmin(m) if not m.all() else steps - 1
        out[i] = taus[window, i]
    return out


def start(theta, jac, sigma, lo, hi, walkers, rng):
    """Walkers from the linearised posterior around the fit, a weak prior of the bounds' width
    added, redrawn until inside the bounds."""
    width = hi - lo
    precision = jac.T @ jac / sigma**2 + np.diag(1.0 / (0.5 * width) ** 2)
    cov = np.linalg.inv(precision)
    cov = 0.5 * (cov + cov.T)
    x = np.empty((walkers, len(theta)))
    filled = 0
    while filled < walkers:
        y = rng.multivariate_normal(theta, cov, size=4 * walkers)
        y = y[np.all((y > lo) & (y < hi), axis=1)][: walkers - filled]
        x[filled : filled + len(y)] = y
        filled += len(y)
    return x


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("sample", nargs="?", default="02Cl", choices=tuple(GAP_eV))
    args.add_argument("--device", choices=tuple(DEVICES), default="mirrored")
    args.add_argument("--move", choices=("de", "stretch"), default="de")
    args.add_argument("--walkers", type=int, default=64)
    args.add_argument("--steps", type=int, default=1500)
    args.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="the likelihood's σ over J_sc (default: the fit's rms misfit)",
    )
    args.add_argument("--seed", type=int, default=0)
    args.add_argument(
        "--out", default=None, help="write the chain to this .npz (also every --every steps)"
    )
    args.add_argument("--every", type=int, default=100, help="a progress line every N steps")
    a = args.parse_args()
    if a.walkers % 2 or a.walkers < 2 * len(FREE):
        raise SystemExit(f"--walkers must be even and at least {2 * len(FREE)}")

    t0 = time.perf_counter()
    r = fit(a.sample, "cold", a.device)
    out = r["out"]
    rms = float(np.sqrt(np.mean(out.fun**2)))
    sigma = rms if a.sigma is None else a.sigma
    t1 = time.perf_counter()
    print(f"{a.sample} ({a.device}): fit in {t1 - t0:.0f} s, rms misfit {rms:.2e} of J_sc = "
          f"{r['scale']:.1f} A/m²; σ = {sigma:.2e} of J_sc", flush=True)  # fmt: skip

    pr = problem(a.sample, a.device)
    lo, hi, many = pr["lo"], pr["hi"], pr["residuals_many"]
    calls = [0]

    def log_prob(points):
        calls[0] += 1
        inside = np.all((points > lo) & (points < hi), axis=1)
        res = np.asarray(many(jnp.asarray(np.clip(points, lo, hi))))
        lp = -0.5 * np.sum(res**2, axis=1) / sigma**2
        return np.where(inside, lp, -np.inf)

    rng = np.random.default_rng(a.seed)
    x0 = start(out.x, out.jac, sigma, lo, hi, a.walkers, rng)
    t2 = time.perf_counter()

    def save(chain, lps):
        if a.out:
            np.savez_compressed(a.out, chain=chain, log_prob=lps, fit=out.x, sigma=sigma,
                                names=[f[0] for f in FREE], log=[f[1] for f in FREE], lo=lo,
                                hi=hi, sample=a.sample, device=a.device)  # fmt: skip

    def progress(done, chain, lps, acc):
        print(f"  step {done}/{a.steps}, {time.perf_counter() - t2:.0f} s: acceptance "
              f"{acc:.2f}, best log-probability {lps.max():.1f}, median now "
              f"{np.median(lps[-1]):.1f}", flush=True)  # fmt: skip
        save(chain, lps)

    move = de_move if a.move == "de" else stretch_move
    chain, lps, acc = sample(log_prob, x0, a.steps, rng, move, a.every, progress)
    t3 = time.perf_counter()
    burn = a.steps // 3
    kept = chain[burn:]
    tau = autocorr_time(kept)
    flat = kept.reshape(-1, kept.shape[2])
    print(f"  {a.walkers} walkers x {a.steps} steps in {t3 - t2:.0f} s ({calls[0]} calls of "
          f"{a.walkers // 2} x 5 curves, {(t3 - t2) / calls[0]:.2f} s each); acceptance "
          f"{acc:.2f}; the first {burn} steps discarded")  # fmt: skip
    print(f"  autocorrelation time {tau.min():.0f}-{tau.max():.0f} steps: about "
          f"{kept.shape[0] * a.walkers / tau.max():.0f} independent samples"
          + ("" if kept.shape[0] > 50 * tau.max() else "; fewer than 50 times the longest "
             "time kept, so the estimate of it is itself uncertain"))  # fmt: skip
    print(f"  best log-probability in the chain {lps.max():.1f}, at the fit "
          f"{-0.5 * np.sum(out.fun**2) / sigma**2:.1f}")  # fmt: skip

    try:
        lin = np.sqrt(np.diag(np.linalg.inv(out.jac.T @ out.jac))) * sigma
    except np.linalg.LinAlgError:
        lin = np.full(len(FREE), np.inf)
    q16, q50, q84 = np.percentile(flat, [16, 50, 84], axis=0)
    edge = 0.02 * (hi - lo)
    near = np.mean((flat < lo + edge) | (flat > hi - edge), axis=0)

    def show(i, t):
        return f"{np.exp(t):10.3g}" if FREE[i][1] else f"{t:10.3f}"

    def factor(i, w):
        if not np.isfinite(w) or w > 50:
            return f"{'>e^50':>10}"
        text = "x" + format(np.exp(w), ".3g") if FREE[i][1] else "±" + format(w, ".3f")
        return f"{text:>10}"

    print(f"  {'param':>8} {'fit':>10} {'median':>10} {'16 %':>10} {'84 %':>10} "
          f"{'(84-16)/2':>10} {'linear 1σ':>10} {'at bound':>9} {'tau':>5}")  # fmt: skip
    for i, (name, _, _, _) in enumerate(FREE):
        half = 0.5 * (q84[i] - q16[i])
        print(f"  {name:>8} {show(i, out.x[i])} {show(i, q50[i])} {show(i, q16[i])} "
              f"{show(i, q84[i])} {factor(i, half)} {factor(i, lin[i])} {near[i]:9.2f} "
              f"{tau[i]:5.0f}")  # fmt: skip
    corr = np.corrcoef(flat.T)
    pairs = sorted(((abs(corr[i, j]), i, j) for i in range(len(FREE)) for j in range(i)),
                   reverse=True)[:6]  # fmt: skip
    print("  strongest correlations: " + ", ".join(
        f"{FREE[i][0]}-{FREE[j][0]} {corr[i, j]:+.2f}" for _, i, j in pairs))  # fmt: skip
    save(chain, lps)


if __name__ == "__main__":
    main()
