"""A fit by gradients (M7): a device's parameters recovered from its SimSS curve.

    .venv/bin/python validation/fit.py CASE_DIR [--start S] [--fix NAME ...] [--noise SIGMA]

The case's input gives the true parameters and SimSS's `JV.dat` the curve. The fit starts away
from them (each fitted parameter moved by `S` times its entry in `FITTED`), builds the device in
JAX from its parameters (`batch.device_tree`), and minimises the curve's misfit by
`scipy.optimize.least_squares`, its Jacobian from `jax.jacfwd` (forward mode through the
converged solution, by implicit differentiation: one tangent per parameter). Densities and
mobilities are fitted in their logarithms, the work function as it is.

Printed: the parameters fitted against the true ones; the misfit, also at the true parameters
(jaxdd and SimSS agree to about 1e-6 of the scale, which is the floor); the singular values of
the Jacobian at the end, in the fitted coordinates: what the curve determines and what it does
not. With `--noise`, Gaussian noise of SIGMA times the curve's scale on SimSS's curve, and each
parameter's standard error from the Jacobian (σ²·(JᵀJ)⁻¹, linearised).
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from jaxdd.batch import device_tree, solve_batch
from jaxdd.device import device_parameters
from jaxdd.simss_input import read_setup

ABSORBER = 1
#: (name, layer or None, fitted in its logarithm, the start's offset: a factor for a logarithm,
#: else a shift)
FITTED = [
    ("N_t_bulk", ABSORBER, True, 3.0),
    ("k_direct", ABSORBER, True, 1 / 3),
    ("mu_n", ABSORBER, True, 2.0),
    ("mu_p", ABSORBER, True, 0.5),
    ("W_L", None, False, 0.05),
]
#: Fitted too where the absorber holds mobile anions.
ION = ("N_anion", ABSORBER, True, 2.0)


@dataclass
class Fit:
    names: list[str]
    log: list[bool]
    true: np.ndarray  # fitted coordinates
    start: np.ndarray
    out: object  # scipy's OptimizeResult
    misfit_true: float
    misfit_start: float
    seconds_compile: float
    seconds_fit: float
    curves: int
    jacobians: int

    def errors(self) -> np.ndarray:
        """Relative for a logarithm, absolute otherwise."""
        diff = self.out.x - self.true
        return np.where(self.log, np.expm1(diff), diff)


def _rms(r) -> float:
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def fit(case: Path, start: float = 1.0, fix=(), noise: float = 0.0, seed: int = 0) -> Fit:
    """Fit `FITTED` (and `ION` where there are mobile anions), but `fix`, to the case's SimSS
    curve plus `noise`, from `start` times the offsets."""
    structure, true = device_parameters(read_setup(case / "simulation_setup.txt"))
    fitted = FITTED + ([ION] if structure.anion_mobile.any() else [])
    fitted = [f for f in fitted if f[0] not in fix]
    ref = np.genfromtxt(case / "JV.dat", names=True)
    V, J_ref = jnp.asarray(ref["Vint"]), np.asarray(ref["Jext"])
    scale = np.max(np.abs(J_ref))
    J_data = J_ref + noise * scale * np.random.default_rng(seed).normal(size=J_ref.shape)

    def value(p, name, layer):
        return p[name] if layer is None else p[name][layer]

    theta_true = np.array(
        [np.log(value(true, n, k)) if log else value(true, n, k) for n, k, log, _ in fitted]
    )
    offsets = np.array([np.log(s) if log else s for _, _, log, s in fitted])
    theta_start = theta_true + start * offsets

    def params(theta):
        p = {k: jnp.asarray(v) for k, v in true.items()}
        for t, (name, layer, log, _) in zip(theta, fitted, strict=True):
            v = jnp.exp(t) if log else t
            p[name] = v if layer is None else p[name].at[layer].set(v)
        return p

    @jax.jit
    def residuals(theta):
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, params(theta)))
        J, ok, _ = solve_batch(batch, V, bernoulli_name="simss")
        # A voltage not converged costs a whole scale: left out (zero), it would reward the
        # optimiser for leaving the region where the solver converges. Its gradient is zero, so
        # the trust region rejects such a step rather than following it.
        return jnp.where(ok[0], (J[0] - J_data) / scale, 1.0)

    jacobian = jax.jit(jax.jacfwd(residuals))
    calls = {"r": 0, "J": 0}

    def fun(theta):
        calls["r"] += 1
        return np.asarray(residuals(jnp.asarray(theta)))

    def jac(theta):
        calls["J"] += 1
        return np.asarray(jacobian(jnp.asarray(theta)))

    t0 = time.perf_counter()
    misfit_start = _rms(fun(theta_start))
    jac(theta_start)
    t1 = time.perf_counter()
    # Bounds: a factor of 1e3 either way for a logarithm, ±0.5 eV for a work function.
    width = np.array([np.log(1e3) if log else 0.5 for _, _, log, _ in fitted])
    out = least_squares(
        fun, theta_start, jac=jac, method="trf", x_scale="jac", xtol=1e-12, ftol=1e-14,
        gtol=1e-14, bounds=(theta_start - width, theta_start + width),
    )  # fmt: skip
    t2 = time.perf_counter()
    return Fit(
        names=[n if k is None else f"{n}[{k}]" for n, k, _, _ in fitted],
        log=[log for _, _, log, _ in fitted],
        true=theta_true, start=theta_start, out=out,
        misfit_true=_rms(fun(theta_true)), misfit_start=misfit_start,
        seconds_compile=t1 - t0, seconds_fit=t2 - t1, curves=calls["r"], jacobians=calls["J"],
    )  # fmt: skip


def report(f: Fit, noise: float) -> None:
    out = f.out
    print(f"compile {f.seconds_compile:.1f} s, fit {f.seconds_fit:.1f} s ({f.curves} curves, "
          f"{f.jacobians} Jacobians): {out.message}")  # fmt: skip
    active = [f.names[i] for i in np.flatnonzero(out.active_mask)]
    print(f"misfit (rms, of the curve's scale): start {f.misfit_start:.2e}, end "
          f"{_rms(out.fun):.2e}, at the true parameters {f.misfit_true:.2e}; gradient "
          f"{out.optimality:.1e}; bounds active: {active}")  # fmt: skip
    J = out.jac
    sv, Vt = np.linalg.svd(J, full_matrices=False)[1:]
    cov = None
    if noise > 0:
        sigma2 = float(np.sum(out.fun**2)) / max(J.shape[0] - J.shape[1], 1)
        cov = sigma2 * np.linalg.pinv(J.T @ J)
    header = f"{'parameter':>16} {'true':>12} {'start':>12} {'fitted':>12} {'error':>10}"
    print(header + ("  std. error" if cov is not None else ""))
    for i, (name, log, err) in enumerate(zip(f.names, f.log, f.errors(), strict=True)):
        show = np.exp if log else float
        line = (f"{name:>16} {show(f.true[i]):12.5g} {show(f.start[i]):12.5g} "
                f"{show(out.x[i]):12.5g} {err:+10.2e}")  # fmt: skip
        if cov is not None:
            se = np.sqrt(cov[i, i])
            line += f"  ×{np.exp(se):.4g}" if log else f"  ±{se:.4g}"
        print(line)
    print("singular values of the Jacobian (fitted coordinates):",
          " ".join(f"{s:.2e}" for s in sv))  # fmt: skip
    weakest = [f"{w:+.2f}·{n}" for w, n in zip(Vt[-1], f.names, strict=True) if abs(w) > 0.1]
    print("the least determined direction:", " ".join(weakest))


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("case", type=Path)
    args.add_argument("--start", type=float, default=1.0, help="a multiple of the offsets")
    args.add_argument("--fix", nargs="*", default=[], help="parameters held at their true value")
    args.add_argument("--noise", type=float, default=0.0)
    args.add_argument("--seed", type=int, default=0)
    a = args.parse_args()
    print(f"{a.case.name}: start {a.start:g}, fixed {a.fix}, noise {a.noise:g} of the scale")
    report(fit(a.case, a.start, a.fix, a.noise, a.seed), a.noise)


if __name__ == "__main__":
    main()
