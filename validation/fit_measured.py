"""A fit to measured curves (M7): the five light intensities of one sample of Heester et al.
(2025) at once, by gradients.

    .venv/bin/python validation/fit_measured.py [SAMPLE ...] [--start cold|paper]

SAMPLE is `01Cl`, `02Cl` or `06Cl` (all three by default; `validation/data/heester2025/`). The
device is `tests/data/pin_iface_light`: the paper's stack (Table S2), mirrored to put the cathode
left, with interface traps at both interfaces, and the sample's band gap. The paper's model as
far as it is stated; what it does not state is assumed: a uniform generation in the absorber
(its scale fitted; the paper's is transfer-matrix optics), the interface traps donors at 4.7 eV
with capture coefficients of 2e-14 m³/s (SIMsalabim's defaults), the transport layers' outer bands
(HTL `E_c` 2.5 eV, ETL `E_v` 6.2 eV). No series resistance, as in the paper. The intensities are
the measured short-circuit currents' ratios to the 1-sun curve's (0.126 where the files say 0.10,
0.357 for 0.38).

Fitted: the paper's free parameters (Table I, its bounds) and the generation's scale, by
`scipy.optimize.least_squares` with the Jacobian from `jax.jacfwd` through the converged
solutions. Printed per sample: the misfit (rms of the residual over the 1-sun short-circuit
current, and the mean |ΔJ| per curve), the parameters against the paper's fit, which of them
end at a bound, and the singular values of the Jacobian.
"""

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from jaxdd.batch import device_tree, solve_batch
from jaxdd.device import device_parameters
from jaxdd.simss_input import read_setup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "validation" / "data" / "heester2025"
DEVICE = ROOT / "tests" / "data" / "pin_iface_light"
FILES = ("1sun", "083sun", "053sun", "038sun", "01sun")
GAP_eV = {"01Cl": 1.68, "02Cl": 1.66, "06Cl": 1.65}  # the paper's Fig. S2
ETL, ABSORBER, HTL = 0, 1, 2  # the mirrored stack's layers

#: The paper's best fits (Table S2 of its supplementary material), for comparison and as a start.
PAPER = {
    "01Cl": dict(mu_n=4.68e-4, mu_p=6.08e-5, N_t=3.08e20, k=3.11e-16, R_sh=1.02, mu_htl=2.35e-6,
                 mu_etl=8.52e-6, Ev_htl=5.5, Ec_etl=3.96, N_cat=1.25e21, N_an=1.5e21,
                 Nti_etl=2.12e14, Nti_htl=4.29e13),
    "02Cl": dict(mu_n=5.02e-4, mu_p=6.32e-5, N_t=4.21e20, k=3.56e-16, R_sh=1.32, mu_htl=1.89e-6,
                 mu_etl=3.6e-6, Ev_htl=5.5, Ec_etl=4.00, N_cat=1.2e21, N_an=1.5e21,
                 Nti_etl=4.63e14, Nti_htl=5.01e13),
    "06Cl": dict(mu_n=1.30e-4, mu_p=2.11e-5, N_t=1.67e21, k=5.63e-16, R_sh=0.26, mu_htl=6.36e-6,
                 mu_etl=2.57e-7, Ev_htl=5.54, Ec_etl=4.27, N_cat=1.2e21, N_an=1.4e21,
                 Nti_etl=5.15e14, Nti_htl=6.32e12),
}  # fmt: skip
#: (name, in its logarithm, lower, upper): the paper's bounds (its Table I; the transport layers'
#: mobilities to 1e-5, the paper's 1e-6 being where its own fits end), the generation's scale.
FREE = [
    ("mu_n", True, 1e-6, 1e-3), ("mu_p", True, 1e-6, 1e-3), ("N_t", True, 1e19, 1e23),
    ("Nti_htl", True, 1e11, 1e15), ("Nti_etl", True, 1e11, 1e15), ("k", True, 1e-18, 1e-15),
    ("R_sh", True, 1e-2, 1e3), ("mu_htl", True, 1e-8, 1e-5), ("mu_etl", True, 1e-8, 1e-5),
    ("Ev_htl", False, 5.4, 5.6), ("Ec_etl", False, 3.9, 4.3), ("N_cat", True, 1e20, 1e23),
    ("N_an", True, 1e20, 1e23), ("G", True, 0.7, 1.3),
]  # fmt: skip


def fit(sample: str, start: str = "cold") -> dict:
    curves = [np.loadtxt(DATA / f"{sample}_{f}_fwd_JV.dat", skiprows=1) for f in FILES]
    V = jnp.asarray(curves[0][:, 0])
    J_data = np.stack([c[:, 1] for c in curves])
    jsc = np.array([abs(np.interp(0.0, c[:, 0], c[:, 1])) for c in curves])
    suns, scale = jsc / jsc[0], jsc[0]
    structure, base = device_parameters(read_setup(DEVICE / "simulation_setup.txt"))
    base = {k: jnp.asarray(v) for k, v in base.items()}
    base["E_v"] = base["E_v"].at[ABSORBER].set(base["E_c"][ABSORBER] + GAP_eV[sample])
    base["gen_G"] = base["gen_G"] * jsc[0] / 187.0  # the device's profile was made for 187 A/m²

    def unpack(theta):
        pairs = zip(theta, FREE, strict=True)
        return {n: (jnp.exp(t) if log else t) for t, (n, log, _, _) in pairs}

    def params(v, g):
        p = dict(base, G_frac=g, gen_G=base["gen_G"] * v["G"])
        p["mu_n"] = jnp.stack([v["mu_etl"], v["mu_n"], v["mu_htl"]])
        p["mu_p"] = jnp.stack([v["mu_etl"], v["mu_p"], v["mu_htl"]])
        p["N_t_bulk"] = p["N_t_bulk"].at[ABSORBER].set(v["N_t"])
        p["k_direct"] = p["k_direct"].at[ABSORBER].set(v["k"])
        p["E_v"] = p["E_v"].at[HTL].set(v["Ev_htl"])
        p["E_c"] = p["E_c"].at[ETL].set(v["Ec_etl"])
        p["N_anion"] = p["N_anion"].at[ABSORBER].set(v["N_an"])
        p["N_cation"] = p["N_cation"].at[ABSORBER].set(v["N_cat"])
        p["N_t_int"] = jnp.stack([v["Nti_etl"], v["Nti_htl"], p["N_t_int"][HTL]])
        p["R_shunt"] = v["R_sh"]
        return p

    @jax.jit
    def residuals(theta):
        v = unpack(theta)
        trees = [device_tree(structure, params(v, g)) for g in suns]
        J, ok, _ = solve_batch(jax.tree_util.tree_map(lambda *x: jnp.stack(x), *trees), V)
        # A voltage not converged costs a whole scale (validation/fit.py).
        return jnp.where(ok, (J - J_data) / scale, 1.0).ravel()

    jacobian = jax.jit(jax.jacfwd(residuals))
    lo = np.array([np.log(a) if log else a for _, log, a, _ in FREE])
    hi = np.array([np.log(b) if log else b for _, log, _, b in FREE])
    if start == "paper":
        v0 = dict(PAPER[sample], G=1.0)
        theta0 = np.clip([np.log(v0[n]) if log else v0[n] for n, log, _, _ in FREE], lo, hi)
    else:
        theta0 = 0.5 * (lo + hi)
    calls = {"r": 0, "J": 0}

    def fun(theta):
        calls["r"] += 1
        return np.asarray(residuals(jnp.asarray(theta)))

    def jac(theta):
        calls["J"] += 1
        return np.asarray(jacobian(jnp.asarray(theta)))

    t0 = time.perf_counter()
    fun(theta0), jac(theta0)
    t1 = time.perf_counter()
    out = least_squares(fun, theta0, jac=jac, bounds=(lo, hi), method="trf", x_scale="jac",
                        xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=400)  # fmt: skip
    t2 = time.perf_counter()
    return dict(sample=sample, start=start, out=out, suns=suns, scale=scale, theta0=theta0,
                fitted={k: float(x) for k, x in unpack(out.x).items()},
                seconds=(t1 - t0, t2 - t1), calls=dict(calls))  # fmt: skip


def report(r: dict) -> None:
    out, scale = r["out"], r["scale"]
    per_curve = np.mean(np.abs(out.fun.reshape(5, -1)), axis=1) * scale
    print(f"{r['sample']} from {r['start']}: compile {r['seconds'][0]:.1f} s, fit "
          f"{r['seconds'][1]:.1f} s ({r['calls']['r']} curves, {r['calls']['J']} Jacobians, "
          f"each for 5 intensities): {out.message}; intensities "
          f"{np.round(r['suns'], 3).tolist()}")  # fmt: skip
    print(f"  rms misfit {np.sqrt(np.mean(out.fun**2)):.2e} of J_sc = {scale:.1f} A/m²; mean "
          f"|ΔJ| per curve " + " ".join(f"{x:.2f}" for x in per_curve) + " A/m²")  # fmt: skip
    print(f"  {'param':>8} {'paper':>10} {'fitted':>10}")
    for i, (name, _, _, _) in enumerate(FREE):
        paper = PAPER[r["sample"]].get(name, 1.0)
        bound = "  at a bound" if out.active_mask[i] else ""
        print(f"  {name:>8} {paper:10.3g} {r['fitted'][name]:10.3g}{bound}")
    sv = np.linalg.svd(out.jac, compute_uv=False)
    print("  singular values:", " ".join(f"{s:.1e}" for s in sv))


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("samples", nargs="*", default=list(GAP_eV))
    args.add_argument("--start", choices=("cold", "paper"), default="cold")
    a = args.parse_args()
    for sample in a.samples:
        report(fit(sample, a.start))


if __name__ == "__main__":
    main()
