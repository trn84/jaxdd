"""A hard workload: perovskite cells with frozen ions over a wide parameter range, light and dark.

    python validation/hard.py [N] [batch] [N_max]   # 16384 16384 on an A100, 1024 256 on a laptop

`bench.py`'s devices are easy: every one converges in a few Newton steps, so a batch pays little
for its slowest device. A parameter grid of real interest is not. These `N` devices are the
committed case with ions frozen after a pre-bias (`tests/data/frozen_1e17_x1_thin_light`),
drawn at random (a fixed seed) over the ranges such grids span: absorber 250–1000 nm, its
trap density over three decades (diffusion lengths from a fifth of the absorber to many times
it), 1e15–4e18 cm⁻³ of ions frozen at one of five accumulations (the pre-bias at
`(1 − x_ion)·V_bi`, `x_ion` 0–1), transport layers 1–300 nm doped 0–1e18 cm⁻³; half of them in
the dark. Strong frozen charge in a thin absorber is what used to stall Newton
(docs/solver.md).

Prints, for `solve_batch`'s two passes and for one pass, curves per second (compilation
excluded) and the unconverged voltages; and from the one pass, which returns every device's own
steps, the Newton steps per voltage the devices needed and a batch paid (docs/batching.md), and
the most any device took at one voltage. With `N_max` (m⁻³) the absorber's ions are bounded by
that many sites, both species (`N_max_anion`, `N_max_cation`; docs/model.md): the same, and
how far the curves are from Boltzmann's, unbounded, per device over its current's scale.
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import Layer, Setup, read_setup

CASE = Path(__file__).resolve().parents[1] / "tests" / "data" / "frozen_1e17_x1_thin_light"
V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
SEED = 20261111
ABSORBER_NM = (250.0, 350.0, 500.0, 700.0, 1000.0)
LAYER_NM = (1.0, 10.0, 30.0, 50.0, 100.0, 200.0, 300.0)
X_ION = (0.0, 0.25, 0.5, 0.75, 1.0)


def hard_setups(n: int, seed: int = SEED, n_max: float | None = None) -> list[Setup]:
    """`n` SIMsalabim inputs drawn over the ranges of the module docstring, every other one
    in the dark; with `n_max` the absorber's ions bounded by that many sites."""
    rng = np.random.default_rng(seed)
    base = read_setup(CASE / "simulation_setup.txt")
    v_bi = base.f("W_R") - base.f("W_L")
    setups = []
    for k in range(n):
        raw = dict(base.raw)
        etl, absorber, htl = (dict(ly.raw) for ly in base.layers)
        absorber["L"] = f"{rng.choice(ABSORBER_NM) * 1e-9:.6g}"
        absorber["N_t_bulk"] = f"{float(absorber['N_t_bulk']) * 10 ** rng.uniform(-1.5, 1.5):.6g}"
        ions = 10 ** rng.uniform(21, 24.6)  # m⁻³: 1e15–4e18 cm⁻³
        absorber["N_anion"] = absorber["N_cation"] = f"{ions:.6g}"
        raw["Vpre"] = f"{(1 - rng.choice(X_ION)) * v_bi:.6g}"
        for layer, dopant in ((etl, "N_D"), (htl, "N_A")):
            layer["L"] = f"{rng.choice(LAYER_NM) * 1e-9:.6g}"
            layer[dopant] = "0" if rng.random() < 0.2 else f"{10 ** rng.uniform(21, 24):.6g}"
        raw["G_frac"] = "0" if k % 2 else "1"
        if n_max is not None:
            absorber["N_max_anion"] = absorber["N_max_cation"] = f"{n_max:.6g}"
        layers = tuple(Layer(r) for r in (etl, absorber, htl))
        setups.append(Setup(raw=raw, layers=layers, directory=base.directory))
    return setups


def run(batches: list, first_max_iter: int | None) -> dict:
    """Every batch once to compile every shape it meets (the second pass's sizes vary), then
    timed."""
    for b in batches:
        np.asarray(solve_batch(b, V, first_max_iter=first_max_iter)[0])
    start = time.perf_counter()
    its, ok = [], []
    for b in batches:
        _, k, i = solve_batch(b, V, first_max_iter=first_max_iter)
        ok.append(np.asarray(k))
        its.append(np.asarray(i))
    seconds = time.perf_counter() - start
    return {"seconds": seconds, "its": its, "ok": np.concatenate(ok)}


def main(n: int = 1024, batch_size: int = 256, n_max: float | None = None) -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}")
    start = time.perf_counter()
    devices = [build_device(s) for s in hard_setups(n, n_max=n_max)]
    print(f"{n} devices built in {time.perf_counter() - start:.1f} s")
    batches = [prepare(devices[i : i + batch_size]) for i in range(0, n, batch_size)]
    for name, limit in (("two passes (default)", 6), ("one pass", None)):
        r = run(batches, limit)
        print(f"{name:21s} {n / r['seconds']:8.1f} curves/s ({r['seconds']:.1f} s); unconverged "
              f"voltages {int((~r['ok']).sum())} of {r['ok'].size}")  # fmt: skip
    needed = np.mean([i.mean(axis=0).mean() for i in r["its"]])
    paid = np.mean([i.max(axis=0).mean() for i in r["its"]])
    top = max(int(i.max()) for i in r["its"])
    hard = sum(int((i >= 10).any(axis=1).sum()) for i in r["its"])
    print(
        f"one pass, batch {batch_size}: Newton steps per voltage {needed:.2f} needed, "
        f"{paid:.2f} paid; at most {top} at one voltage, {hard} devices with 10 or more"
    )
    if n_max is not None:  # against Boltzmann's: the same devices, the ions unbounded
        free = [prepare([build_device(s) for s in hard_setups(n)[i : i + batch_size]])
                for i in range(0, n, batch_size)]  # fmt: skip

        def curves(bs):
            runs = [solve_batch(b, V) for b in bs]
            return (np.concatenate([np.asarray(r[k]) for r in runs]) for k in (0, 1))

        J, ok = curves(batches)
        J0, ok0 = curves(free)
        both = ok.all(axis=1) & ok0.all(axis=1)
        rel = np.max(np.abs(J - J0), axis=1) / np.maximum(np.max(np.abs(J0), axis=1), 1.0)
        q = np.quantile(rel[both], [0.5, 0.9, 0.99, 1.0])
        print(
            f"N_max = {n_max:.3g} m⁻³ against Boltzmann, the {int(both.sum())} devices converged "
            f"in both: the curves' largest difference over their scale, median {q[0]:.1e}, "
            f"90 % {q[1]:.1e}, 99 % {q[2]:.1e}, most {q[3]:.1e}; unconverged devices "
            f"{int((~ok.all(axis=1)).sum())} bounded, {int((~ok0.all(axis=1)).sum())} Boltzmann"
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(int(a) for a in args[:2]), *(float(a) for a in args[2:3]))
