"""A hard workload: perovskite cells with frozen ions over a wide parameter range, light and dark.

    python validation/hard.py [N] [batch]       # e.g. 16384 16384 on an A100, 1024 256 on a laptop

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
the most any device took at one voltage.
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


def hard_setups(n: int, seed: int = SEED) -> list[Setup]:
    """`n` SIMsalabim inputs drawn over the ranges of the module docstring, every other one
    in the dark."""
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


def main(n: int = 1024, batch_size: int = 256) -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}")
    start = time.perf_counter()
    devices = [build_device(s) for s in hard_setups(n)]
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


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
