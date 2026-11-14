"""The ion slots on a GPU: devices with two mobile ion regions per species against the same
devices with one, through the block sweep's kernel and through the scan.

    python validation/slots.py [N] [batch]      # e.g. 16384 4096 on an A100, 256 64 on a CPU

Each mobile ion region is a slot of the bordered Newton system (docs/solver.md): one region per
species gives K = 2, the layout every device had before slots; two per species K = 4, and the
block sweep takes 1 + 2K = 9 right-hand sides instead of 5. The `N` devices are
`tests/data/two_regions_light` (anions and cations mobile in the transport layers, the absorber
closed to them) with the ion densities of both transport layers drawn over two decades (a fixed
seed), every other one in the dark; their twins are the same with no ions in the right
transport layer (one region per species).

Prints, for both sets and both sweeps, curves per second (compilation excluded), the Newton
steps a batch paid per voltage (one pass), the time per paid Newton step and the unconverged
voltages; and how far the kernel's curves are from the scan's, as a share of each device's
largest |J|. On a CPU, or where no kernel runs, both sweeps are the scan.
"""

import os
import sys
import time
from pathlib import Path

import jax
import numpy as np

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import Layer, Setup, read_setup, simss_voltages
from jaxdd.sweep import sweep_kind

CASE = Path(__file__).resolve().parents[1] / "tests" / "data" / "two_regions_light"
SEED = 20261115


def slot_setups(n: int, two: bool, seed: int = SEED) -> list[Setup]:
    """`n` inputs from the case: two regions per species if `two`, else the right transport
    layer's ions removed."""
    rng = np.random.default_rng(seed)
    base = read_setup(CASE / "simulation_setup.txt")
    setups = []
    for k in range(n):
        raw = dict(base.raw)
        layers = [dict(ly.raw) for ly in base.layers]
        for j in (0, 2):
            for species in ("anion", "cation"):
                density = 10 ** rng.uniform(21, 23)  # m⁻³
                layers[j][f"N_{species}"] = f"{density if two or j == 0 else 0.0:.6g}"
        raw["G_frac"] = "0" if k % 2 else "1"
        setups.append(Setup(raw=raw, layers=tuple(Layer(r) for r in layers),
                            directory=base.directory))  # fmt: skip
    return setups


def run(batches: list, V) -> dict:
    """One pass (every device to convergence at each voltage): once to compile, then timed."""
    for b in batches:
        np.asarray(solve_batch(b, V, first_max_iter=None)[0])
    start = time.perf_counter()
    out = [solve_batch(b, V, first_max_iter=None) for b in batches]
    J, ok, its = (np.concatenate([np.asarray(o[i]) for o in out]) for i in range(3))
    seconds = time.perf_counter() - start
    paid = sum(int(np.asarray(o[2]).max(axis=0).sum()) for o in out)
    return {"seconds": seconds, "J": J, "ok": ok, "its": its, "paid": paid}


def main(n: int = 256, batch_size: int = 64) -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}")
    V = simss_voltages(read_setup(CASE / "simulation_setup.txt"))
    kernel = sweep_kind()
    results = {}
    for two in (False, True):
        devices = [build_device(s) for s in slot_setups(n, two)]
        batches = [prepare(devices[i : i + batch_size]) for i in range(0, n, batch_size)]
        ion = batches[0]["ion"]
        K = ion["a_slots"].shape[1] + ion["c_slots"].shape[1]
        for kind in dict.fromkeys(("scan", kernel)):
            os.environ["JAXDD_SWEEP"] = kind
            jax.clear_caches()
            r = results[two, kind] = run(batches, V)
            steps = r["paid"] / len(batches) / V.size
            print(f"K = {K}, {kind:6s}: {n / r['seconds']:8.1f} curves/s ({r['seconds']:.1f} s); "
                  f"{steps:.2f} Newton steps per voltage paid, "
                  f"{1e3 * r['seconds'] / r['paid']:.2f} ms per paid step (batch {batch_size}); "
                  f"unconverged voltages {int((~r['ok']).sum())} of {r['ok'].size}")  # fmt: skip
        if kernel != "scan":
            a, b = results[two, "scan"]["J"], results[two, kernel]["J"]
            scale = np.abs(a).max(axis=1, keepdims=True)
            print(
                f"K = {K}: kernel against scan, at most {np.max(np.abs(b - a) / scale):.1e} "
                "of a device's largest |J|"
            )
    for kind in dict.fromkeys(("scan", kernel)):
        one, two = results[False, kind], results[True, kind]
        ratio = (two["seconds"] / two["paid"]) / (one["seconds"] / one["paid"])
        print(
            f"{kind}: a paid Newton step with two regions per species takes {ratio:.2f}× one "
            "with one"
        )


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
