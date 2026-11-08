"""Throughput of the batched solver: N variants of the reference devices, on whatever JAX runs.

    python validation/bench.py [N] [batch]      # e.g. 65536 16384 on an A100, 64 64 on a laptop

Needs only this repository: the variants are the committed SimSS inputs under `tests/data`
with the trap density, the ion density and the transport layers' doping drawn at random (a
fixed seed), so the same command gives the same devices anywhere. Prints devices per second
after compilation, and SimSS's rate on the 64-core server for comparison. A device here is one
run, one J-V curve; SimSS's 7.8 devices/s on the server (PLAN.md §1) are FauxPV devices of two
runs each (light and dark), so the comparison is against 15.6 runs/s.
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import Layer, Setup, read_setup

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
#: SimSS on the 64-core server, runs (curves) per second: 7.8 FauxPV devices of two runs each.
SIMSS_SERVER_RUNS_PER_S = 2 * 7.8
#: SimSS on one core, one device (61 voltages, light): 2.0 s (PLAN.md §1a, M0).
SIMSS_SECONDS_PER_DEVICE_ONE_CORE = 2.0


CASES = ("thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light")


def variants(n: int, seed: int = 20261107) -> list:
    """`n` devices: `variant_setups` built."""
    return [build_device(setup) for setup in variant_setups(n, seed)]


def variant_setups(n: int, seed: int = 20261107) -> list[Setup]:
    """`n` SIMsalabim inputs: the committed cases in turn, the trap density, the ion density
    and the transport layers' doping drawn at random. Each keeps its case's directory, from
    which its generation profile is read (and `benchmarks/run.py` copies its files for SimSS)."""
    rng = np.random.default_rng(seed)
    bases = [read_setup(DATA / c / "simulation_setup.txt") for c in CASES]
    setups = []
    for k in range(n):
        base = bases[k % len(bases)]
        layers = [dict(ly.raw) for ly in base.layers]
        layers[1]["N_t_bulk"] = f"{float(layers[1]['N_t_bulk']) * 10 ** rng.uniform(-1.5, 1.0):.6g}"
        if float(layers[1]["N_anion"]) > 0:
            ions = 10 ** rng.uniform(21, 24)
            layers[1]["N_anion"] = layers[1]["N_cation"] = f"{ions:.6g}"
        layers[0]["N_D"] = f"{10 ** rng.uniform(18, 23):.6g}"
        layers[2]["N_A"] = f"{10 ** rng.uniform(18, 23):.6g}"
        setups.append(Setup(raw=dict(base.raw), layers=tuple(Layer(r) for r in layers),
                            directory=base.directory))  # fmt: skip
    return setups


def main(n: int = 64, batch_size: int = 64) -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}")
    devices = variants(n)
    batches = [prepare(devices[i : i + batch_size]) for i in range(0, n, batch_size)]
    start = time.perf_counter()
    J, ok, _ = solve_batch(batches[0], V)
    J.block_until_ready()
    print(f"compilation and first batch of {batch_size}: {time.perf_counter() - start:.1f} s")
    start = time.perf_counter()
    converged = 0
    for b in batches:
        J, ok, _ = solve_batch(b, V)
        J.block_until_ready()
        converged += int(np.asarray(ok).sum())
    seconds = time.perf_counter() - start
    rate = n / seconds
    print(f"{n} devices x {len(V)} voltages in {seconds:.1f} s: {rate:.1f} devices/s, "
          f"converged {converged}/{n * len(V)} points")  # fmt: skip
    if jax.devices()[0].platform == "cpu":
        # On a CPU compare per core: SimSS runs one device per core. XLA's own threads do not
        # help this solver (tiny 3x3 operations in a sequential sweep); for throughput run one
        # process per core with XLA_FLAGS="--xla_cpu_multi_thread_eigen=false
        # intra_op_parallelism_threads=1".
        print(f"  per core, if this ran on one: {SIMSS_SECONDS_PER_DEVICE_ONE_CORE * rate:.1f}x "
              "SimSS on one core")  # fmt: skip
    else:
        print(f"  {rate / SIMSS_SERVER_RUNS_PER_S:.1f}x SimSS on the 64-core server, per run")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]))
