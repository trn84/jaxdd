"""Throughput of transients: batches of ZimT runs, on whatever JAX runs.

    python validation/transient_bench.py [N] [batch]     # e.g. 16384 4096 on an A100, 64 64 here

Three workloads from the committed ZimT inputs under `tests/data`, each device with its trap
density and its transport layers' doping drawn at random (a fixed seed), the ions' density too
where there are ions:

- `traps`: the light switched on and off at short circuit (`zimt_traps_photo_light`, 241 steps
  over six decades of time), electrons, holes and the absorber's traps: 3×3 blocks;
- `ions`: a sweep to 0.9 V and back in 2 × 50 ms with mobile ions (`zimt_ions_fast_light`, 241
  steps): 5×5 blocks;
- `oc`: the photovoltage after a pulse at the open circuit (`zimt_tpv_light`, 151 steps): 3×3
  blocks and the internal voltage's border.

Printed per workload, after compilation: transients and time steps (device × step) per
second; the Newton steps per time step, the devices' mean and what the batch pays (each step
in lockstep costs its slowest device's); and the time of one batched Newton step per device,
which compares the 5×5 step with the 3×3 one.
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np

from jaxdd.batch import prepare
from jaxdd.device import build_device
from jaxdd.simss_input import Layer, Setup, read_setup, read_tvg
from jaxdd.transient import solve_transient

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
WORKLOADS = {"traps": "zimt_traps_photo_light", "ions": "zimt_ions_fast_light",
             "oc": "zimt_tpv_light"}  # fmt: skip


def variant_setups(case: str, n: int, seed: int = 20261115) -> list[Setup]:
    """`n` inputs of `case`: the absorber's trap density, its ion density (where it has ions)
    and the transport layers' doping drawn at random, as `validation/bench.py` draws them."""
    rng = np.random.default_rng(seed)
    base = read_setup(DATA / case / "simulation_setup.txt")
    setups = []
    for _ in range(n):
        layers = [dict(ly.raw) for ly in base.layers]
        if float(layers[1]["N_t_bulk"]) > 0:
            layers[1]["N_t_bulk"] = f"{float(layers[1]['N_t_bulk']) * 10 ** rng.uniform(-1, 1):.6g}"
        if float(layers[1]["N_anion"]) > 0:
            ions = 10 ** rng.uniform(22, 23.5)
            layers[1]["N_anion"] = layers[1]["N_cation"] = f"{ions:.6g}"
        layers[0]["N_D"] = f"{10 ** rng.uniform(18, 22):.6g}"
        layers[2]["N_A"] = f"{10 ** rng.uniform(18, 22):.6g}"
        setups.append(Setup(raw=dict(base.raw), layers=tuple(Layer(r) for r in layers),
                            directory=base.directory))  # fmt: skip
    return setups


def run(name: str, n: int, batch_size: int) -> None:
    case = WORKLOADS[name]
    setups = variant_setups(case, n)
    tvg = read_tvg(setups[0])
    steps = len(tvg.t)
    batches = [prepare([build_device(s) for s in setups[i : i + batch_size]])
               for i in range(0, n, batch_size)]  # fmt: skip
    start = time.perf_counter()
    solve_transient(batches[0], tvg, bernoulli_name="simss")
    compiled = time.perf_counter() - start
    start = time.perf_counter()
    its, ok = [], 0
    for b in batches:
        r = solve_transient(b, tvg, bernoulli_name="simss")  # returns NumPy: synchronised
        its.append(r.iterations)
        ok += int(r.converged.sum())
    seconds = time.perf_counter() - start
    its = np.concatenate(its)
    mean = its.mean()
    paid = np.mean([b.max(axis=0).mean() for b in np.split(its, len(batches))])
    newton = seconds / (len(batches) * steps * paid) / batch_size
    print(f"{name:>5} ({case}, {steps} steps): {n} transients in batches of {batch_size}, "
          f"compiled in {compiled:.0f} s")  # fmt: skip
    print(f"       {n / seconds:9.1f} transients/s, {n * steps / seconds:11.0f} time steps/s; "
          f"converged {ok}/{n * steps}")  # fmt: skip
    print(f"       Newton steps per time step: {mean:.2f} per device, {paid:.2f} paid by the "
          f"batch; one batched Newton step {newton * 1e6:.2f} µs per device")  # fmt: skip


def main(n: int = 64, batch_size: int = 64, names: str = "traps,ions,oc") -> None:
    jax.config.update("jax_enable_x64", True)
    print(f"JAX {jax.__version__} on {jax.devices()}")
    for name in names.split(","):
        run(name, n, batch_size)


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:3]), *sys.argv[3:4])
