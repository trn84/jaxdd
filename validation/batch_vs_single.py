"""M3: the batched solver against the one-device solver and against SimSS, on CPU.

    .venv/bin/python validation/batch_vs_single.py CASE_DIR [CASE_DIR ...]

Every case directory holds SimSS's input and its JV.dat; all run in one batch at FauxPV's 61
voltages.
"""

import sys
import time
from pathlib import Path

import numpy as np

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import check_supported, read_setup
from jaxdd.solver import solve_jv

V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
BERN = "simss"
LINEAR = "auto"
if len(sys.argv) > 1 and sys.argv[1].startswith("--linear="):
    LINEAR = sys.argv.pop(1).split("=")[1]


def main(cases: list[Path]) -> None:
    devices = []
    for case in cases:
        setup = read_setup(case / "simulation_setup.txt")
        check_supported(setup)
        devices.append(build_device(setup))
    batch = prepare(devices)
    start = time.perf_counter()
    J, ok, its = solve_batch(batch, V, bernoulli_name=BERN, linear=LINEAR)
    J.block_until_ready()
    first = time.perf_counter() - start
    start = time.perf_counter()
    J, ok, its = solve_batch(batch, V, bernoulli_name=BERN, linear=LINEAR)
    J.block_until_ready()
    again = time.perf_counter() - start
    J, ok, its = np.asarray(J), np.asarray(ok), np.asarray(its)
    print(f"{len(cases)} devices: {first:.1f} s with compilation, {again:.1f} s again")
    for k, case in enumerate(cases):
        single = solve_jv(devices[k], V, bernoulli_name=BERN)
        ref = np.genfromtxt(case / "JV.dat", names=True)
        sim = np.interp(V, ref["Vint"], ref["Jext"], left=np.nan, right=np.nan)
        have = np.isin(np.round(V, 6), np.round(ref["Vint"], 6))
        scale = max(np.nanmax(np.abs(ref["Jext"])), 1.0)
        vs_single = np.max(np.abs(J[k] - single.J)) / scale
        vs_simss = np.max(np.abs(J[k] - sim)[have]) / scale
        name = "/".join(case.parts[-2:])
        counts = f"conv {ok[k].sum()}/{len(V)} it {its[k].mean():5.1f} (max {its[k].max()})"
        print(f"  {name:32s} {counts}  vs single {vs_single:.1e}  vs SimSS {vs_simss:.1e}")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
