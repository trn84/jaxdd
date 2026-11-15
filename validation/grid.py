"""M4's baseline: how many grid points SIMsalabim's grid needs on hard devices (PLAN.md 2.1).

    python validation/grid.py [N] [batch] [NPs]   # e.g. 4096 4096 on an A100, 16 8 100,200 on a CPU

`N` devices of `hard.py` (frozen ions up to 4e18 cm⁻³ in absorbers of 250–1000 nm, trap
densities over three decades, transport layers 1–300 nm), each in the light and in the dark,
solved on SIMsalabim's grid (`Make_Grid`, `NP` points, graded by `grad`) at each `NP` of `NPs`
(default 200, 400, 800, 1600, 3200); the largest is the reference. Per `NP`, against it, the
validation criteria (docs/validation.md): `η = (J_dark − J_light)/J_gen` within 1e-3 at every
voltage up to 1.0 V, `J_SC` within 0.1 %. Prints, per `NP`, the share of devices that meet
them, the median, 99th percentile and largest error of each, the unconverged voltages, and
curves per second. `batch` is the batch at 400 points; at `NP` points it is `batch·400/NP`
(at least 16), as the memory a device takes grows with `NP`.
"""

import sys
import time

import jax
import numpy as np
from hard import V, hard_setups

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import Setup

ETA_TOL, JSC_TOL = 1e-3, 1e-3


def pairs(n: int) -> list[Setup]:
    """Each of `n` hard devices in the light and then in the dark."""
    out = []
    for s in hard_setups(n):
        for g in ("1", "0"):
            out.append(Setup(raw=dict(s.raw, G_frac=g), layers=s.layers, directory=s.directory))
    return out


def solve(setups: list[Setup], NP: int, batch: int):
    """`J`, `converged` (2n × voltages), the generation current per device, and seconds."""
    setups = [Setup(raw=dict(s.raw, NP=str(NP)), layers=s.layers, directory=s.directory)
              for s in setups]  # fmt: skip
    devices = [build_device(s) for s in setups]
    J, ok, scale, seconds = [], [], [], 0.0
    for i in range(0, len(devices), batch):
        b = prepare(devices[i : i + batch])
        if i == 0:  # compile
            np.asarray(solve_batch(b, V)[0])
        start = time.perf_counter()
        j, k, _ = solve_batch(b, V)
        J.append(np.asarray(j))
        seconds += time.perf_counter() - start
        ok.append(np.asarray(k))
        scale.append(np.asarray(b["J_scale"]))
    return np.concatenate(J), np.concatenate(ok), np.concatenate(scale), seconds


def main(n: int = 16, batch: int = 8, nps: str = "200,400,800,1600,3200") -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}")
    grid_sizes = sorted(int(v) for v in nps.split(","))
    setups = pairs(n)
    up_to = V <= 1.0 + 1e-9
    sc = int(np.argmin(np.abs(V)))
    results = {}
    for NP in grid_sizes:
        size = max(16, batch * 400 // NP)
        J, ok, scale, seconds = solve(setups, NP, size)
        results[NP] = (J, ok, scale)
        print(f"NP {NP:5d}: {2 * n / seconds:8.1f} curves/s (batch {size}), unconverged "
              f"voltages {int((~ok).sum())} of {ok.size}", flush=True)  # fmt: skip
    J_ref, ok_ref, scale = results[grid_sizes[-1]]
    eta_ref = (J_ref[1::2] - J_ref[0::2]) / scale[0::2, None]
    print(f"\nagainst NP {grid_sizes[-1]}, {n} devices (criteria: η within {ETA_TOL:g} up to "
          f"1.0 V, J_SC within {100 * JSC_TOL:g} %):")  # fmt: skip
    for NP in grid_sizes[:-1]:
        J, ok, _ = results[NP]
        eta = (J[1::2] - J[0::2]) / scale[0::2, None]
        valid = (ok[0::2] & ok[1::2] & ok_ref[0::2] & ok_ref[1::2])[:, up_to]
        d_eta = np.where(valid, np.abs(eta - eta_ref)[:, up_to], 0.0).max(axis=1)
        jsc, jsc_ref = J[0::2, sc], J_ref[0::2, sc]
        d_jsc = np.abs(jsc - jsc_ref) / np.maximum(np.abs(jsc_ref), 1e-300)
        meets = (d_eta <= ETA_TOL) & (d_jsc <= JSC_TOL)

        def q(x):
            return f"median {np.median(x):.1e}, 99 % {np.quantile(x, 0.99):.1e}, max {x.max():.1e}"

        print(f"NP {NP:5d}: {meets.mean():6.1%} meet both; η {q(d_eta)}; J_SC {q(d_jsc)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(int(a) for a in args[:2]), *args[2:3])
