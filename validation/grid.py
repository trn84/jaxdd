"""M4: how many grid points the hard devices need, on SIMsalabim's grid and on the graded one
(PLAN.md 2.1).

    python validation/grid.py [N] [batch] [NPs] [grids]   # 4096 4096 on an A100; 16 8 100,200

`N` devices of `hard.py` (frozen ions up to 4e18 cm⁻³ in absorbers of 250–1000 nm, trap
densities over three decades, transport layers 1–300 nm), each in the light and in the dark,
solved at each `NP` of `NPs` (default 200, 400, 800, 1600, 3200) on each grid of `grids`
(default `simss,graded`): SIMsalabim's (`Make_Grid`, graded by `grad`) and jaxdd's graded one
(`build_device(grid="graded")`, geometric towards every layer's ends). The reference is the
last grid at the largest `NP`. Per grid and `NP`, against it, the validation criteria
(docs/validation.md): `η = (J_dark − J_light)/J_gen` within 1e-3 at every voltage up to 1.0 V,
`J_SC` within 0.1 %. Prints, per grid and `NP`, the share of devices that meet them, the
median, 99th percentile and largest error of each, the unconverged voltages, and curves per
second. `batch` is the batch at 400 points; at `NP` points it is `batch·400/NP` (at least 16),
as the memory a device takes grows with `NP`.

Each grid size's curves are saved as soon as they are solved, to `grid_<N>.npz` (beside this
script, not committed); a run that finds the file takes the sizes it holds from it, so one that
was stopped goes on where it stopped, and sizes can be added later.
"""

import sys
import time
from pathlib import Path

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


def solve(setups: list[Setup], NP: int, batch: int, grid: str = "simss"):
    """`J`, `converged` (2n × voltages), the generation current per device, and seconds."""
    setups = [Setup(raw=dict(s.raw, NP=str(NP)), layers=s.layers, directory=s.directory)
              for s in setups]  # fmt: skip
    devices = [build_device(s, grid=grid) for s in setups]
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


def main(n: int = 16, batch: int = 8, nps: str = "200,400,800,1600,3200",
         grids: str = "simss,graded") -> None:  # fmt: skip
    print(f"JAX {jax.__version__} on {jax.devices()}")
    grid_sizes = sorted(int(v) for v in nps.split(","))
    grid_kinds = grids.split(",")
    setups = pairs(n)
    up_to = V <= 1.0 + 1e-9
    sc = int(np.argmin(np.abs(V)))
    saved = Path(__file__).with_name(f"grid_{n}.npz")
    store = dict(np.load(saved)) if saved.exists() else {}
    results = {}
    for grid in grid_kinds:
        tag = "" if grid == "simss" else f"{grid}_"  # SIMsalabim's keys as they were
        for NP in grid_sizes:
            if f"J_{tag}{NP}" in store:
                results[grid, NP] = tuple(store[f"{k}_{tag}{NP}"] for k in ("J", "ok", "scale"))
                print(f"{grid:>6} NP {NP:5d}: from {saved.name}", flush=True)
                continue
            size = max(16, batch * 400 // NP)
            J, ok, scale, seconds = solve(setups, NP, size, grid)
            results[grid, NP] = (J, ok, scale)
            store.update({f"J_{tag}{NP}": J, f"ok_{tag}{NP}": ok, f"scale_{tag}{NP}": scale})
            np.savez(saved, **store)
            print(f"{grid:>6} NP {NP:5d}: {2 * n / seconds:8.1f} curves/s (batch {size}), "
                  f"unconverged voltages {int((~ok).sum())} of {ok.size}", flush=True)  # fmt: skip
    reference = (grid_kinds[-1], grid_sizes[-1])
    J_ref, ok_ref, scale = results[reference]
    eta_ref = (J_ref[1::2] - J_ref[0::2]) / scale[0::2, None]
    print(f"\nagainst {reference[0]} at NP {reference[1]}, {n} devices (criteria: η within "
          f"{ETA_TOL:g} up to 1.0 V, J_SC within {100 * JSC_TOL:g} %):")  # fmt: skip
    for key in results:
        if key == reference:
            continue
        J, ok, _ = results[key]
        eta = (J[1::2] - J[0::2]) / scale[0::2, None]
        valid = (ok[0::2] & ok[1::2] & ok_ref[0::2] & ok_ref[1::2])[:, up_to]
        d_eta = np.where(valid, np.abs(eta - eta_ref)[:, up_to], 0.0).max(axis=1)
        jsc, jsc_ref = J[0::2, sc], J_ref[0::2, sc]
        d_jsc = np.abs(jsc - jsc_ref) / np.maximum(np.abs(jsc_ref), 1e-300)
        meets = (d_eta <= ETA_TOL) & (d_jsc <= JSC_TOL)

        def q(x):
            return f"median {np.median(x):.1e}, 99 % {np.quantile(x, 0.99):.1e}, max {x.max():.1e}"

        print(f"{key[0]:>6} NP {key[1]:5d}: {meets.mean():6.1%} meet both; η {q(d_eta)}; "
              f"J_SC {q(d_jsc)}")  # fmt: skip


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(int(a) for a in args[:2]), *args[2:4])
