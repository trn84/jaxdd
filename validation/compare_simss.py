"""Compare jaxdd's J-V curve with SimSS's on SimSS case directories (inputs + JV.dat).

.venv/bin/python validation/compare_simss.py ../jaxdd-ref/cases/thin_l1/light [...]
"""

import sys
import time
from pathlib import Path

import numpy as np

from jaxdd.device import build_device
from jaxdd.simss_input import check_supported, read_setup
from jaxdd.solver import solve_jv

BERN = "exact"


def compare(case: Path) -> dict:
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    dev = build_device(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    start = time.perf_counter()
    res = solve_jv(dev, ref["Vint"], bernoulli_name=BERN)
    seconds = time.perf_counter() - start
    scale = max(np.max(np.abs(ref["Jext"])), 1.0)
    gap = np.abs(res.J - ref["Jext"])
    worst = int(np.argmax(gap))
    return {
        "case": str(case),
        "points": len(ref),
        "converged": int(res.converged.sum()),
        "max_abs_A_m2": float(gap.max()),
        "max_rel_to_scale": float(gap.max() / scale),
        "at_V": float(ref["Vint"][worst]),
        "sim": float(ref["Jext"][worst]),
        "jaxdd": float(res.J[worst]),
        "J_at_0": (
            float(np.interp(0, ref["Vint"], ref["Jext"])),
            float(np.interp(0, ref["Vint"], res.J)),
        ),
        "mean_iterations": float(res.iterations.mean()),
        "seconds": seconds,
    }


if __name__ == "__main__":
    if sys.argv[1].startswith("--bernoulli="):
        BERN = sys.argv.pop(1).split("=")[1]
    for arg in sys.argv[1:]:
        r = compare(Path(arg))
        print(
            f"{r['case'].split('cases/')[-1]:22s} conv {r['converged']}/{r['points']}  "
            f"max |ΔJ| {r['max_abs_A_m2']:.3e} A/m² ({r['max_rel_to_scale']:.2e} of scale) "
            f"at {r['at_V']:+.3f} V [SimSS {r['sim']:.4f}, jaxdd {r['jaxdd']:.4f}]  "
            f"J(0) {r['J_at_0'][0]:.4f}/{r['J_at_0'][1]:.4f}  "
            f"it {r['mean_iterations']:.1f}  {r['seconds']:.1f} s"
        )
