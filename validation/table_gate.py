"""The §7 gate on FauxPV's table: jaxdd's η against SimSS's on sampled devices without ions.

    .venv/bin/python validation/table_gate.py ../jaxdd-ref/table_sample [simss|exact]

`η = (J_dark − J_light)/J_gen`, compared from −0.2 to 1.0 V where SimSS converged; written to
validation/table_gate_<bernoulli>.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

from jaxdd.device import build_device
from jaxdd.simss_input import read_setup
from jaxdd.solver import solve_jv

V_TOP = 1.0


def main(sample: Path, bernoulli_name: str = "simss") -> None:
    ref = json.loads((sample / "reference.json").read_text())
    V = np.array(ref["V"])
    mask = V <= V_TOP + 1e-9
    j_gen = ref["J_gen"]
    rows = []
    start = time.perf_counter()
    for k, name in enumerate(ref["names"]):
        curves = {}
        converged = True
        for kind in ("light", "dark"):
            dev = build_device(read_setup(sample / name / kind / "simulation_setup.txt"))
            res = solve_jv(dev, V, bernoulli_name=bernoulli_name)
            curves[kind] = res.J
            converged &= bool(res.converged.all())
        eta = (curves["dark"] - curves["light"]) / j_gen
        sim = (np.array(ref["J_dark"][k]) - np.array(ref["J_light"][k])) / j_gen
        ok = mask & np.isfinite(sim)
        gap = np.abs(eta - sim)[ok]
        at_0 = int(np.argmin(abs(V)))
        rows.append({"name": name, "index": ref["index"][k], "converged": converged,
                     "eta_max": float(gap.max()), "eta_0V": float(abs(eta - sim)[at_0]),
                     "skipped_by_simss": int((~np.isfinite(sim) & mask).sum())})  # fmt: skip
        if (k + 1) % 100 == 0:
            done = np.array([r["eta_max"] for r in rows])
            print(f"{k + 1}: max {done.max():.2e}, 99th pct {np.percentile(done, 99):.2e}, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)  # fmt: skip
    out = Path(__file__).with_name(f"table_gate_{bernoulli_name}.json")
    out.write_text(json.dumps(rows, indent=1))
    eta = np.array([r["eta_max"] for r in rows])
    print(
        f"{len(rows)} devices, converged {sum(r['converged'] for r in rows)}; η gap max "
        f"{eta.max():.2e}, median {np.median(eta):.2e}, beyond 1e-3: {(eta > 1e-3).sum()}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), *(sys.argv[2:3] or []))
