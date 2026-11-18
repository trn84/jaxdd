"""Wall time of the same TRPL transient (the 600 nm FAPbI3 film of `trpl_common`, 142 steps to
20 µs) in jaxdd and ZimT 5.36. Both solve it with electrodes, the one model ZimT has: 'sfb', the
electrons' S at the glass and the holes' on top 0.1 m/s (the two ends where SimSS 5.36 has S's
sign right, docs/differences.md §7), 0 V. jaxdd: batches of devices with their mobility, k_rad
and deep traps spread by up to ±0.3 decades (a fit's or a study's spread), after compilation, on
JAX's default device. ZimT: one run alone, then `--parallel` runs at once (Docker, `simss/`).

    .venv/bin/python validation/trpl_timing.py [--batches 4 28 256 1024] [--parallel 16]

writes validation/trpl_timing.json. docs/validation.md, "TRPL without electrodes".
"""

import argparse
import json
import os
import platform
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from trpl_common import REPO, pl_of, pulse, run_zimt, setup, times, zimt_case

from jaxdd.batch import device_tree
from jaxdd.device import device_parameters
from jaxdd.transient import solve_transient, tj_columns

W, S = ("sfb", "sfb"), (0.1, 0, 0, 0.1)


def batch_of(n, rng):
    st, p = device_parameters(setup("film", W=W, S=S, NP=300))
    trees = []
    for _ in range(n):
        f = 10.0 ** rng.uniform(-0.3, 0.3, 3)
        q = dict(p)
        q["mu_n"] = jnp.asarray(p["mu_n"]) * f[0]
        q["mu_p"] = jnp.asarray(p["mu_p"]) * f[0]
        q["k_direct"] = jnp.asarray(p["k_direct"]) * f[1]
        q["bulk_trap_N"] = jnp.asarray(p["bulk_trap_N"]).at[0, 0].multiply(f[2])
        trees.append(device_tree(st, q))
    return jax.tree_util.tree_map(lambda *a: jnp.stack(a), *trees)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batches", type=int, nargs="+", default=[4, 28, 256, 1024])
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--cap", type=float, default=1800.0, help="seconds per ZimT run")
    a = ap.parse_args()
    t = times()
    rng = np.random.default_rng(0)
    cpu = platform.processor() or platform.machine()
    out = {"jax_device": jax.devices()[0].device_kind, "cpu": cpu, "cpus": os.cpu_count(),
           "steps": len(t), "jaxdd": {}}  # fmt: skip
    for n in a.batches:
        batch = batch_of(n, rng)
        r = solve_transient(batch, pulse(t), ions=False, vint=False)  # compiles
        t0 = time.time()
        r = solve_transient(batch, pulse(t), ions=False, vint=False)
        dt = time.time() - t0
        conv = float(np.asarray(r.converged).mean())
        out["jaxdd"][n] = {"seconds": dt, "per_transient": dt / n, "converged": conv}
        each = dt / n * 1e3
        print(f"jaxdd {n:5d} transients: {dt:8.2f} s, {each:8.1f} ms each, converged {conv:.3f}")
    reference = pl_of(tj_columns(r, 0), 1)
    work = Path(tempfile.mkdtemp(prefix="trpl_timing_"))

    def one(i):
        d = zimt_case(work / str(i), "film", W, S, t)
        t0 = time.time()
        z, finished, _ = run_zimt(d, a.cap)
        return time.time() - t0, finished and z is not None and len(z) == len(t)

    dt, ok = one(0)
    out["zimt_1"] = {"seconds": dt, "finished": ok}
    print(f"ZimT one run alone: {dt:.1f} s (all steps written: {ok})")
    t0 = time.time()
    with ThreadPoolExecutor(a.parallel) as ex:
        runs = list(ex.map(one, range(1, a.parallel + 1)))
    wall = time.time() - t0
    out["zimt_parallel"] = {"runs": a.parallel, "seconds": wall, "per_transient": wall / a.parallel,
                            "finished": sum(r[1] for r in runs)}  # fmt: skip
    print(f"ZimT {a.parallel} runs at once: {wall:.1f} s, {wall / a.parallel:.1f} s per transient")
    out["reference_pl_end"] = float(reference[-1])
    path = REPO / "validation" / "trpl_timing.json"
    path.write_text(json.dumps(out, indent=1))
    print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    main()
