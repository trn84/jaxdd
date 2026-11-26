"""Bit-identity on the CPU (PLAN.md §3.1): the same jaxdd, JAX and backend give the same bits
in a new process, which a caller that promises reproducible datasets relies on."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SOLVES = Path(__file__).parent / "reproducible_solves.py"

#: One XLA thread, as the tests' workers run (tests/conftest.py), and XLA's default threads.
ONE_THREAD = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


def test_solves_repeat_bit_for_bit_in_fresh_processes(tmp_path) -> None:
    """`solve_batch` (a J-V scan with ions frozen after a pre-bias), `solve_states` and
    `solve_transient` (mobile ions, with the profiles) in two new processes on the CPU, one on
    one XLA thread and one on XLA's default threads, without the compilation cache: every array
    equal to the bit (`array_equal`), NaN where NaN."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_XDIST")}
    env.update(JAX_PLATFORMS="cpu", PYTHONPATH=os.pathsep.join(sys.path))
    env.pop("JAX_COMPILATION_CACHE_DIR", None)
    runs = []
    for k, flags in enumerate((ONE_THREAD, None)):
        e = dict(env)
        if flags is None:
            e.pop("XLA_FLAGS", None)
        else:
            e["XLA_FLAGS"] = flags
        out = tmp_path / f"run{k}.npz"
        runs.append((out, subprocess.Popen([sys.executable, str(SOLVES), str(out)], env=e)))
    for _, proc in runs:
        assert proc.wait(timeout=1200) == 0
    a, b = (np.load(out) for out, _ in runs)
    assert sorted(a.files) == sorted(b.files) and len(a.files) > 20
    differ = [k for k in a.files if not np.array_equal(a[k], b[k], equal_nan=True)]
    assert not differ, differ
    assert a["jv_ok"].all() and np.isfinite(a["tr_J"]).all()
