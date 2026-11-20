"""The tests' JAX setup.

In parallel (pytest-xdist, `addopts` in pyproject.toml), each worker one XLA thread on a CPU,
since the workers already fill the cores, and on a GPU no memory reserved ahead, so that the
workers share it. Set before JAX is imported.

A persistent compilation cache, so that a second run (and every worker after the first to
compile a program) loads what an earlier one compiled: in `.pytest_cache/jax` (not committed),
or `JAXDD_TEST_CACHE`; `JAXDD_TEST_CACHE=` (empty) turns it off. Tracing is not cached: a test
whose batch mixes many modules still takes its seconds.

`JAXDD_TEST_CLEAR_CACHES=1` empties JAX's in-memory caches after every test, as the CI does: a
worker otherwise keeps every program it compiled, on JAX 0.4.31 9–10 GB by the end of the suite
(3 GB at most with the caches emptied, 5 % slower: what is needed again comes from the
persistent cache).

On a GPU, `-n auto` takes a worker per `GIB_PER_WORKER` of the first card's memory, at most one
per physical core: the workers' memory grows and is not given back, and 16 workers on a 16 GB
card ran out (`RESOURCE_EXHAUSTED` in tests that change from run to run); 8 do not. The card's
memory comes from `nvidia-smi`, so that the main process does not take the GPU. With
`JAX_PLATFORMS=cpu`, or without `nvidia-smi`, a worker per physical core."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

if "PYTEST_XDIST_WORKER" in os.environ:
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false "
                                       "intra_op_parallelism_threads=1")  # fmt: skip
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

_cache = os.environ.get(
    "JAXDD_TEST_CACHE", str(Path(__file__).parents[1] / ".pytest_cache" / "jax")
)
if _cache:
    jax.config.update("jax_compilation_cache_dir", _cache)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


if os.environ.get("JAXDD_TEST_CLEAR_CACHES"):

    @pytest.fixture(autouse=True)
    def _clear_jax_caches():
        yield
        jax.clear_caches()


#: GPU memory per worker of `-n auto` (module docstring).
GIB_PER_WORKER = 2.0


def _gpu_memory_gib():
    """The first GPU's memory (GiB), or None without one that JAX would take."""
    if os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip() == "cpu":
        return None
    smi = shutil.which("nvidia-smi")
    if smi is None:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return float(out.split()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def pytest_xdist_auto_num_workers(config):
    """`-n auto`'s workers: a worker per physical core, on a GPU at most a worker per
    `GIB_PER_WORKER` of its memory (module docstring)."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        cores = os.cpu_count() or 1
    memory = _gpu_memory_gib()
    if memory is None:
        return cores
    return max(1, min(cores, int(memory // GIB_PER_WORKER)))
