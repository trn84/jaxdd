"""The tests' JAX setup.

In parallel (pytest-xdist, `addopts` in pyproject.toml), each worker one XLA thread on a CPU,
since the workers already fill the cores, and on a GPU no memory reserved ahead, so that the
workers share it. Set before JAX is imported.

A persistent compilation cache, so that a second run (and every worker after the first to
compile a program) loads what an earlier one compiled: in `.pytest_cache/jax` (not committed),
or `JAXDD_TEST_CACHE`; `JAXDD_TEST_CACHE=` (empty) turns it off. Tracing is not cached: a test
whose batch mixes many modules still takes its seconds."""

import os
from pathlib import Path

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
