"""jaxdd: a batched one-dimensional drift-diffusion solver in JAX, after SIMsalabim.

SIMsalabim's input files in (`read_setup`, `setup_from_texts`), a device built from them
(`build_device`), and its J-V curve (`solve_jv`), a batch's (`prepare`, `solve_batch`) or a
batch's transients through ZimT's time steps (`read_tvg`, `solve_transient`). The handbook is in
the repository's `docs/`; the modules hold the rest (`jaxdd.device`, `jaxdd.batch`, ...).

Importing jaxdd switches JAX to 64-bit floats (`jax_enable_x64`), which the solver needs.
"""

from importlib.metadata import PackageNotFoundError, version

from jaxdd.batch import device_tree, external_voltages, prepare, solve_batch, solve_states
from jaxdd.device import Device, Structure, build_device, device_parameters
from jaxdd.fit import least_squares_many
from jaxdd.simss_input import (
    TVG,
    Layer,
    Setup,
    check_supported,
    read_setup,
    read_tvg,
    setup_from_texts,
    simss_voltages,
)
from jaxdd.solver import JVResult, StateResult, solve_jv, solve_state
from jaxdd.transient import TransientResult, solve_transient, tj_columns

try:
    __version__ = version("jaxdd")
except PackageNotFoundError:  # a source tree on the path, not installed
    __version__ = "unknown"

__all__ = [
    "TVG",
    "Device",
    "JVResult",
    "Layer",
    "Setup",
    "StateResult",
    "Structure",
    "TransientResult",
    "__version__",
    "build_device",
    "check_supported",
    "device_parameters",
    "device_tree",
    "external_voltages",
    "least_squares_many",
    "prepare",
    "read_setup",
    "read_tvg",
    "setup_from_texts",
    "simss_voltages",
    "solve_batch",
    "solve_jv",
    "solve_state",
    "solve_states",
    "solve_transient",
    "tj_columns",
]
