# Tutorial: from a SimSS input to J-V curves

## Install

```bash
git clone https://github.com/trn84/jaxdd.git && cd jaxdd
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pip install "jax[cuda12]"          # only on an NVIDIA GPU
.venv/bin/python -c "import jax; print(jax.devices())"
.venv/bin/python -m pytest -q
```

jaxdd needs Python ≥ 3.11, JAX ≥ 0.4.30 and NumPy. It switches JAX to 64-bit floats when it is
imported. With a CUDA jaxlib installed, JAX uses the GPU by default; `JAX_PLATFORMS=cpu` forces
the CPU.

## One device

A device is described by SIMsalabim's files: `simulation_setup.txt`, one parameter file per
layer (`l1`, `l2`, ... in the setup), and the generation profile it names. The repository has
four such devices under `tests/data/`.

```python
from pathlib import Path

import numpy as np

from jaxdd.device import build_device
from jaxdd.simss_input import check_supported, read_setup
from jaxdd.solver import solve_jv

setup = read_setup(Path("tests/data/thin_l1_light/simulation_setup.txt"))
check_supported(setup)              # NotImplementedError naming anything outside the scope
dev = build_device(setup)           # SIMsalabim's grid and every static array (NumPy)

V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
res = solve_jv(dev, V)
res.J            # A/m², SIMsalabim's sign: the photocurrent is negative
res.converged    # per voltage
res.iterations   # Newton steps per voltage
```

`read_setup` takes a `Path`; the layer files and the generation profile are found relative to
it. `solve_jv` solves the voltages in the order given, each from the solution at the one before.
The device's own `G_frac` sets the light; `solve_jv(dev, V, g_frac=0.0)` gives the dark curve of
the same device. If the input has `preCond = 1`, the device is first settled at `Vpre` with its
ions free (and with `fixIons = 1` the ions are then held for the scan); a pre-bias that does not
converge raises `RuntimeError`.

`solve_jv` is the reference implementation: a Python loop over voltages and Newton steps, easy
to step through, about 20 curves per second on one CPU core. For many devices use the batch.

## Many devices

```python
from jaxdd.batch import prepare, solve_batch

cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
devices = [build_device(read_setup(Path(f"tests/data/{c}/simulation_setup.txt"))) for c in cases]

batch = prepare(devices)            # every array stacked along a new first axis
J, converged, iterations = solve_batch(batch, V)     # each: devices × voltages
```

The devices of one batch may differ in everything but the number of grid points `NP` (and they
share the voltages). `prepare` refuses a mix of grid sizes; group devices by `NP` and solve each
group as its own batch.

The first call compiles. That takes seconds on a CPU and up to a minute on a GPU at large
batches, and it happens again for every new **shape or option**: another batch size, another
`NP`, another number of voltages, another `max_iter`, `first_max_iter`, `bernoulli_name` or
`strict`. For a long sweep, cut the devices into batches of one fixed size (pad the last one by
repeating a device) and keep the options fixed; then every batch after the first runs without
compiling, but for the second pass, which compiles once per power of two of stragglers.

### Options

| Argument | Default | Meaning |
|---|---|---|
| `tol` | 1e-10 | the Newton step (potential in `V_T`, densities relatively) at which a voltage is converged |
| `max_iter` | 60 | Newton steps per voltage at most; a voltage that reaches it is flagged, not dropped |
| `first_max_iter` | 6 | the first pass's limit: devices with a voltage that misses it are solved again with `max_iter` ([batching.md](batching.md)); `None` for one pass |
| `bernoulli_name` | `"exact"` | `"simss"` reproduces SimSS's series approximation of the Bernoulli function ([differences.md](differences.md) §2) |
| `strict` | `False` | `True` turns off the two shortcuts (the quadratic stop and the settled current, [solver.md](solver.md)), for checking them; about one Newton step more per voltage |

### Reading the result

Every voltage of every device carries its `converged` flag; nothing is cut off silently. A
device whose pre-bias did not converge is flagged at every voltage. Mask by the flag before
using the curves (for a table, a fit or a training set). The `iterations` show where a batch
spends its time ([batching.md](batching.md)).

## Building devices in code

`build_device` takes a `Setup`, which is the parsed files as strings. To vary parameters
without writing files, change the strings and build again; `validation/bench.py`
(`variant_setups`) does exactly this:

```python
from jaxdd.simss_input import Layer, Setup

base = read_setup(Path("tests/data/thin_l1_light/simulation_setup.txt"))
layers = [dict(ly.raw) for ly in base.layers]
layers[1]["N_t_bulk"] = "3e21"                       # trap density of the absorber, m⁻³
variant = Setup(raw=dict(base.raw), layers=tuple(Layer(r) for r in layers),
                directory=base.directory)           # the generation profile is read from here
dev = build_device(variant)
```

Building a device is NumPy work on the host, about 25 ms. For very large sweeps build the next
batch's devices in worker processes while the solver runs on the current one.

## Comparing with SimSS

To compare with a SimSS run of the same input:

```bash
.venv/bin/python validation/compare_simss.py path/to/case_dir      # inputs + SimSS's JV.dat
```

Expect agreement to 1e-6 of the current scale with `bernoulli_name="simss"` and to a few 1e-4
with the exact Bernoulli function. Larger gaps are usually SimSS's: rerun SimSS with
`tolDens = tolPois = 1e-9` before suspecting jaxdd ([differences.md](differences.md),
[validation.md](validation.md)).

## Known limits

- **Strong frozen charge.** At 1e18 cm⁻³ of ions frozen at full accumulation (thin absorber,
  `L_diff/d` 0.2), the ions pile up in nanometre layers and leave a carrier-free bulk between
  them. There the potential keeps a few 1e-5 `V_T` of round-off noise; the batched solver
  converges by the settled current (noise at most 3e-6 of the scale). SimSS skips a third of
  those voltages. In the dark, from about 0.8 V up, with ions frozen at full accumulation and
  5e17 cm⁻³ or more on the device, single voltages can still end unconverged (flagged): on a
  large parameter grid under 2e-5 of all voltages ([validation.md](validation.md)).
- **`fixIons` without `preCond`** is refused: SimSS would hold the ions at their initial uniform
  density, which is rarely meant.
- **One grid size per batch**, and the grid is SIMsalabim's; no adaptive mesh yet
  ([PLAN.md](../PLAN.md)).
