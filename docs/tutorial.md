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

The voltages are the internal ones. With a shunt resistance `J` includes its current; with a
series resistance the external voltage of each point is `batch.external_voltages(batch, V, J)`
(SimSS's external circuit, [model.md](model.md)).

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
| `first_max_iter` | 6 | the first pass's limit: devices with a voltage that misses it are solved again, together, with `max_iter` ([batching.md](batching.md)); `None` for one pass, and one pass always inside `jax.jit`, `jax.jvp` or `jax.vmap` |
| `bernoulli_name` | `"exact"` | `"simss"` reproduces SimSS's series approximation of the Bernoulli function ([differences.md](differences.md) §2) |
| `strict` | `False` | `True` turns off the two shortcuts (the quadratic stop and the settled current, [solver.md](solver.md)), for checking them; about one Newton step more per voltage |

### Inside JAX transformations

`solve_batch` runs inside `jax.jit`, `jax.vmap`, `jax.jvp` and `jax.grad`, in one pass there
(the second pass's shape depends on how many devices the first leaves). The curves are
differentiable in every array of the batch and in the voltages, in forward and reverse mode. The
derivatives are those of the converged solution, by implicit differentiation
([solver.md](solver.md#derivatives)); they agree with finite differences (tests):

```python
def curves(g):                                   # J(V) at a share g of the generation
    return solve_batch(dict(batch, g_frac=batch["g_frac"] * g), V)[0]

J, dJ_dg = jax.jvp(curves, (1.0,), (1.0,))      # forward mode
```

A voltage not converged has zero derivatives.

To differentiate in the input's own parameters — a layer's mobility, thickness, band edge, trap
or ion density, the temperature, a work function — build the device in JAX from them.
`device_parameters` splits an input into its **structure** (points per layer, which layers
generate, hold traps or ions, the pre-bias sequence) and its **parameters** (a dict: each of
`device.LAYER_PARAMETERS` an array over the layers, and `T`, `W_L`, `W_R`, `G_frac`, `V_pre`,
the generation profile `gen_x`, `gen_G`); `device_tree` makes the batch's arrays from both, by
`build_device`'s formulas in `jax.numpy`:

```python
from jaxdd.batch import device_tree
from jaxdd.device import device_parameters

structure, params = device_parameters(setup)

def misfit(log_mu):                               # a curve against a measured one
    mu_n = jnp.asarray(params["mu_n"]).at[1].set(jnp.exp(log_mu))   # the absorber's
    p = dict(params, mu_n=mu_n)
    batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
    J, ok, _ = solve_batch(batch, V)
    return jnp.sum(jnp.where(ok[0], J[0] - J_measured, scale) ** 2)   # see below

grad = jax.grad(misfit)(jnp.log(1e-3))            # reverse mode
```

In a fit, do not leave out a voltage that did not converge: a zero residual there rewards the
optimiser for leaving the region where the solver converges, and it will (a recombination
coefficient run to infinity, every voltage unconverged, the misfit zero). Give it a residual of
the curve's scale instead; `validation/fit.py` is a whole fit (least squares, the Jacobian by
`jax.jacfwd`), `validation/fit_measured.py` one of measured curves at five light intensities
at once, `validation/sample_measured.py` its posterior by an ensemble sampler over the batch,
and [validation.md](validation.md) shows what such curves determine.

Devices of one structure batch by `jax.vmap(functools.partial(device_tree, structure))` over
stacked parameters, built on the GPU. A parameter change keeps the structure: the points per
layer stay those of the input's thicknesses (a thickness moves the grid, not its counts), and a
trap or ion density set on a layer that has none does nothing. The generation profile from a file
is stretched to the device as SimSS does, so a thickness moves it with the grid; it is not
recomputed optically.

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

Without any files, give `setup_from_texts` the texts by file name, under the names the setup
file uses (`l1`, `l2`, …, `genProfile`):

```python
from jaxdd.simss_input import setup_from_texts

setup = setup_from_texts({
    "simulation_setup.txt": setup_text,
    "L1_parameters.txt": etl_text, "L2_parameters.txt": absorber_text,
    "L3_parameters.txt": htl_text,
    "generation_profile.txt": profile_text,
})
dev = build_device(setup)
```

Reading and building a device is NumPy work on the host, about 1.5 ms on one core of a desktop
CPU; whatever writes the inputs usually costs more. For very large sweeps build the next
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
  `L_diff/d` 0.2), the ions pile up in nanometre layers at 1500 times their mean density and
  leave a carrier-free bulk between them. jaxdd converges there at every voltage (since its
  Jacobian is assembled by intervals, [solver.md](solver.md)); SimSS skips a third of those
  voltages. Whether such a profile is physical is another question: Boltzmann statistics let
  ions stack without limit ([PLAN.md](../PLAN.md), section 3).
- **`fixIons` without `preCond`** is refused: SimSS would hold the ions at their initial uniform
  density, which is rarely meant.
- **One grid size per batch**, and the grid is SIMsalabim's; no adaptive mesh yet
  ([PLAN.md](../PLAN.md)).
