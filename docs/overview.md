# What jaxdd is, and why it is built this way

jaxdd is a one-dimensional drift-diffusion solver for semiconductor devices, perovskite solar
cells first. It solves the same discrete equations as
[SIMsalabim](https://github.com/kostergroup/SIMsalabim) 5.36 (its steady-state program SimSS),
reads SIMsalabim's own input files, and is validated against SimSS's output. What it adds is
scale: written in JAX, it solves thousands to hundreds of thousands of devices at once, on a GPU
or on every core of a CPU.

## The problem it solves

A drift-diffusion run of one device is small: about 400 grid points, three unknowns per point,
a Newton step of a few hundred kFLOP. SimSS does it in a fraction of a second to a few seconds,
and a faster single-device code would not change much. What costs time is the *sweep*: a table
of J-V curves over a parameter grid (absorber thickness, diffusion length, doping, ion density,
...) is hundreds of thousands to millions of devices, each another SimSS process. On all cores
of a 16-core desktop SimSS manages about 45 J-V curves per second; a million curves take six
hours, and every further axis multiplies that.

The devices of a sweep are independent. Stacked along a batch axis, 10⁴–10⁵ of them form one
large computation: a block-tridiagonal Newton system per device, sequential along the grid,
parallel across the batch. That is the shape a GPU (and, differently, a CPU with many cores)
handles well. jaxdd writes the solver once, for one device, as a pure JAX function, and lets
`jax.vmap` turn it into the batched one.

**FP64 throughout.** Carrier densities span some 20 orders of magnitude, and the
Scharfetter–Gummel fluxes subtract nearly equal exponentials. jaxdd runs with `jax_enable_x64`
everywhere. That decides the hardware: data-centre GPUs (A100, H100, MI50) do FP64 at half
their FP32 rate, consumer GPUs at 1/64, and there a CPU is as fast ([hardware.md](hardware.md)).

## Words used throughout

| Word | Meaning |
|---|---|
| device | one solar cell: its layers, parameters and generation profile, as one SIMsalabim input |
| run, curve | one J-V curve of one device: the steady state at every voltage of a list (61 voltages, −0.2 to 1.3 V, in the benchmarks) |
| batch | devices solved together in one compiled program; they share the grid size `NP` and the voltages |
| lane | one device's slot in a batch |
| pre-bias | SimSS's `preCond`: the device is first settled at `Vpre` with its ions free |
| frozen ions | SimSS's `fixIons` after a pre-bias: the ion profile of the pre-bias held fixed for the scan |
| current scale | the generation current `q·∫G dx` (at least 1 A/m²): accuracy is stated relative to it |
| `η` | collection efficiency `(J_dark − J_light)/J_gen`, a normalised measure in which curves of different devices are compared |

## Scope: what jaxdd solves today

The physics is the steady state of a solar cell with mobile ions ("Stage A", the part of SimSS
that parameter sweeps of perovskite cells use most), general in structure: any number of layers,
each with its own parameters.

| Part | Supported |
|---|---|
| Layers | any number; per layer thickness, permittivity, band edges, effective density of states, mobilities, donor and acceptor doping |
| Grid | SIMsalabim's: `NP` points, exponentially graded in each layer (`grad`) |
| Interfaces | band offsets; thermionic transfer at a finite velocity (`nu_int_n`, `nu_int_p`); interface traps, one level per interface or several from a trap file (`intTrapFile`), neutral or charged (`N_t_int`, `E_t_int`, `intTrapType`; Stage B) |
| Contacts | at the work functions `W_L`, `W_R` (or `sfb`), anywhere in the gap (a Schottky barrier away from the band edges); infinite or finite surface recombination (`S_n_L`, `S_p_L`, `S_n_R`, `S_p_R`, Stage B); the left electrode the cathode or the anode (`leftElec = -1` or `1`, Stage B); series and shunt resistance as SimSS's external circuit (Stage B) |
| Recombination | direct (`k_direct`); bulk SRH through one trap level per layer or several from a trap file (`bulkTrapFile`), neutral or charged (`bulkTrapType`; Stage B) |
| Generation | a profile from a file, uniform per layer (`genProfile = none`), or by the transfer-matrix method from n,k data and a spectrum (`calc`, Stage B); scaled by `G_frac` |
| Ions | anions and cations, each mobile in one layer: free at every voltage, or settled at a pre-bias and frozen (`preCond`, `fixIons`, as SimSS with [`simss/fixed_ions.patch`](../simss/README.md)); immobile ions as fixed charge |
| Output | `J(V)` and a `converged` flag per device and voltage; the number of Newton steps |

Everything else is **refused by name**, never approximated: `check_supported` raises
`NotImplementedError` listing each parameter outside this scope (field-dependent mobility or
generation, Langevin recombination, an ionic region across layers, `fixIons`
without `preCond`). What comes next in physics is
in [PLAN.md](../PLAN.md).

## Where jaxdd follows SIMsalabim, and where it does not

jaxdd solves SIMsalabim's **discrete equations**: the same grid, the same Scharfetter–Gummel
discretisation in the generalised potentials, the same treatment of interfaces, the same
integral current, SIMsalabim's physical constants (`q`, `k`, `ε₀` to five digits). A converged
jaxdd solution is therefore SimSS's solution, not merely a similar one, and the two can be
compared to round-off.

It solves them **differently**: SimSS iterates Poisson's equation and the two continuity
equations in turn (Gummel), one device at a time; jaxdd takes a coupled Newton step on all
unknowns of all devices at once ([solver.md](solver.md)). Where SimSS's answer is not the
solution of its own equations (it stops early, it freezes an unconverged pre-bias) or uses an
approximation jaxdd can avoid (its Bernoulli series), jaxdd takes the correct answer by default
and offers a switch that reproduces SimSS's number. Each such case is in
[differences.md](differences.md).

## Design decisions

| Question | Choice | Why |
|---|---|---|
| Language | JAX (Python) | batching by `vmap`, the Jacobian by automatic differentiation, CPU and GPU from one code, gradients of the curves (by implicit differentiation); Python, where most sweeps and fits are written |
| Unknowns | `(V, ln n, ln p)` per grid point | densities never go negative, the 20 orders of magnitude live in the logarithm; Scharfetter–Gummel stays in `n`, `p` |
| Nonlinear solver | coupled Newton | few steps per voltage, robust at high injection; a Gummel scheme needs many more iterations and does not batch better |
| Jacobian | by intervals: each Scharfetter–Gummel flux and its derivatives once (`B'` by `jax.jvp`), each point's 3×3 blocks from its two intervals; tested against `jax.jacfwd` of one point's residual | 4× faster than autodiff of each point, and no round-off floor where frozen ions leave a carrier-free bulk; never the dense Jacobian of the whole device |
| Linear solver | block Thomas sweep: a `lax.scan`, and for a batch on an NVIDIA GPU one Pallas kernel | the system is block tridiagonal; the kernel saves the scan's ~1450 kernel launches per sweep; parallel cyclic reduction was built and measured slower on every machine and batch size |
| Input | SIMsalabim's parameter files | every SimSS user's sweep runs unchanged; a Python API of pytrees beside it |
| Scope | general structure, physics in stages | anyone who sweeps SimSS can use it; each stage of physics validated against SimSS before the next |
| First GPU | NVIDIA A100 (CUDA) | JAX works there as is; the code stays plain JAX, so other platforms have nothing to undo |
| Licence | LGPL-3.0, like SIMsalabim | SIMsalabim's code may be carried over where it helps; programs under other licences may call jaxdd |

## The repository

| Path | What |
|---|---|
| `src/jaxdd/simss_input.py` | reading SIMsalabim's input files, from a directory (`read_setup`) or from texts (`setup_from_texts`); `check_supported` |
| `src/jaxdd/device.py` | SIMsalabim's grid and every static array of one device: its structure and parameters (`device_parameters`), the arrays from them in NumPy (`build_device`) or JAX (`device_arrays`) |
| `src/jaxdd/solver.py` | the discrete equations, the Newton step, the block sweep; `solve_jv` for one device |
| `src/jaxdd/batch.py` | the same solver as one pure function, `vmap`ped: `prepare`, `solve_batch` |
| `src/jaxdd/sweep.py` | the batch's block sweep in one GPU kernel (Pallas), and the switch between it and the scan |
| `src/jaxdd/constants.py` | SIMsalabim's physical constants |
| `tests/` | pytest suite; `tests/data/` four SimSS runs (inputs and `JV.dat`) as reference |
| `validation/` | scripts that compare with SimSS and time the solver ([validation.md](validation.md)) |
| `benchmarks/` | jaxdd against SimSS per machine, the README's figure ([hardware.md](hardware.md)) |
| `rocm/` | running on AMD MI50 (gfx906) cards ([hardware.md](hardware.md)) |
| `simss/` | building SimSS, the reference, with the patch for frozen ions ([README](../simss/README.md)) |
| `PLAN.md` | what is planned and what may come |
| `NOTICE`, `COPYING*` | licence and attribution |

## Licence and attribution

jaxdd is under the LGPL-3.0, as SIMsalabim. [`NOTICE`](../NOTICE) names SIMsalabim and lists
what derives from its code; it is kept current with every such change:

- A file whose code is translated or carried over from SIMsalabim keeps SIMsalabim's
  copyright notice in its header, says that it was translated and changed, names the
  SIMsalabim unit and version it follows and the date, and is listed in `NOTICE`
  (GPL-3.0 §5 a).
- A module that follows SIMsalabim's equations or routines names them in its docstring, as
  `device.py` and `solver.py` do; where it carries over code, the rule above applies too.
- A change to SIMsalabim itself (the patch in `simss/`) is listed in `NOTICE` with the file,
  the version and the date.

Every published result obtained with jaxdd cites SIMsalabim, as the README's
[Citing](../README.md#citing) asks.
