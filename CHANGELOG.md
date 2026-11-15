# Changelog

What changes from one release of jaxdd to the next. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the versions
[semantic versioning](https://semver.org/): before 1.0 a minor version (0.2, 0.3, …) may change
the API, and says so here.

## [Unreleased]

### Changed

- **The current in time.** With the integral form of the current (`currDiffInt = 2`, ZimT's
  default) `solve_transient` sums each density's change over its point's control volume, where
  0.1.0 summed as ZimT 5.36 does (over the interval after each point, the left contact's
  change among them). ZimT's sums converge at first order in the grid: on ZimT's cases at 400
  points the current was 2e-3 to 3e-2 of its scale off, on either grid; it is now the current
  the equations balance, the differential form's to 5e-11, 4e-5 to 9e-4 off at 400 points
  (the grid's own error). Transients with `currDiffInt = 2` change by that much, the open
  circuit's and a series resistance's voltages with them; the steady state and
  `currDiffInt = 1` do not. ZimT's sums with the new `solve_transient(..., zimt_current=True)`
  ([docs/differences.md](docs/differences.md) §13).

### Fixed

- **The QFLS in the dark and in transport layers.** A density below Newton's floor of
  convergence (the minority carriers in the dark, in a transport layer) was left wherever
  Newton stopped, and the QFLS, which takes its logarithm, was up to 5e-3 V off in the dark
  after the light. Such densities are now stepped in the density instead of its logarithm and
  converge with the rest at no more Newton steps; the QFLS is ZimT's to 3.5e-6 V in the dark
  after the light and 5e-11 V under light. The currents do not change; steady-state curves
  within the settled-current stop (docs/solver.md, "Densities below the floor").

## [0.1.0] – 2026-11-15

The first release: SimSS 5.36's steady state and ZimT 5.36's transients, batched over thousands
of devices on CPUs and NVIDIA GPUs, differentiable in the input's parameters.

### The steady state (SimSS 5.36)

- SIMsalabim's input files, from a directory (`read_setup`) or from texts
  (`setup_from_texts`); `check_supported` refuses by name whatever jaxdd does not solve.
- SimSS's physics with all of its modules: any number of layers; band offsets and thermionic
  interfaces; bulk and interface traps, neutral or charged, one level or several from a trap
  file; direct, Langevin and SRH recombination; field-dependent mobilities; contacts anywhere in
  the gap, finite surface recombination, either electrode the cathode; series and shunt
  resistance; generation from a file, uniform, or by the transfer-matrix method, with Braun's
  field-dependent dissociation; mobile ions free at every voltage, frozen after a pre-bias, or
  held from the start; SimSS's voltages (`simss_voltages`).
- `solve_jv` for one device; `prepare` and `solve_batch` for a batch.
- Against SimSS 5.36 on SimSS's grid, to 1e-6 of the current scale with SimSS's approximation of
  the Bernoulli function (`bernoulli_name="simss"`). Where jaxdd's answer differs on purpose
  (SimSS stopping early, its Bernoulli series, an unconverged pre-bias, …) it takes the correct
  one and a switch reproduces SimSS's: [docs/differences.md](docs/differences.md).

### Transients (ZimT 5.36)

- `solve_transient` through ZimT's time steps (`read_tvg`): electrons, holes, traps and mobile
  ions in time by implicit Euler, the voltage given, at the open circuit or behind a series
  resistance; ZimT's automatic stop (`Track`) and `failureMode`; every column of its `tJFile`
  (`tj_columns`). Against ZimT 5.36 to 1e-9 of the current's scale on the committed cases.

### The grid

- jaxdd's own grid by default, graded towards every layer's ends: at SimSS's 400 points it meets
  the validation criteria on 99.6 % of 4096 hard devices, where SimSS's grid meets them on
  27 %. SIMsalabim's grid is `build_device(setup, grid="simss")`, for comparisons with SimSS
  point by point ([docs/model.md](docs/model.md#the-grid), [docs/validation.md](docs/validation.md)).

### Speed

- A batch is one compiled program: Newton steps of every device at once, a second pass for the
  devices a first one leaves, the block sweep in one Pallas kernel (Triton or Mosaic GPU) on
  NVIDIA GPUs, several GPUs of one process sharing a batch.
- About 2800 J-V curves per second on one A100, 730 on a 16-core desktop CPU: 60× and 16× SimSS
  on all cores of that CPU ([docs/hardware.md](docs/hardware.md)).

### Derivatives and fits

- The curves and the transients are differentiable, by implicit differentiation of the
  converged solution: the steady state in forward and reverse mode, the transients in forward
  mode. `device_parameters` and `device_tree` build a batch in JAX from the input's parameters
  (thicknesses, mobilities, energies, densities, …), so that gradients reach them.
- `least_squares_many`: least squares from many starts at once, their evaluations one batch
  (needs SciPy: the `fit` extra).

### The package

- `import jaxdd` gives the API above at the top level, and `jaxdd.__version__`. Importing it
  switches JAX to 64-bit floats.
- Python 3.11 to 3.13; JAX 0.4.31 or newer with NumPy 1.26 or newer, both ends tested in CI.
  Extras: `cuda12` (the CUDA jaxlib), `fit` (SciPy), `dev` (tests, lint), `bench` (the figure).

### Known limits

- One grid size `NP` per batch, the grid set from the layers' thicknesses before the solve.
- Transients: the quasi-Fermi level splitting where a carrier lies below Newton's floor of
  convergence is up to 5e-3 V off ZimT's in the dark and up to 3 V in an ionic region's
  depletion zones; ZimT's `varFile` is not written. A transient is not split across several GPUs.
- AMD GPUs: prepared for the MI50 (gfx906, JAX 0.4.31 in a ROCm image), not yet measured.

[Unreleased]: https://github.com/trn84/jaxdd/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trn84/jaxdd/releases/tag/v0.1.0
