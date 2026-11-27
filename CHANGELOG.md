# Changelog

What changes from one release of jaxdd to the next. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the versions
[semantic versioning](https://semver.org/): before 1.0 a minor version (0.2, 0.3, …) may change
the API, and says so here.

## [Unreleased]

### Added

Devices at several temperatures (PLAN.md §1 of 0.5.0):

- **Temperature laws** (`jaxdd.temperature`, jaxdd's): `at_temperature(params, T, laws, gap)`
  maps `device_parameters`' dict from its own `T` to another in front of `device_tree`, pure
  and differentiable in `T` and the laws' coefficients, batched by `vmap`. A `Law(m, E)` scales
  a density of states, mobility or rate parameter by `(T/T0)^m·exp(−(E/k)(1/T − 1/T0))`, or the
  caller's own function does; `textbook_laws` gives `N_c` `m = 3/2`, `ν_int` `1/2` and the ions'
  mobilities `−1` (an Arrhenius `D`). A `Gap(slope, share)` moves `E_c` and `E_v`; trap levels
  and fixed work functions stay or follow a band or the mid-gap (`traps`, `contacts`), and
  `edge=True` moves the absorption edge with it. `check_gaps` checks a mapped dict's levels and
  work functions against its gaps. At the reference temperature every device is the unmapped
  one to the bit.
- **An absorption edge that follows the band gap** (jaxdd's): a per-layer `dE_nk` (eV, 0 by
  default, `device.OPTICS_PARAMETERS`) shifts the layer's n,k in photon energy under
  `genProfile = calc`, traced and differentiable, so devices at several temperatures share a
  `Structure` and a batch. A zero shift gives the device and its derivatives in the thicknesses
  as before, to the bit.

### Fixed

- **docs/model.md, "Interfaces"**: the Scharfetter–Gummel flux across an interface is thermionic
  emission at the effective velocity `ν_int·|x|/(1 − e^(−|x|))`, about `ν_int·ΔE/kT` at a large
  offset, not at `ν_int`; the model was and is SIMsalabim's.

## [0.5.0] – 2026-11-26

### Added

What a caller that ages devices step by step needs (PLAN.md §3.1 of 0.4.0):

- **A transient that goes on from a state**: `solve_transient(..., initial=state)` starts every
  device from a `TransientState` (the potential, densities and ions on every point, the traps'
  occupations, the internal voltage, the time) instead of a steady state.
  `TransientResult.final` is the last state kept, `TransientResult.state(k)` any step's (with
  `profiles=True`), and `TransientResult.t` the steps' times. A transient split at any step and
  continued so gives the unsplit one's currents to 1e-11 of their scale
  (`tests/test_transient_state.py`, four ZimT cases: mobile ions, charged bulk traps, interface
  traps, surface recombination).
- **The ions and the traps' occupations in a transient's profiles**: `profiles` holds `anion`,
  `cation` (0 outside their regions), `f_bulk`, `f_interface` and each step's `dti` beside `V`,
  `n`, `p`; the ions count each region's total at every step.
- **Auger recombination** (jaxdd's): per layer `C_n_Auger`, `C_p_Auger` (m⁶/s), the rate
  `(C_n·n + C_p·p)·(n·p − n_i²)` in the steady state and in time, differentiable; a transient's
  `JAuger` column per layer. 0 where a layer file does not name them, and a device without any
  has no Auger term at all, so SimSS's inputs solve to the bit as before.
- **Bit-identity on the CPU, tested**: `solve_batch`, `solve_states` and `solve_transient` give
  the same bits in two new processes, on one XLA thread and on XLA's default threads
  (`tests/test_reproducible.py`).

### Changed

- **The TRPL posterior with the SnO2 decays' background taken off**: `validation/trpl_sample.py
  --outcoupling --neutral --background` (chains `trpl_bg_chain_s<seed>.npz`); where its best fit
  is missing it is refitted first from the fit without the background, and
  `benchmarks/apptainer.sh`'s trpl step does so once before its chains (`--fit-only`) and
  uploads the TRPL fits and summaries with the chains.
- **The TRPL fit's bounds widened** where that posterior ran against them or was cut by them, in
  `validation/trpl_joint.py` for every model that has the parameter: the SnO2 interface's SRV to
  1e5 m/s, Spiro's E_v from 4.8 eV, the boost to 1000 (sampled boost × reach to their bounds'
  product, not 1e-5 m), the interface states' level from 4.35 eV (with them SnO2's E_c at most
  4.3 eV). The absorbed shares stay at most 1 (opened, one refit traded them against `k_rad`) and
  SnO2's doping at most 10^25.5 m⁻³ (Boltzmann's statistics; docs/validation.md).
  `trpl_sample.py --neutral` takes SnO2's E_c from 3.9 eV; with `--background` its fit, chains
  and summaries are now named `ec3.9` (`trpl_bg_ec3.9_chain_s<seed>.npz`). The chains save their
  bounds, and `--resume` refuses a chain with others or none.
- `trpl_sample.py`'s refit takes up to 200 evaluations (60 did not converge), refuses a saved fit
  outside the bounds, and its batches are padded to multiples of 8 parameter sets, so that the
  solver compiles for at most 16 batch sizes in the half-steps instead of 128.

- `docs/validation.md`'s TRPL section describes the current model, its bounds and the posterior
  of the full model within them (two chains of 256 walkers, 1000 steps from a converged chain
  within the narrower bounds, steps 750–1000 kept): mobility 39.5 (37–43) cm²/Vs, `k_rad` 1.30
  (1.22–1.42)·10⁻¹⁰ cm³/s, the SnO2 interface's SRV off its bound at 1.1·10⁵ cm/s, the boost
  bounded only from below. The posteriors of the simpler models and the history of the bounds are
  gone.
- The handbook (`docs/`) describes the current state only: dates, "Stage" labels and accounts of
  what was tried before are gone or stated as the reasons for what is there, and the pages are
  shorter (about 170 lines fewer) with the same results. `docs/tutorial.md` no longer calls ions
  bounded by their sites a plan (they are `N_max_anion`, `N_max_cation`).

### Fixed

- `docs/validation.md` gave the TRPL posterior without the SnO2 decays' background about 2200
  independent samples: the autocorrelation time estimated from its chains grows with the window,
  so no such number holds.
- `docs/validation.md` gave the TRPL posterior with the saturable states (in 0.4.0) 4200
  independent samples: the steps kept hold about 2500, and roughly so (five autocorrelation
  times).
- The SnO2 decays' background in `docs/validation.md` and `trpl_common.sno2_data`: 1.19e-4,
  2.42e-4 and 1.73e-3 of the peak over 15–20 µs, which is what is subtracted (the values given
  were 8–20 µs's); the second decay is flat only from about 12 µs, and the count rate alone does
  not explain the third's level. The fits are unaffected.
- `trpl_sample.py` started walkers whose boost was outside its bounds (inside those of boost ×
  reach and reach), at a log-probability of -inf; with the residuals stubbed some stayed into the
  samples (`sample_measured.start(..., allowed=)`).
- `docs/validation.md` said the bounds no longer carry the TRPL posterior without the
  background; they cut five of its 26 parameters.
- `trpl_sample.py --outcoupling` without `--neutral` wrote `trpl_chain.npz` and
  `trpl_sample.json`, so two seeds side by side overwrote each other's chain; again
  `trpl_out_chain_s<seed>.npz` and `trpl_sample_out_s<seed>.json`.

## [0.4.0] – 2026-11-21

Impedance in the frequency domain, the carriers' statistics beyond Boltzmann's (Blakemore's,
Fermi–Dirac's), mobile ions bounded by their sites in time with two flux schemes, adaptive sparse
grids for tables, and ChargeTransport.jl as a second reference. Nothing of 0.3.0's API is removed;
with Boltzmann's statistics and ions without sites every result is the same, except where the
fixes below let a solve converge that failed before.

### Added

- **Ions bounded by their sites in time.** `solve_transient` takes mobile ions with
  `N_max_anion`, `N_max_cation` (it refused them): they flow down their electrochemical
  potential at a constant mobility, the Scharfetter–Gummel flux with the excess chemical
  potential `−ln(1 − n/N_max)` beside the potential's (ChargeTransport.jl's scheme;
  docs/model.md). Without sites every transient is the same to the bit.
- **Sparse grids.** `jaxdd.sparse.refine_sparse(evaluate, ranges, tol)`: a spatially adaptive
  sparse grid (piecewise multilinear hierarchical basis, children where a surplus exceeds
  `tol`, each round in one call) and `interpolate_sparse`. On a perovskite cell's table over
  four axes within 5e-2 with 10 858 points where `jaxdd.axes.refine_axes` took 50 274
  (docs/validation.md).
- **Impedance in the frequency domain.** `jaxdd.solve_impedance(batch, frequencies, voltage)`:
  one complex linear solve per frequency at the steady state, with mobile ions (each region's
  number kept), the bulk traps' occupations at their own rates, a series resistance and a
  shunt (`ImpedanceResult`: `Z`, `Y`, `capacitance`). With `dt=` it is what `solve_transient`
  sees through implicit Euler at that step: against its response to a small sine to 1e-7
  (docs/model.md, docs/validation.md), with interface traps too. A free end is refused.
- **The ions' other dynamics.** `solve_transient(..., ion_flux="modified-drift")`: ions bounded
  by their sites flow with the drift modified by the free sites' share at a constant
  diffusivity, Bessemoulin-Chatard's scheme as ChargeTransport.jl's
  `DiffusionEnhancedModifiedDrift`; the same steady states, other transients. Against
  ChargeTransport.jl's voltage steps to 1e-2 in the current (docs/validation.md).
- **ChargeTransport.jl as a reference** (`validation/chargetransport/psc_ions.jl`,
  `tests/test_chargetransport.py`): its Ex103 perovskite cell with vacancies bounded by their
  sites, in equilibrium, dark and lit steady states and a voltage's step in time; jaxdd agrees to
  the grids' difference (docs/validation.md).
- **The carriers' statistics**: `build_device(..., statistics=)` takes
  Blakemore's (`"blakemore"`) or Fermi–Dirac's `F_{1/2}` (`"fermi-dirac"`, TeSCA's
  approximation) beside Boltzmann's, the default and SIMsalabim's (`jaxdd.statistics`): the fluxes
  follow the electrochemical potential with the excess chemical potential beside the potential,
  the contacts' densities and `sfb` follow the statistics, and the recombination and the traps
  compare with the equilibrium at the point's own quasi-Fermi levels (docs/model.md). Against
  ChargeTransport.jl with degenerate transport layers to 2 % in the densities, 0.5 mV and 1.3e-3
  in the current, and in time (`solve_transient`) after a voltage's step to 4.5e-3 in the
  current. Interface traps and free surfaces compare with each point's own equilibrium, so a
  degenerate layer's recombine nothing at a flat Fermi level. A steady state with `F_{1/2}`
  compiles in a sixth of the time it took (a contact's Newton on its density as a loop, the
  assembly's excess potentials once). On the graded grid the contacts' spacing then starts from
  the contact layer's Debye length at `N_c` (it was 1 pm): a blocking contact's accumulation
  converges to −0.5 V where it stalled beyond −0.2 V. Boltzmann's grid is unchanged.

### Fixed

- **Blocking contacts under bias.** Each voltage of a scan starts in the contacts' branches of
  the last converged solution, not in those its start shows (with `S` = 0 the start could block
  a carrier at both ends and Newton left the solution); `solve_jv` and `solve_state` accept a
  step at the problem's own round-off floor where an accumulation layer puts it above 1e-6
  (docs/solver.md, "A floor above 1e-6").
- **A batch without a pre-bias** no longer fails its whole curve where the first solve, at the
  lowest voltage, fails: only frozen ions make that solve the scan's.
- **The start of a solve with ions bounded by their sites** (`solve_states`, `solve_batch`,
  `solve_transient`'s first step, now one code, `batch._start`) takes a bounded solution that
  converges from Boltzmann's iterate whether or not Boltzmann's converged, else solves again from
  Poisson's with the bounded ions.
- **Many ions.** Where the mobile ions far outnumber the doping Newton cycled from Poisson's
  start (ChargeTransport.jl's Ex103 cell, 1.6e25 m⁻³ of vacancies: no steady state even in
  equilibrium; with 4e24 a cold start below −0.2 V, and on a GPU the start in about one run of
  two). A start that does not converge at once now goes through a share of the ions, 1/64, 1/16,
  1/4 and all of them (`ION_DENSITY_STAGES`); the others skip the stages, and a batch that needs
  none pays what it paid before (docs/solver.md, "Many ions"). Ex103's cell agrees with
  ChargeTransport.jl in equilibrium, dark, lit and in time.

### Changed

- The tests take on a GPU at most a worker per 2 GB of its memory (`tests/conftest.py`).

## [0.3.0] – 2026-11-19

Films and half-stacks without electrodes (free surfaces), in the steady state and in time, each
trap level's own capture coefficients, photon recycling and each step's profiles in time. Nothing
of 0.2.0's API is removed. Results change only where a contact with finite surface recombination
sat at its branches' tie (such states now converge) and where a trap level is nearly full over a
time step (now to its tolerance).

### Added

- **Free surfaces** (jaxdd's; SIMsalabim fixes the potential at both ends). `W_L = free` or
  `W_R = free` ends the device without an electrode: no current and no field beyond it, the
  surface recombining by SRH at that end's `S_n`, `S_p`. One end free (a film on a contact) or
  both (a film on glass, neutral as a whole). `solve_state` gives one steady state with its
  quasi-Fermi level splitting, the implied `V_oc` and the PLQY of direct recombination;
  `batch.solve_states` the same for a batch, free ends beside contacted devices, differentiable
  in `device_tree`'s parameters; both with mobile ions and their profiles (`anion`,
  `cation`), and with the rates in the field (Braun's generation, Langevin's in a
  field-dependent mobility), and in time (`solve_transient`, e.g. a photoluminescence decay).
  A device with a free end discretises Poisson's equation in flux form and counts its mobile
  ions over the control volumes, so that it keeps its charge exactly ([docs/model.md](docs/model.md#free-surfaces-jaxdds-w_l-or-w_r--free)).
- **Profiles in time.** `solve_transient(..., profiles=True)` returns each step's potential and
  densities on every point (`TransientResult.profiles`), the core of ZimT's `varFile`, for band
  diagrams through a transient. Without it a transient is the same to the bit.
- **Photon recycling in time** (jaxdd's; SIMsalabim counts direct recombination as lost).
  `jaxdd.recycling.slab_kernel` gives, for an absorbing slab between two media, the share of
  each point's emitted photons absorbed in each control volume and the shares leaving at
  either face (Fresnel, total reflection beyond the escape cones, van Roosbroeck–Shockley's
  spectrum from an absorption edge, e.g. `urbach_alpha`); `device_kernel` the matrix on a
  device's grid. `solve_transient(..., recycle=K)` adds the reabsorbed photons to each step's
  generation, from the state before the step. Without it a transient is the same to the bit
  ([docs/model.md](docs/model.md#photon-recycling-jaxdds-in-time)).
- **Each trap level's own capture coefficients** (jaxdd's). A trap file whose header reads
  `E Ntrap C_n C_p` gives each level its own; SIMsalabim's levels share their layer's, so a
  shallow electron trap (`C_p` = 0) could not sit beside a deep recombination level. Without the
  columns a device is the same to the bit. In the steady state, the batch and in time
  ([docs/model.md](docs/model.md#bulk-traps-init_trapping-calc_trap_filling_charge)).

### Fixed

- **A contact's branch at a tie.** With finite surface recombination a contact's branch
  (extraction or injection) was decided by the sign of a difference that is round-off where
  the two coincide, at equilibrium: it flipped at every check, and Newton never converged (a
  blocking contact, `S` = 0, not even in the dark at 0 V). Within round-off (1e-13) the branch held
  now stays (`solve_jv`, `solve_state`, the batch).
- **Traps nearly full, in time.** Over a time step a level's empty share was `1 − f̄`, which
  loses every digit of a level nearly full: its capture `n·(1 − f̄) − n_t0·f̄` became round-off
  times `n`, and a step's Newton (the transient's first, a steady state, too) could stall
  short of its tolerance, e.g. at a heavily doped transport layer's interface traps
  (1e25 m⁻³). The empty share is now its own, as the steady state has it, bulk and interface.

## [0.2.0] – 2026-11-15

Tables with refined axes, ions bounded by their sites, the current in time as the equations
balance it, and the QFLS below Newton's floor. Nothing of 0.1.0's API is removed; transients
with the integral current change by its grid error (below).

### Added

- **Tables with refined axes** (`jaxdd.axes`). `refine_axes` chooses the points of a
  tensor-product table's axes by the multilinear interpolation's error, estimated in each cell
  from the table's own second differences, and solves only the table's points; `tabulate`,
  `interpolate` and `in_chunks` build and read such tables. On a perovskite cell's table over
  four axes it meets a tolerance of 5e-2 in `η` with 50 274 points where evenly spaced axes need
  83 521; where a feature moves across the axes with the voltage it saves no more than that
  ([docs/validation.md](docs/validation.md#a-table-with-refined-axes),
  [docs/tutorial.md](docs/tutorial.md#a-table-over-parameters)).
- **Ions bounded by their sites** (jaxdd's, off by default). A layer's `N_max_anion` and
  `N_max_cation` (m⁻³) bound each mobile ion species by its density of lattice sites, the
  occupancy of a site in place of Boltzmann's unbounded density (Bikerman's modified
  Poisson–Boltzmann), in the free steady state and so in the frozen profile; differentiable in
  them. Without them the curves are Boltzmann's to the bit. Transients with bounded mobile ions
  are refused ([docs/model.md](docs/model.md#ions-bounded-by-their-sites-jaxdds-off-by-default),
  [docs/differences.md](docs/differences.md) §14).

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

[Unreleased]: https://github.com/trn84/jaxdd/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/trn84/jaxdd/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/trn84/jaxdd/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/trn84/jaxdd/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/trn84/jaxdd/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/trn84/jaxdd/releases/tag/v0.1.0
