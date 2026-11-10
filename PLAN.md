# Plan

What is planned for jaxdd, and what may come after. What exists and how it works is in the
handbook, [docs/](docs/README.md). Finished work leaves this file; what it taught goes into
the handbook.

## 1 Open now

### 1.0 Bug: the scan's start extrapolates noise (dark runs lost from short circuit on)

**Found** 2026-11-10 on FauxPV's ion grid (layer axes to 1 µm, A100): 834 of about 3.5 million
scans are unconverged from −0.075…0 V to the last voltage. Always the dark run; every device
has a 1 µm transport layer doped 1e15 cm⁻³, a short diffusion length, ions frozen at flat band.

**Cause.** `batch._solve_one.scan` starts each voltage from the last two solutions extrapolated
linearly, `guess = y + ratio·(y − y_prev)` (SIMsalabim's `Extrapolate_Solution`). In the dark a
minority density far below `n_floor` is weighted out of the convergence test, so its logarithm
is undetermined: between two neighbouring converged voltages `ln p` differs by 30–100 at single
nodes. The extrapolation carries that difference into the start (e^100 off), Newton's capped
steps (5 per logarithm) do not bring it back, the state becomes NaN, and the scan carries the
NaN through every later voltage. Measured on 12 of 12 sampled devices: as it is, 53–56 of 61
voltages unconverged and ~3400 Newton steps; with the extrapolated change clipped to Newton's
own caps (10 `V_T`, 5, 5; the ion logs 5), 0 unconverged and ~190 steps; with the last solution
as the start, 0 and ~230; currents equal to 1e-9 of the generated current, and to 1e-10 where
the scan as it is still converged.

**Where the first difference comes from** (same device, the dark run, 2026-11-10). The first
scan voltage (−0.2 V) is declared converged while hole densities below `n_floor` still move:
40 more Newton steps there move `ln p` at hundreds of nodes. With the ions frozen at flat band
the pre-bias is `V_bi`, the longest way to the first voltage (1.25 V), and the leftover is
largest: at `ℓ` 0.5 a node had overshot to 4.5e-5 m⁻³ and comes back by 29 in the logarithm
(true value 1.4e8); at `ℓ` 0.2 and 1.0 on the same stack the leftover is 2.6 and 12.5, at a
pre-bias of 0.79 V 4.6, and none of those scans fails. So the ion distribution is systematic
(all 834 are at flat band); which diffusion length fails is the timing of when the densities
above the floor converge, not a threshold (0.2 and 0.5 fail on the grid, never 1 or more).
The clip below bounds what the extrapolation makes of a leftover; scanning from `V_pre`
outwards (1.2 a) would shorten the way that leaves it.

**Fix** (small; `batch.py` and the same place in `solver.py` if the single-device scan
extrapolates too):

1. Clip the extrapolated change to the Newton step's caps before adding it (`jnp.clip`), for
   `y` and for the ion logs. One named constant shared with `_newton_solve`'s `caps`.
2. Do not carry an unconverged solution: where a voltage did not converge (`ok` false) or the
   new state is not finite, keep the carry's last converged `y`, `logs` (`jnp.where`), so one
   failed voltage costs one voltage, not the rest of the scan.
3. Regression test in `tests/`: one of the failing devices as SimSS input files under
   `tests/data/` (as the three there; written by FauxPV's `write_point` (in its `tools/dd_tables.py`) for
   ion-grid index `(3, 1, 0, 0, 0, 0, 8, 1)`, the dark run: 1000 nm absorber, `ℓ` 0.5, ions
   1e15 at flat band, ETL 1 nm undoped, HTL 1000 nm on the axis at 1e15). Assert: every voltage
   converged, and a property, not only a value: the dark current is monotone in `V` and 0 at
   0 V within `CURRENT_TOL` of the scale.
4. Check that nothing else moved: `tests/` green; the benchmark and validation sets of
   `docs/validation.md` within their criteria; Newton steps per voltage on the benchmark
   batch not higher than before (the clip should lower them: note the number in
   `docs/batching.md`).
5. `docs/solver.md` (the start of a voltage; why a density below the floor has no determined
   logarithm) and `docs/differences.md` (jaxdd no longer extrapolates as SIMsalabim does where
   the change exceeds a Newton step). Then this section leaves the plan.

**Done when** the regression device converges at all 61 voltages, the existing tests and
validation hold, and the commit is named to FauxPV (its `docs/tasks/M20/M20-00f-dark-scan-start.md`
re-solves the 834 scans with it).

**Not here:** re-solving FauxPV's tables (FauxPV's task); the lockstep work of 1.2, although
(b) there is this start.

### 1.1 Validation at scale: a large grid with ions

A grid of 918 050 perovskite cells with mobile ions (four absorber thicknesses, diffusion
lengths, ion densities and distributions, light and dark: 1.84 M J-V curves) is solved on an
A100 in about two hours, and compared point by point with SimSS's run of the same grid.

**Done when** every point is within the criteria of [docs/validation.md](docs/validation.md), or
each miss is explained as SimSS's or jaxdd's, and SimSS's are in
[docs/differences.md](docs/differences.md).

### 1.2 Speed: the lockstep tail on real grids

Under `vmap` every voltage costs the batch's slowest device ([docs/batching.md](docs/batching.md)).
On a chunk of that grid (250 nm absorber, every diffusion length, ion density and ion
distribution, batch 16 384) the devices needed 2.97 Newton steps per voltage and the batch paid
25.5. Two passes (`max_iter` 25, then the ~0.15 % that miss it again with 60) brought it to 12.5:
a chunk 75 → 64 s. The rest is the tail of the *ordinary*
devices: at nearly every voltage some device of 16 384 needs about 12 steps.

Next: measure on thousands of devices, not hundreds, where those ~12 steps sit (voltage,
ion distribution, stack). By what that shows:

- (a) The scan from `V_pre` outwards instead of from −0.2 V: a device's hardest voltage is
  usually its first, up to 1.25 V from the frozen pre-bias it starts from. It would also let
  the first pass's limit drop to about 10.
- (b) A better start per voltage than linear extrapolation where the tail sits.
- (c) Smaller batches: the expected maximum over 4096 devices is lower, and an A100 runs 4096 at
  88 % of the rate of 16 384.
- (d) The host on smaller machines: writing and reading a device's input files costs ~90 ms
  on one core. `read_setup` from a mapping of file names to texts, so that sweeps build devices
  without files.
- (f) The two passes as part of jaxdd (`solve_batch` re-solving its stragglers), instead of
  each caller writing them.
- (e) The Jacobian by edges (each interval's flux and its derivatives once, not twice from both
  neighbouring rows) and in one `jacfwd` with the ion columns: estimated 1.5–2×. A Newton step
  of 16 384 devices on an A100 is 67 ms, 46 of it the Jacobian blocks and ion columns, 15 the
  sweep.

A fused kernel for the whole step stays the large lever (an estimated 15–30× from the A100's
bandwidth). It is not needed for the steady state; decide it with Stage C (section 4) and after
M7 (section 2).

### 1.3 M8: the MI50 cards

Prepared: JAX 0.4.31 in `rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10` runs `jax.jit` on
gfx906, and jaxdd passes its tests on JAX 0.4.31 ([docs/hardware.md](docs/hardware.md)). Waiting
for a run on the cards:

1. On the MI50 host: `rocm/mi50.sh`, then send `rocm/mi50_probe.log`.
2. The batch a 16 GB card holds; the four cards as four processes (`HIP_VISIBLE_DEVICES`).
3. A large sweep split across the four cards.

If the image fails: the ROCm 6.1.3 and 6.2.3 images with the same JAX; then jaxlib built from
source for gfx906 against a ROCm that still ships its libraries (6.2/6.3); last, the block sweep
as a hand-written HIP kernel beside JAX.

**Done when** the benchmark's results are reproduced on the cards within the validation
criteria, at a throughput worth the cards (four MI50 are, for this bandwidth-bound FP64 solver,
about two A100s).

## 2 Milestones after that

| | Work | Done when |
|---|---|---|
| **M4** | The grid in space (2.1) | 400 points or fewer meet the validation criteria on the hardest devices, or bucketing by grid size is justified |
| **M6** | Adaptive parameter axes (2.2) | the refined axes meet a table's interpolation tolerance with fewer devices than hand-made ones |
| **M7** | Gradients of the converged solution by implicit differentiation (`jax.custom_root` or a custom VJP: one transposed linear solve per voltage); `lax.while_loop` has no reverse mode | gradients against finite differences on the test devices; a fit of a measured curve |
| — | Several GPUs in one process: the batch axis sharded (`NamedSharding` or `shard_map`, not `pmap`); the devices are independent, so no communication | the rate scales with the number of GPUs |

M7 comes before any fused kernel: written afterwards, the gradients would have to be written
twice (the kernel needs its own adjoint, the transposed block sweep).

### 2.1 M4: the grid in space

JAX compiles for static shapes: every device in a batch has the same number of grid points, but
the node **positions** may differ per device. Three steps, cheapest first:

1. **A priori graded mesh per device.** Geometric refinement towards every interface and
   contact, scaled by the local Debye length `L_D = √(ε·V_T/(q·N))` of dopants and ions. At
   1e18 cm⁻³ of ions `L_D` is about 6 nm (ε 24.1), and strongly screening ions are where
   interpolated tables are worst.
2. **r-adaptivity (moving nodes, fixed count).** After a first solve, move the nodes to
   equidistribute a monitor function (the arc length of `V`, `ln n`, `ln p`) and solve again.
   The count stays fixed, so it fits static shapes and batching.
3. **h-adaptivity (changing count)** only by bucketing: batches compiled at 200, 400, 800
   points, each device in the smallest bucket that meets its error estimate. Only if 1 and 2
   are not enough.

The study: `J(V)` and `η` at 200, 400, 800, 1600 points, uniform against graded against
r-adaptive, on the hardest devices known (`L_diff/d` 0.1–0.2, ions at 1e18 cm⁻³, thick undoped
layers).

### 2.2 M6: fewer devices instead of faster ones

Sample the parameter space where the result changes, instead of on a full tensor grid. A fast
solver makes this loop interactive.

- **Axis refinement (keeps an interpolated table).** A table interpolated on a regular grid
  stays a tensor product, but each axis's points are chosen by error: solve at the midpoints
  between neighbouring axis points across a sample of the other axes, and insert a point where
  the interpolation misses by more than the tolerance. This automates what is otherwise done
  by hand, axis by axis.
- **Sparse grids** (Smolyak) or **scattered samples with a surrogate** (a GP or a small network).
  Far fewer devices in 7–9 dimensions, at the price of a model evaluation instead of an
  interpolation.

Either needs the solver callable from Python in a loop at low latency: a requirement on the API,
not only on throughput.

## 3 Proposal: a steric limit on ion accumulation

**Why.** Frozen at full accumulation, 1e18 cm⁻³ of ions pile up in nanometre layers at the
interfaces at 1.5e21 cm⁻³, 1500 times their mean and close to the perovskite's lattice-site
density (~1e22 cm⁻³). Boltzmann statistics let ions stack without limit; real vacancies cannot.
The carrier-free bulk between such layers is what leaves the solver at its round-off floor, in
SimSS as in jaxdd (the hard case in [docs/tutorial.md](docs/tutorial.md), "Known limits").

**What.** The modified Poisson–Boltzmann (Bikerman) occupancy for the ions: saturation at a site
density `N_max`, in the free steady state and so in the frozen profile. Physically the better
model; it caps the peaks and removes the near-singular regime at its source. It departs from
SimSS (to be documented in docs/differences.md, switchable, SimSS's Boltzmann the reproduction),
and it changes what an ion density means at the high end, so results computed with and without
it must not be mixed. `N_max` needs a source (a literature value for the halide vacancy site
density; open). Not started.

## 4 The physics in stages

What jaxdd solves today ("Stage A") is in [docs/overview.md](docs/overview.md). After it:

- **Stage B: SimSS's steady state in full.** One module at a time, each validated against SimSS
  on its own test devices before the next: charged traps at several levels in the bulk and at
  interfaces; Schottky contacts and finite surface recombination; series and shunt resistance;
  the remaining recombination and mobility options. At the end, a SimSS input runs on jaxdd or is
  refused with the missing feature named.
- **Stage C: transients (ZimT).** Time-dependent drift-diffusion with mobile ions: scan-rate
  hysteresis, transient photocurrent and photovoltage, impedance. Its own plan when Stage B is
  done. This is where a fused kernel is most likely needed (section 1.2).

Not planned: a line-by-line port of SIMsalabim (its one-device Gummel structure does not batch),
a GUI. SimSS stays the reference jaxdd is validated against.

## 5 Applications: what the speed could be for

None of these is planned yet. They are the case for or against a fused kernel and for M7's
gradients; each says how much speed it needs.

| | Curves/s | Per hour | 1024 curves |
|---|---|---|---|
| A100 today (batch 16 384) | ~1200 | 4.3 M | 0.85 s |
| A100 with a fused kernel (estimate, uncertain by 3×) | 10⁴–3·10⁴ | 36–110 M | 35–100 ms |
| a 16-core desktop CPU today | ~360 | 1.3 M | 2.9 s |
| one CPU core today | ~20 | 72 k | 50 s |
| a neural surrogate, evaluated | ~10⁶ | — | ~1 ms |

1. **A surrogate trained on data simulated during training.** Simulator and network are both
   JAX, so they can run in one process on one GPU, with no dataset written or stored.
   Simulation dominates a training step by about 100×; a replay buffer (each curve serves
   several steps) makes it work today. It pays most with active learning: simulate where the
   surrogate is still wrong, in spaces too large for a grid (a 16-dimensional space of process
   recipes, layer stacks with many parameters).
   Training must mask by the `converged` flag, and active learning steers towards the hard
   devices, which the lockstep makes slow. For a fixed dataset today's speed is enough: 10 M
   curves in about 2.3 h on one A100.
2. **Parameters from measurements, with uncertainty.** MCMC needs 10⁵–10⁶ simulations per cell,
   1.4–14 min on one A100. Ensemble samplers and sequential Monte Carlo map onto the batch;
   gradient-based samplers and optimisers need M7.
3. **Sensitivity analysis and uncertainty propagation.** Sobol indices by Saltelli sampling
   (N·(d+2) simulations: 1.8·10⁵ for N = 10⁴, d = 16, under 3 min) and Monte Carlo over process
   spread: independent devices, exactly the batch.
4. **Inverse design.** Optimise a stack for efficiency or stability, by gradient (M7), from many
   starting points in parallel.
5. **Drift-diffusion inside a larger simulation.** One solve per simulated measurement, for
   example in a digital twin of a laboratory that ages and measures many cells; mostly
   transient (Stage C).
6. **Transients** cost 10–100× a steady-state curve.
7. **Interactive tools.** A page where one slides a parameter and watches the curve: about 50 ms
   per curve on one CPU core is already enough. The GPU earns its place when each update needs
   many curves (a fit, an uncertainty band).
8. **Two dimensions (an outlook, a project of its own).** Grain boundaries, pinholes, patterned
   or interdigitated contacts, edges; Monte Carlo over random grain structures condensed into
   effective 1D parameters. What changes: about 10⁵ unknowns per device instead of 1200, so the
   parallelism moves into the device and batches shrink to tens; the Newton system is sparse
   and banded, solved either by a sweep over grid columns with dense (3·n_y)² blocks (batched
   LU on FP64 tensor cores, up to n_y ≈ 100–200) or matrix-free Newton–Krylov (JVPs from
   `jax.jvp`, GMRES; the preconditioner is the hard part, and JAX has no algebraic multigrid or
   sparse direct solver on the GPU); structured grids suit JAX, unstructured meshes are much
   more work. The physics and the Jacobian-by-autodiff carry over; the linear solver, the
   batching, the mesh and the reference do not (SIMsalabim is 1D only; references would be
   Sentaurus, ATLAS, COMSOL or ChargeTransport.jl).

**The counter-argument.** The faster the simulator, the less a surrogate is needed: fits and
sensitivity analyses can call jaxdd directly, with the exact physics. A surrogate pays where
microseconds per evaluation are needed.

| Use | Today enough? | Fused kernel |
|---|---|---|
| tables, a fixed training set (1) | yes | not needed |
| on-the-fly training, active learning (1) | yes, with a replay buffer | makes it nearly free |
| MCMC per cell (2) | yes for a few cells | helps for many cells |
| gradients: HMC, fits, inverse design (2, 4) | after M7 | needs its own adjoint |
| Sobol, Monte Carlo (3) | yes | not needed |
| inside other simulations, transients (5, 6) | doubtful | most likely needed |
| interactive (7) | yes, even on a CPU | not needed |
| 2D (8) | a different solver | dense batched LU or a preconditioned Krylov method |

## 6 Before a release

- `NOTICE` lists the modules derived from SIMsalabim code (none so far): keep it current, and
  where code follows SIMsalabim's, keep its copyright notice with a note that it was translated
  and changed, with the date (GPL-3.0 §5 a).
- A word with SIMsalabim's maintainers (who also maintain pySIMsalabim): a GPU backend may be
  of use to them, at best a joint effort.
- Citation of SIMsalabim in the README and with every published result.
- A package release on PyPI, versioned, with a changelog from then on.

## 7 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The slowest lane | one stubborn device sets a batch's pace | section 1.2; difficulty-sorted batches; two passes |
| Convergence on new physics (Stage B, steric ions) | the hard devices fail first | each module validated on exactly those devices; flag, never guess |
| ROCm on gfx906 | the MI50 may never run it | not on the critical path; three ways in (1.3), the last independent of XLA's ROCm support |
| Scope creep towards a full SIMsalabim | months | every addition its own decision, module by module, each released when validated |

## 8 Decisions ahead

| Decision | When | With |
|---|---|---|
| A fused kernel for the Newton step | with Stage C's plan, after M7 | the applications' needs (section 5) |
| The steric ion model and its `N_max` | when a source for `N_max` is found | its users: results with and without it differ at high ion density |
