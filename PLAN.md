# Plan

What is planned for jaxdd, and what may come after. What exists and how it works is in the
handbook, [docs/](docs/README.md). Finished work leaves this file; what it taught goes into
the handbook.

## 1 Open now

0.1.0 is released (CHANGELOG.md). The order now (2026-11-15, an estimate at the pace so far):

1. **The current in time: ZimT's integral form converges slowly.** Found 2026-11-15: the light
   with traps switched on and off (1.6e-3 of the scale off at 400 graded points) is not the
   traps' doing: without traps (`zimt_photo_light`) the error is the same to three digits. The
   solution converges; the current's estimate does not. The integral form (`currDiffInt = 2`,
   ZimT's default) converges at about first order in time, the differential form (`= 1`) of the
   same solution at the steady state's rate. Against the differential form at 6400 graded
   points, at 400 / 1600: the light on and off (with and without traps) integral 1.6–2.1e-3 /
   5–6.5e-4, differential ~5e-5 / ~2e-6; with surface recombination 3.0e-2 / 7.8e-3 against
   4e-5 / 2.4e-6; the sweep without ions 3.9e-3 / 7.9e-4 against 9e-6 / 6e-7; the photovoltage
   at the open circuit (through `Jext = 0`) 1.8e-3 / 3.5e-4 of it against 1e-5 / 5e-7. At 1 ps
   after the switch, where the current comes from the carriers within √(D·Δt) ≈ 5 nm of the
   interfaces, the integral form at 6400 points is still 2 % off that step's current. With
   mobile ions both forms are the same (6e-4 at 400: the ions' own spatial error); in the steady
   state the two forms agree to 1e-9 at 6400 points and converge alike. Earlier references (the
   integral form at 3200 graded points) were themselves ~3e-4 of the scale off.
   Next: why the integral form is first order in time (a guess: its trapezoids of
   `dti·(n − n_prev)`, which varies exponentially across the interfaces' layers, against the
   residual's control volumes), and whether a quadrature consistent with the residual converges
   as the differential form does. Then the default in time, by the rule for SimSS's errors: the
   accurate current by default, ZimT's integral form a switch, docs/differences.md; the
   validation's transient numbers again against the differential form.
2. **The QFLS below Newton's floor** (4.1).
3. M6 (2.2): 0.2.

The fused kernel is decided against for now (2.3). Waiting: M8 (1.3), the measured cell's
optics (2.4, waiting for n,k data), the steric ions (3, waiting for a source of `N_max`).

### 1.3 M8: the MI50 cards

Not the focus: jaxdd is made as fast as it can be on NVIDIA first (Mosaic GPU and other
NVIDIA-only kernels are fine there); an AMD track with its own optimisations follows later,
beside it, starting from the scan, which runs everywhere.

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
| **M6** | Adaptive parameter axes (2.2) | the refined axes meet a table's interpolation tolerance with fewer devices than hand-made ones |

Gradients are taken by implicit differentiation of the converged solution: the backward pass
needs only the solution and the JAX residual, never the iterations that found it, so any forward
solver is a black box to them, and a fused kernel (section 2.3) needs no adjoint of its own.

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

### 2.3 The fused kernel: decided against for now (2026-11-15)

The idea: the whole Newton step in one hand-written CUDA kernel called through the XLA FFI, one
thread per device, the blocks assembled in registers where the sweep uses them instead of
written to memory and read back; a mode beside the JAX solver (as `JAXDD_SWEEP`), the JAX solver
its reference and fallback, the gradients unchanged (implicit differentiation needs no adjoint
of it). From the A100's bandwidth an estimated 10⁴–3·10⁴ curves/s, 4–10× today's 2800, and the
lockstep per warp of 32 instead of per batch (on the 250 nm ion grid the batch pays 4.59 Newton
steps per voltage where 2.76 are needed).

Not built, because:

- **No planned use needs it.** Of section 5's applications only drift-diffusion inside a larger
  simulation would, and none is planned.
- **The transients with ions were its strongest case and are no longer one.** With the 5×5
  sweep kernel a Newton step with ions is 2.3× a 3×3 one (it was 4.5×), close to the ratio of
  their blocks' entries (25/9): 185 transients/s with ions on the A100. What is left is the
  memory traffic every workload shares.
- **Its cost.** The solver's core written twice (800–1200 lines of CUDA, the Jacobian by hand
  where JAX takes it by autodiff), compiled wheels per CUDA (and ROCm) version where 0.1.0 is
  one pure-Python wheel, and little gain on consumer cards, whose FP64 rate becomes the limit.

Reopened when an application needs 10⁴ curves/s or more, or transients inside another
simulation; then first measured, as planned before: the Newton step of Stage A's physics
without ions as one kernel against the JAX step on the A100. Cheaper steps come before it, each
measured on the A100:

1. **The blocks' structural zeros.** 24 of their 36 entries per grid point are non-zero (`A`
   and `C` 5 of 9 each, the right-hand sides 5 of 9); a sweep kernel that reads the 24, as
   separate arrays, spares a third of the blocks' writes and reads. The assembly is 9.1 of the
   20 ms of a Newton step at 16 384 devices.
2. **The assembly inside the Pallas sweep kernel** (untried): the blocks never in memory, the
   residual still the JAX one traced into the kernel (its derivatives by `jax.jvp`, if Pallas
   lowers them), no compiled wheels.

### 2.4 What fits still need

The curves, steady and in time, are differentiable in the input's parameters; fits, many starts
at once and the posteriors of the measured samples are in docs/validation.md. Left:

- **The optics for the measured cell**: the measured fits assume a uniform generation. The cell
  in its own orientation with the transfer-matrix profile is `tests/data/pin_anode_left_light`,
  but the paper's perovskite n,k data (Werner et al., ACS Energy Lett. 3, 742 (2018)) are not
  among SIMsalabim's files; a stand-in's profile is a different assumption, not a better one.

## 3 Proposal: a steric limit on ion accumulation

**Why.** Frozen at full accumulation, 1e18 cm⁻³ of ions pile up in nanometre layers at the
interfaces at 1.5e21 cm⁻³, 1500 times their mean and close to the perovskite's lattice-site
density (~1e22 cm⁻³). Boltzmann statistics let ions stack without limit; real vacancies cannot.
The carrier-free bulk between such layers is where SimSS skips a third of the voltages (jaxdd
converges there since its Jacobian is assembled by intervals; [docs/tutorial.md](docs/tutorial.md),
"Known limits").

**What.** The modified Poisson–Boltzmann (Bikerman) occupancy for the ions: saturation at a site
density `N_max`, in the free steady state and so in the frozen profile. Physically the better
model; it caps the peaks. It departs from
SimSS (to be documented in docs/differences.md, switchable, SimSS's Boltzmann the reproduction),
and it changes what an ion density means at the high end, so results computed with and without
it must not be mixed. `N_max` needs a source (a literature value for the halide vacancy site
density; open). Not started.

## 4 The physics in stages

SimSS 5.36's steady state (Stages A and B) and ZimT 5.36's transients (Stage C) are done: an
input runs on jaxdd or is refused with the reason named, and what jaxdd solves is in
[docs/overview.md](docs/overview.md); the two inputs it refuses are SimSS's ionic regions that
SimSS solves against what they say ([docs/differences.md](docs/differences.md) §10).

### 4.1 Stage C: transients, as ZimT 5.36 solves them

Done: ZimT 5.36's transients reproduced, batched and differentiable (docs/model.md
"Transients", docs/validation.md, docs/hardware.md for the speed). Open: the QFLS where a
carrier lies below Newton's floor of convergence, in the dark and in an ionic region's
depletion zones (up to 5e-3 V and 3 V off ZimT's): a floor, or a last step for the output,
that converges those densities without costing the ions' convergence (a lower floor alone
converges them only without ions).

Where ZimT turns out wrong (the rescaling against conservation first), it is documented in
docs/differences.md and the physically correct answer is a switch, as for SimSS.

Later, not in Stage C: impedance in the frequency domain, from the steady state's Jacobian
(`(J + iω·M)·δx = b`, one complex solve per frequency) instead of a transient per frequency;
ZimT has no reference for it but its own time-domain response.

Not planned: a line-by-line port of SIMsalabim (its one-device Gummel structure does not batch),
a GUI. SimSS stays the reference jaxdd is validated against.

## 5 Applications: what the speed could be for

None of these is planned yet. They are the case for or against a fused kernel and for M7's
gradients; each says how much speed it needs.

| | Curves/s | Per hour | 1024 curves |
|---|---|---|---|
| A100 today (40 GB SXM4, batch 16 384) | ~2800 | 10 M | 0.37 s |
| A100 with a fused kernel (estimate, uncertain by 3×) | 10⁴–3·10⁴ | 36–110 M | 35–100 ms |
| a 16-core desktop CPU today | ~730 | 2.6 M | 1.4 s |
| one CPU core today | ~58 | 210 k | 18 s |
| a neural surrogate, evaluated | ~10⁶ | — | ~1 ms |

1. **A surrogate trained on data simulated during training.** Simulator and network are both
   JAX, so they can run in one process on one GPU, with no dataset written or stored.
   Simulation dominates a training step by about 100×; a replay buffer (each curve serves
   several steps) makes it work today. It pays most with active learning: simulate where the
   surrogate is still wrong, in spaces too large for a grid (a 16-dimensional space of process
   recipes, layer stacks with many parameters).
   Training must mask by the `converged` flag, and active learning steers towards the hard
   devices, which the lockstep makes slow. For a fixed dataset today's speed is enough: 10 M
   curves in about 1 h on one A100.
2. **Parameters from measurements, with uncertainty.** MCMC needs 10⁵–10⁶ simulations per cell,
   0.6–6 min on one A100. Ensemble samplers and sequential Monte Carlo map onto the batch;
   gradient-based samplers and optimisers need M7.
3. **Sensitivity analysis and uncertainty propagation.** Sobol indices by Saltelli sampling
   (N·(d+2) simulations: 1.8·10⁵ for N = 10⁴, d = 16, about a minute) and Monte Carlo over process
   spread: independent devices, exactly the batch.
4. **Inverse design.** Optimise a stack for efficiency or stability, by gradient (M7), from many
   starting points in parallel. Its thicknesses need the computed optics (Stage B); with a profile
   from a file only the electrical parameters are worth optimising.
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
| gradients: HMC, fits, inverse design (2, 4) | yes (fits of SimSS's and measured curves: [docs/validation.md](docs/validation.md)) | faster forward solves; the gradients stay M7's |
| Sobol, Monte Carlo (3) | yes | not needed |
| inside other simulations, transients (5, 6) | doubtful | most likely needed |
| interactive (7) | yes, even on a CPU | not needed |
| 2D (8) | a different solver | dense batched LU or a preconditioned Krylov method |

## 6 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The slowest lane | one stubborn device sets a batch's pace | two passes; difficulty-sorted batches; a fused kernel, should it be reopened (2.3) |
| Convergence on new physics (Stage B, steric ions) | the hard devices fail first | each module validated on exactly those devices; flag, never guess |
| ROCm on gfx906 | the MI50 may never run it | not on the critical path; three ways in (1.3), the last independent of XLA's ROCm support |
| Scope creep towards a full SIMsalabim | months | every addition its own decision, module by module, each released when validated |

## 7 Decisions ahead

| Decision | When | With |
|---|---|---|
| The steric ion model and its `N_max` | when a source for `N_max` is found | its users: results with and without it differ at high ion density |
