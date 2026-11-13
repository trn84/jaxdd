# Plan

What is planned for jaxdd, and what may come after. What exists and how it works is in the
handbook, [docs/](docs/README.md). Finished work leaves this file; what it taught goes into
the handbook.

## 1 Open now

### 1.2 Speed: the Newton step

The lockstep's tail is gone on the hardest grid known: on 16 384 devices of the 250 nm grid with
ions no voltage takes more than 12 Newton steps and none fails ([docs/batching.md](docs/batching.md)),
since the scan runs outwards from the pre-bias and the Jacobian is assembled by intervals
([docs/solver.md](docs/solver.md)); `solve_batch` makes two passes by default, and on an
NVIDIA GPU the block sweep is one Pallas kernel instead of a scan's ~1450 kernel launches
(`jaxdd.sweep`). On the RTX 4070 Ti SUPER those devices took 116 s with the Jacobian by rows
(two passes, limit 25) and take 24.5 s now; on an A100-SXM4-40GB 9.0 s (64 s before, on an
A100 80 GB PCIe; `validation/chunk.py`). On the A100 the benchmark runs at 2800
curves/s (batch 16 384; 2140 at 4096), the hard workload (`validation/hard.py`) at 1425 in one
pass and 1547 in two; a Newton step of 16 384 devices there is 20 ms: the assembly 9.1, the block sweep 7.2, the
current 2.7, the ion constraint 1.3 (`validation/breakdown.py`).

Next:

- The assembly is now the largest part (9.1 of 20 ms on the A100). Each interval's fluxes
  and each transcendental are computed once (the compiled program has 18 kernels for it, none
  repeating another's work); what is left is memory traffic: the blocks have 36 entries per
  grid point of which 24 are structurally non-zero (`A` and `C` 5 of 9 each, the right-hand
  sides 5 of 9), and the kernel reads all 36. A kernel that takes the 24, as separate arrays,
  would spare a third of the blocks' writes and reads; it is a step towards the fused kernel
  below and wants the same decision.

A fused kernel for the whole step stays the large lever (from the A100's bandwidth an estimated
10⁴–3·10⁴ curves/s, 4–10× today's), as an optional mode beside the JAX solver (section 2.3).
It is not needed for the steady state; decide it with Stage C (section 4).

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
| **M4** | The grid in space (2.1) | 400 points or fewer meet the validation criteria on the hardest devices, or bucketing by grid size is justified |
| **M6** | Adaptive parameter axes (2.2) | the refined axes meet a table's interpolation tolerance with fewer devices than hand-made ones |
| **M7** | A fit of a measured curve by gradients (2.4); the curves are differentiable in the input's parameters ([docs/solver.md](docs/solver.md#derivatives)) | a SimSS curve's parameters recovered from a perturbed start; a measured curve fitted |
| — | Several GPUs in one process: the batch axis sharded (`NamedSharding` or `shard_map`, not `pmap`); the devices are independent, so no communication | the rate scales with the number of GPUs |

Gradients are taken by implicit differentiation of the converged solution: the backward pass
needs only the solution and the JAX residual, never the iterations that found it, so any forward
solver is a black box to them, and a fused kernel (section 2.3) needs no adjoint of its own.

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

### 2.3 The fused kernel: an optional fast mode

The whole Newton step, or the whole scan of a device, in one hand-written CUDA kernel called
through the XLA FFI: one thread per device, the blocks assembled in registers where the sweep
uses them instead of written to memory and read back. It is a mode beside the JAX solver, not a
second solver: chosen like `JAXDD_SWEEP` today, with the JAX solver as reference and fallback.

What it would solve:

- **Memory traffic**, today's limit (36 block entries per grid point written by the assembly and
  read by the sweep): the 4–10× estimated in section 1.2.
- **The lockstep**, mostly: each thread iterates its device to convergence on its own, so a
  device waits for the slowest of its warp (32), not of the batch. On the 250 nm ion grid the
  batch pays 4.59 Newton steps per voltage where 2.76 are needed.
- **Pallas's backends**: no dependence on Triton's future or Mosaic GPU's warpgroups, tiles of
  any size.
- **The AMD track**: CUDA C++ ports to HIP (hipify) nearly as it is.

What is written once: reading the input, building devices and grids, `prepare`, the scan's
order and its two passes, batching and flags, the gradients (M7), the CPU path, validation and
the handbook. What is written twice: the solver's core (`solver.py`'s residual rows, ion
constraint, Newton step with its damping, sweep and current, some 300–400 lines of JAX), in CUDA
with the Jacobian by hand where JAX takes it by autodiff; roughly 800–1200 lines.

How the doubled part stays small:

1. **The kernel covers a fixed subset of the physics** (Stage A first). A device that needs
   more goes to the JAX solver automatically, as `check_supported` refuses what jaxdd lacks.
   New physics (Stage B) is written in JAX only, and enters the kernel when it has settled and
   is worth it, or never.
2. **JAX is the reference**: a test solves every test device both ways and asks for the same
   curves, flags and Newton steps to round-off (as the sweep kernel's test does today).
3. **The hand-written Jacobian is tested against `jax.jacfwd`** of the JAX residual, as the
   assembly by intervals is tested against the one by rows.

What it costs: the hand-written Jacobian; compiled wheels per CUDA (and ROCm) version, where
jaxdd is pure Python today; and little gain on consumer cards, whose FP64 rate (1/64 on the
RTX 4070 Ti SUPER) becomes the limit once memory traffic is gone. On the A100, H100 and MI50
FP64 is fast.

Order: Stage B in JAX; then the kernel for the physics that has
settled. A first measurement before deciding: the Newton step of Stage A's physics without ions
as one CUDA kernel on devices without ions, against the JAX step's time on the A100.

### 2.4 M7: a fit

The curves are differentiable in the input's own parameters (a layer's mobility, thickness, band
edge, trap and ion density, the temperature, the work functions): the device built in JAX
(`batch.device_tree`), the converged solution differentiated implicitly
([docs/solver.md](docs/solver.md#derivatives)). What is left is to use it:

- **A fit.** Parameters of one device from a measured curve, by `jax.grad` and an optimiser
  (optax or `jax.scipy.optimize`), with bounds by transformation (logarithms of densities and
  mobilities); on a curve SimSS made from known parameters first (the test cases' `JV.dat`), then
  on a measured one. What it shows: which parameters a curve determines and which it does not
  (the Hessian's or the Jacobian's small singular values), and how a voltage that does not
  converge during a fit is handled (its zero gradient leaves it out of the step).

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
| gradients: HMC, fits, inverse design (2, 4) | yes (M7: a fit still to show) | faster forward solves; the gradients stay M7's |
| Sobol, Monte Carlo (3) | yes | not needed |
| inside other simulations, transients (5, 6) | doubtful | most likely needed |
| interactive (7) | yes, even on a CPU | not needed |
| 2D (8) | a different solver | dense batched LU or a preconditioned Krylov method |

## 6 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| The slowest lane | one stubborn device sets a batch's pace | section 1.2; difficulty-sorted batches; two passes |
| Convergence on new physics (Stage B, steric ions) | the hard devices fail first | each module validated on exactly those devices; flag, never guess |
| ROCm on gfx906 | the MI50 may never run it | not on the critical path; three ways in (1.3), the last independent of XLA's ROCm support |
| Scope creep towards a full SIMsalabim | months | every addition its own decision, module by module, each released when validated |
| The fused kernel and the JAX solver drift apart | two answers to one device | the kernel a fixed subset with fallback to JAX; both compared on every test device (2.3) |

## 7 Decisions ahead

| Decision | When | With |
|---|---|---|
| The fused kernel as an optional mode (2.3) | with Stage C's plan | the applications' needs (section 5) and a first measurement of a Stage A kernel |
| The steric ion model and its `N_max` | when a source for `N_max` is found | its users: results with and without it differ at high ion density |
