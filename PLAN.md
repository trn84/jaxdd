# Plan

What is planned for jaxdd, and what may come after. What exists and how it works is in the
handbook, [docs/](docs/README.md). Finished work leaves this file; what it taught goes into
the handbook.

## 1 Open now

The order (2026-11-15, an estimate at the pace so far):

1. Stage C to its end (section 4.1, steps 1–2): about 2 days.
2. M4's first step, the graded mesh (2.1): 1–3 days, depending on whether it is enough.
3. **Release 0.1.0**: the steady state (SimSS) and the transients (ZimT); a version, a
   changelog, the README's installation, a wheel, CI, a pass over the handbook and the API.
   About a day.
4. M6 (2.2), and the fused kernel if Stage C's speed decides for it (2.3, about 1–2 weeks):
   0.2.

On the hub: 0.2 Cl's second sampling run, which closes M7 (2.4). Not before the release: M8 (1.3), the measured cell's optics (2.4, waiting for n,k
data), the steric ions (3, waiting for a source of `N_max`).

### 1.1 Stage C: transients

Begun: electrons, holes, traps and mobile ions in time, with the voltage given, at the open
circuit and behind a series resistance, with ZimT's automatic stop, failure modes and output
columns, reproduce ZimT 5.36. The next steps (gradients, speed) are in section 4.1.

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
| **M7** | What fits still need (2.4): a second sampling run of 0.2 Cl; the curves are differentiable in the input's parameters, measured curves are fitted from many starts at once (`validation/fit_measured.py --starts N`, `jaxdd.fit`), and the posteriors of 0.1 and 0.6 Cl sampled around their best minimum by two independent runs that agree (`validation/sample_measured.py`) | every measured sample's posterior around the best of many starts, from two independent runs that agree with 1000 independent samples or more between them |

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
layers). Its baseline, SIMsalabim's own grid at 200 to 3200 points on `hard.py`'s
devices against the criteria, is `validation/grid.py` (`STEPS=grid benchmarks/apptainer.sh`),
run on one A100 (2026-11-15; a first run on two hung both GPUs at 1600 points, in the second
pass's scatter across the GPUs, since removed: 800 and 1600 points then ran on two). Against 3200 points, of 4096 hard devices in the light and the dark:

| `NP` | meet both criteria | η: median, 99 %, max | `J_SC`: median, 99 % |
|---|---|---|---|
| 200 | 5.5 % | 3.8e-3, 4.0e-2, 0.16 | 3.3e-3, 2.3e-2 |
| 400 | 32.4 % | 1.3e-3, 1.4e-2, 5.0e-2 | 1.2e-3, 9.4e-3 |
| 800 | 77.8 % | 4.7e-4, 5.7e-3, 1.7e-2 | 4.4e-4, 3.8e-3 |
| 1600 | 95.3 % | 1.4e-4, 2.0e-3, 5.6e-3 | 1.3e-4, 1.3e-3 |

So SIMsalabim's grid at its default 400 points misses η's 1e-3 on two thirds of these devices,
by up to 5e-2; the error falls by about 3 per doubling. Step 1 (the graded mesh) is the next
thing to try against these numbers.

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
  read by the sweep): from the A100's bandwidth an estimated 10⁴–3·10⁴ curves/s, 4–10× the
  2800 of today.
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

A smaller step on the way, with the same decision: the assembly is the largest part of the
Newton step (9.1 of 20 ms at 16 384 devices on the A100). Each interval's fluxes and each
transcendental are computed once already; what is left is memory traffic: the blocks have 36
entries per grid point of which 24 are structurally non-zero (`A` and `C` 5 of 9 each, the
right-hand sides 5 of 9), and the sweep kernel reads all 36. A kernel that takes the 24, as
separate arrays, would spare a third of the blocks' writes and reads.

Order: Stage B in JAX (done); then the kernel for the physics that has settled. A first
measurement before deciding: the Newton step of Stage A's physics without ions as one CUDA
kernel on devices without ions, against the JAX step's time on the A100. The fused kernel is not
needed for the steady state; it is decided with Stage C's speed (section 4.1, step 2).

### 2.4 M7: what fits still need

The curves are differentiable in the input's own parameters; a fit recovers a SimSS curve's
parameters (`validation/fit.py`) and fits the measured curves of Heester et al. (2025), five
intensities at once, to 0.55–2.7 % of the short-circuit current in minutes
(`validation/fit_measured.py`), from 16 starts at once in 10.5 min (`--starts`, `jaxdd.fit`),
and samples the posteriors around the best minimum, for 0.1 and 0.6 Cl by two independent runs
that agree (`validation/sample_measured.py`, [docs/validation.md](docs/validation.md)). Left:

- **A second run of 0.2 Cl's posterior.** 0.1 and 0.6 Cl are sampled around their best minimum
  by two runs that agree (docs/validation.md); 0.2 Cl, whose best minimum is the cold fit's,
  by one only (A100, seed 0, `benchmarks/samples/chain_02Cl.npz`, not committed). Its second,
  on the A100 hub: `STEPS=sample SAMPLES=02Cl SAMPLE_ARGS="--walkers 128 --steps 3000 --seed 1"
  benchmarks/apptainer.sh` (about 80 min; its chain back as `chain_02Cl_best_s1.npz`), compared with the first as `benchmarks/samples/compare_seeds.py` compares
  0.1 and 0.6 Cl's (it reads `chain_{SAMPLE}_best_s{0,1}.npz`: link or adapt), and the 0.2 Cl
  column and the runs' sentence in docs/validation.md from both. M7 is then done.
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

SimSS 5.36's steady state (Stages A and B) is done: an input runs on jaxdd or is refused with
the reason named, and what jaxdd solves is in [docs/overview.md](docs/overview.md); the two
inputs it refuses are SimSS's ionic regions that SimSS solves against what they say
([docs/differences.md](docs/differences.md) §10). After it:

- **Stage C: transients (ZimT).** Time-dependent drift-diffusion with mobile ions: scan-rate
  hysteresis, transient photocurrent and photovoltage, trap filling. Its plan is section 4.1.
  This is where a fused kernel is most likely needed (section 2.3).

### 4.1 Stage C: transients, as ZimT 5.36 solves them

**What ZimT is** (`ZimT/zimt.pas`, 600 lines, on `DDRoutines.pas`, which SimSS shares). It
reads a list of `t Vext G_frac Track` from `tVGFile`; the first line must be `t = 0`, a steady
state. At every line it solves one time step with the same `Main_Solver` as SimSS, with
`dti = 1/Δt`:

- **Electrons, holes and ions: implicit Euler** (Selberherr 6-4.32): the continuity equations
  get `dti·(n − n_prev)` in each control volume; `dti = 0` is the steady state, so a step back to
  `t = 0` is allowed.
- **Traps: no new unknowns.** Each level's occupation follows the linear ODE
  `df/dt = b − a·f` with `a`, `b` from `n`, `p` at the new time, solved over the step:
  `f = b/a + (f_prev − b/a)·exp(−a·Δt)`, bulk and interface alike; the recombination in the
  continuity equations is that ODE's capture integrated over the step, divided by `Δt`. The
  occupations are state carried from step to step, explicit in `n`, `p` at the new time.
- **Ions: full unknowns per grid point**, Scharfetter-Gummel fluxes, no flux through a region's
  ends or the contacts, in every ionic region that is mobile (`UpdateIons` is always on: no
  `preCond`, no `fixIons`). After each solve the region's density is rescaled to its total
  (`Rescale_Ion_Density`, the trapezoid rule over the region): the discrete equation alone
  does not conserve it (the region's end points take a whole control volume, a point at a
  contact none). At convergence the state is therefore the scaled solution, not the discrete
  equation's own: one unknown factor per region with the total as its constraint, which is
  what Stage B's slots are.
- **The current**: either form (`currDiffInt`) with `dti·(n − n_prev)` in the integral form's
  integrand; the ions' currents and the displacement current `ε·dti·Δ(∂V/∂x)` added;
  `Jint` the average over the grid.
- **`Vext`**: given (`Vint = Vext`); with `R_series > 0` the `Vint` that gives it; `oc` the
  open circuit (`Jext = 0`), at `t = 0` too. ZimT brackets `Vint` and runs Brent on whole
  solves, up to 100 of them per step.
- **Stopping**: `Track` 1 marks a reference state, 2 and 3 compare with it; five equal states
  in a row stop the run (`MinCountStatic`). `failureMode` 0 stops at a failed step, 1 keeps it,
  2 skips it.
- **Output** `tJFile`: `t Vext Jext errJ Vint Jint G_frac`, the QFLS and the photo- and
  recombination currents per layer and interface (bulk trapping per carrier, since the
  two differ out of steady state), `JminLeft JminRight JShunt Jnion Jpion JD`.

**How jaxdd solves it.** Newton on the whole step, as for the steady state: the same residual
with the time terms, so ZimT's converged state is jaxdd's to round-off, as SimSS's is. A
transient is a `lax.scan` over its steps, each one Newton solve from the step before; a batch
is `vmap` over devices, each with its own `t`, `Vext`, `G_frac` (padded to one length). The
steady state at `t = 0` is today's solver.

- **Devices without mobile ions** keep the 3×3 blocks `(V, ln n, ln p)`; the trap occupations
  ride along as state.
- **With mobile ions** the blocks grow to 5×5 `(V, ln n, ln p, ln a, ln c)`, identity rows
  where a species cannot be, and one slot per region (the rescaling). A batch takes 5×5 if
  any device needs it. The sweep (scan and Pallas kernel) takes blocks of either size.
- **Gradients** by forward tangents step by step, each one by the implicit function at its
  step, as M7's fits take them; no reverse pass through time (it would hold every state).

**Steps**, each validated against ZimT 5.36 at `tolPois = tolDens = 1e-12` (at 1e-9 ZimT moves
by 1.7e-7 of the scale; `zimt` in the image `jaxdd/simss:5.36`, its input read by `read_tvg`).
Electrons, holes, traps and mobile ions in time, with the voltage given, at the open circuit
and with a series resistance (`Vint` an unknown of each step), ZimT's automatic stop, failure
modes and `tJFile` columns are done (`jaxdd.transient`, docs/model.md "Transients"); next:

1. **Gradients through time**: tangents of a transient in its parameters. Done when they match
   finite differences, and a fit of a transient recovers its parameters.
2. **Speed**: a transient batch on the A100 (steps per second, Newton steps per time step, the
   5×5 step's cost against the 3×3 one). Done when its numbers are in the handbook; with them
   the fused kernel is decided (section 7).

Open beside them: the QFLS where a carrier lies below Newton's floor of convergence, in the
dark and in an ionic region's depletion zones (docs/model.md "Transients", up to 5e-3 V and
3 V off ZimT's): a floor, or a last step for the output, that converges those densities
without costing the ions' convergence (a lower floor alone converges them only without ions).

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
| The slowest lane | one stubborn device sets a batch's pace | two passes; difficulty-sorted batches; the fused kernel (2.3) |
| Convergence on new physics (Stage B, steric ions) | the hard devices fail first | each module validated on exactly those devices; flag, never guess |
| ROCm on gfx906 | the MI50 may never run it | not on the critical path; three ways in (1.3), the last independent of XLA's ROCm support |
| Scope creep towards a full SIMsalabim | months | every addition its own decision, module by module, each released when validated |
| The fused kernel and the JAX solver drift apart | two answers to one device | the kernel a fixed subset with fallback to JAX; both compared on every test device (2.3) |

## 7 Decisions ahead

| Decision | When | With |
|---|---|---|
| The fused kernel as an optional mode (2.3) | after Stage C's step 2 (4.1) | the applications' needs (section 5) and a first measurement of a Stage A kernel |
| The steric ion model and its `N_max` | when a source for `N_max` is found | its users: results with and without it differ at high ion density |
