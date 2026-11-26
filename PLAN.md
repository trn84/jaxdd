# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

0.4.0 is released (CHANGELOG.md). Nothing is next until one of the items below is taken up.

## 2 Waiting

### 2.1 M8: the MI50 cards (waiting for a run on the host)

Not the focus: jaxdd is made as fast as it can be on NVIDIA first. An AMD track with its own
optimisations follows later, beside it, starting from the scan, which runs everywhere. Prepared:
JAX 0.4.31 in `rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10` runs `jax.jit` on gfx906, and
jaxdd passes its tests on JAX 0.4.31 ([docs/hardware.md](docs/hardware.md)).

1. On the MI50 host: `rocm/mi50.sh`, then send `rocm/mi50_probe.log`.
2. The batch a 16 GB card holds; the four cards as four processes (`HIP_VISIBLE_DEVICES`).
3. A large sweep split across the four cards.

If the image fails: the ROCm 6.1.3 and 6.2.3 images with the same JAX; then jaxlib built from
source for gfx906 against a ROCm that still ships its libraries (6.2/6.3); last, the block sweep
as a hand-written HIP kernel beside JAX.

**Done when** the benchmark's results are reproduced on the cards within the validation
criteria, at a throughput worth the cards (four MI50 are, for this bandwidth-bound FP64 solver,
about two A100s).

## 3 Later

### 3.1 Ageing simulators: jaxdd stepped through time by a caller

A simulator that ages devices step by step and measures them in between can call jaxdd as its
solver: a steady state or a transient per device and ageing step, a transient J-V scan per
measurement, the caller owning the time loop. Four things such a caller needs that jaxdd lacks;
the cost per call, measured there, may add more.

1. **A transient from a given state.** `solve_transient` starts every device from a steady state,
   the tVG's first step at `t = 0`; it cannot be handed a state. Slow ions that relax over
   minutes to a day are left off their steady state by an ageing step of an hour, and the next
   step, or a scan at 298 K right after ageing at 338 K, has to start where it left them. To do:
   an initial state (potential, densities, ions per point) accepted in place of the first
   steady state, for every device of a batch. **Done when** a transient split in two at any step,
   the second half started from the first's last state, gives the unsplit transient's currents
   and states within the Newton tolerance.
2. **The ions in the transient's profiles.** `profiles=True` returns `V`, `n` and `p` per step,
   not the ions, so the state of item 1 cannot be taken from a result. To do: `anion` and
   `cation` beside them, as `solve_states` has them. **Done when** a profile's ions integrate to
   each species' total within round-off and item 1's test runs on them.
3. **Auger recombination.** Neither SimSS nor jaxdd has it; it shapes TRPL at high fluence and
   the curve at high injection, and a model whose devices carry an Auger coefficient loses it
   at the drift-diffusion level. To do: jaxdd's own term `(C_n·n + C_p·p)·(n·p − n_i²)` per
   layer, zero by default so that SimSS's inputs solve to the bit as before, with its
   derivatives in the Jacobian. **Done when** the zero default
   leaves every SimSS comparison unchanged to the bit, and a uniform device in high injection
   decays at the closed form's Auger rate within the time steps' error.
4. **Bit-identity on the CPU, tested.** A caller that promises reproducible datasets needs the
   same jaxdd commit, JAX version and backend to give the same bits on the same platform; no test
   of jaxdd's checks that a solve repeated in one process, or in a new one, does. **Done when**
   a test solves `solve_batch`, `solve_states` and `solve_transient` twice in fresh processes on
   the CPU and compares them with `array_equal`.

Not taken up until a caller shows it needs them: a temperature changing in time within a
transient (a scan that warms the device; jaxdd's `T` is one per device and call), and two ion
species of one sign (jaxdd has one anion and one cation species).

## 4 Not planned

- **A fused CUDA kernel**: decided against for now; why, and when it comes back, is in
  [docs/hardware.md](docs/hardware.md#why-there-is-no-fused-kernel).
- **Two dimensions**: a project of its own ([docs/overview.md](docs/overview.md#what-the-speed-is-for)).
