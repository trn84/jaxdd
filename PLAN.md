# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

0.2.0 is released (CHANGELOG.md).

### 1.1 Free surfaces: the rest

A free end (`W_L` or `W_R` = `free`, [docs/model.md](docs/model.md#free-surfaces-jaxdds-w_l-or-w_r--free))
is solved in the steady state, one device (`solve_state`) or a batch (`solve_states`).
Photoluminescence is measured on partial stacks of perovskite films, so what such films hold has
to work at a free end too. Refused at a free end until then:

1. Mobile ions: the ions' constraint over the dead interval, a free end's ion flux 0.
2. Field-dependent generation and Langevin in a field-dependent mobility: the field at the end
   point one-sided.
3. `solve_transient` (a transient photoluminescence decay): the gauge row holds only in the
   steady state, so in time with both ends free it is the total charge's conservation instead.

**Done when** each is in, with a test against an analytic case or a refined grid, and the
refusals are gone.

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

- **Ions bounded by their sites in time.** The steady state has them (docs/model.md), a
  transient refuses them: in time the ions need a flux of their own, nonlinear diffusion at a
  constant mobility or a modified drift at a constant diffusivity (the same steady state,
  different dynamics; Abdel, Courtier, Farrell 2023), a choice for its users. **Done when** one
  (or both, switchable) is in `solve_transient`, against IonMonger's or ChargeTransport.jl's
  transients where those exist.
- **Samples that are not a tensor product**, sparse grids (Smolyak) or scattered samples with a
  surrogate (a GP or a small network): a feature that lies across the axes costs a tensor
  product its points everywhere ([docs/validation.md](docs/validation.md#a-table-with-refined-axes)),
  and in 7–9 dimensions a table is out of reach. The price is a model evaluation instead of an
  interpolation. **Done when** such a sample meets a table's tolerance with fewer devices than
  `refine_axes`.
- **Impedance in the frequency domain**, from the steady state's Jacobian
  (`(J + iω·M)·δx = b`, one complex solve per frequency) instead of a transient per frequency.
  ZimT has no reference for it but its own time-domain response.

## 4 Not planned

- **A fused CUDA kernel**: decided against for now; why, and when it comes back, is in
  [docs/hardware.md](docs/hardware.md#why-there-is-no-fused-kernel).
- **A line-by-line port of SIMsalabim** (its one-device Gummel structure does not batch), a GUI.
  SimSS and ZimT stay the references jaxdd is validated against. Every addition beyond them is
  its own decision, released when validated.
- **Two dimensions**: a project of its own ([docs/overview.md](docs/overview.md#what-the-speed-is-for)).
