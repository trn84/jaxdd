# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

0.1.0 is released (CHANGELOG.md); the transients (Stage C) are complete.

1. **M6: adaptive parameter axes, for 0.2.** Sample the parameter space where the result changes
   instead of on a full tensor grid.
   - *Axis refinement* keeps an interpolated table: each axis stays a tensor-product axis, but its
     points are chosen by error. Solve at the midpoints between neighbouring axis points across a
     sample of the other axes, and insert a point where the interpolation misses by more than
     the tolerance.
   - *Sparse grids* (Smolyak) or *scattered samples with a surrogate* (a GP or a small network)
     need far fewer devices in 7–9 dimensions, at the price of a model evaluation instead of an
     interpolation.

   Either needs the solver callable from Python in a loop at low latency, which is a requirement
   on the API, not only on throughput. **Done when** the refined axes meet a table's
   interpolation tolerance with fewer devices than hand-made ones.

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

### 2.2 The optics for the measured cell (waiting for n,k data)

The measured fits assume a uniform generation. The cell in its own orientation with the
transfer-matrix profile is `tests/data/pin_anode_left_light`, but the paper's perovskite n,k
data (Werner et al., ACS Energy Lett. 3, 742 (2018)) are not among SIMsalabim's files; a
stand-in's profile would be a different assumption, not a better one. **Done when** the measured
fits in docs/validation.md are repeated with the computed profile.

### 2.3 A steric limit on ion accumulation (waiting for a source of `N_max`)

Frozen at full accumulation, 1e18 cm⁻³ of ions pile up in nanometre layers at the interfaces at
1.5e21 cm⁻³, 1500 times their mean and close to the perovskite's lattice-site density
(~1e22 cm⁻³). Boltzmann statistics let ions stack without limit; real vacancies cannot.

The proposal: the modified Poisson–Boltzmann (Bikerman) occupancy for the ions, saturating at a
site density `N_max`, in the free steady state and so in the frozen profile. It departs from
SimSS (documented in docs/differences.md, switchable, SimSS's Boltzmann the reproduction), and
it changes what an ion density means at the high end, so results with and without it must not
be mixed: to be decided with its users. Its convergence must be validated on exactly the hard
devices (flag, never guess). **Done when** `N_max` has a source (a literature value for the
halide vacancy site density) and the model is validated against the Boltzmann one where the
two must agree.

## 3 Later

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
