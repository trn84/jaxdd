# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

0.3.0 is released (CHANGELOG.md). Nothing is next until one of the items below is taken up.

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

- **Samples that are not a tensor product**, sparse grids (Smolyak) or scattered samples with a
  surrogate (a GP or a small network): a feature that lies across the axes costs a tensor
  product its points everywhere ([docs/validation.md](docs/validation.md#a-table-with-refined-axes)),
  and in 7–9 dimensions a table is out of reach. The price is a model evaluation instead of an
  interpolation. **Done when** such a sample meets a table's tolerance with fewer devices than
  `refine_axes`.
- **TRPL, the SnO2 stack lit from the top.** With saturable interface states, the pulse's
  absorbed share by side and SnO2's band edge from 4.0 eV the eight decays of Kober-Czerny et
  al. fit to 0.091 decades rms; the SnO2 stack lit from the top stays at 0.20, the rest 0.04–0.07
  ([docs/validation.md](docs/validation.md#trpl-without-electrodes)), and the fit then holds the
  SnO2 interface's SRV at its bound and the top side's absorbed share at 0.57 (0.78 expected). Not
  explained by SnO2 traps, a dipole (the same as SnO2's band edge) or photon recycling.
  **Done when** that decay is within the others' misfit with every parameter inside its bounds,
  or its misfit is traced to the measurement.
- **Fermi–Dirac statistics, the rest.** The steady state and the transient have Blakemore's and
  `F_{1/2}` (docs/model.md, "The carriers' statistics"), against ChargeTransport.jl, with
  interface traps and free surfaces too. Left: a blocking contact's accumulation on the graded
  grid: with `F_{1/2}` `tests/test_free_surface.py`'s stack converges to −0.5 V on SIMsalabim's
  grid but stalls beyond −0.2 V on the graded one at a round-off floor of 1e-3–1e-2 (its 1 pm
  intervals at the contact; docs/solver.md, "A floor above 1e-6"). **Done when** that stack
  converges at `W_R` 5.4 eV and −0.5 V on the graded grid (a smallest interval no smaller than
  a Debye length, say).
- **Impedance with interface traps**: their occupations on each interface's edge have their
  own dynamics (`solver._interface_occupation`), as the bulk traps' do (`jaxdd.impedance`,
  point by point there); `solve_impedance` refuses them. **Done when** the frequency solve
  with `dt` is `solve_transient`'s sine response with interface traps, as without them
  (`tests/test_impedance.py`).

## 4 Not planned

- **A fused CUDA kernel**: decided against for now; why, and when it comes back, is in
  [docs/hardware.md](docs/hardware.md#why-there-is-no-fused-kernel).
- **Two dimensions**: a project of its own ([docs/overview.md](docs/overview.md#what-the-speed-is-for)).
