# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

0.5.0 is released (CHANGELOG.md). Nothing is next until one of the items below is taken up.

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

### 3.1 Ageing simulators: what may still be needed

The four items an ageing caller needed (a transient from a given state, the ions in the
profiles, Auger, bit-identity on the CPU) are done (CHANGELOG.md). Not taken up until a caller
shows it needs them:

- **A temperature changing in time within a transient** (a scan that warms the device; jaxdd's
  `T` is one per device and call). A caller can split the transient at temperature nodes and go
  on from each segment's state with the next temperature's device; FauxPV's F3 plans that.
- **Two ion species of one sign** (jaxdd has one anion and one cation species). FauxPV's F3
  refuses a second, slow species for now; this is the item it would need.

## 4 Not planned

- **A fused CUDA kernel**: decided against for now; why, and when it comes back, is in
  [docs/hardware.md](docs/hardware.md#why-there-is-no-fused-kernel).
- **Two dimensions**: a project of its own ([docs/overview.md](docs/overview.md#what-the-speed-is-for)).
