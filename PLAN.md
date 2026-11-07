# Plan — `jaxdd`: a batched 1D drift-diffusion solver in JAX for A100/H100

Status: **draft, 2026-11-07.** Nothing built. Written while FauxPV's M20-00b ion grid (918 050
devices) runs on the 64-core server for an estimated 33–40 h.

A one-dimensional drift-diffusion solver for perovskite solar cells, written in JAX so that it
solves **hundreds of thousands of devices at once** on a data-centre GPU. Built on
SIMsalabim (LGPL-3.0) — its model, and where it helps its code, carried into a batched
solver — released under the same licence and validated against SIMsalabim's output. First target: FauxPV's drift-diffusion tables (`tools/dd_tables.py`),
which today cost days of CPU time per refinement. Later, maybe: FauxPV's F3 fidelity level
(drift-diffusion at run time, M20-07), and gradients for fitting.

---

## 1 Why, and when a GPU helps

**Today.** FauxPV's tables run SimSS 5.36 (Pascal, native code) in Docker containers, one
process per device, two runs per device (light and dark), 61 voltages each. On the server
(2× EPYC 7502, 64 cores) M19-14's run reached **7.8 devices/s** with 64 workers, 40 devices per
container: about 8 core-seconds per device. The tables so far cost:

| Run | Devices | Wall time (64 cores) |
|---|---|---|
| M19-13 grid | 37 800 | 65 min |
| M19-14 doping axis | 113 400 | 199 min |
| M19-14 ion axis | 216 000 | 461 min |
| M20-00b ion grid on four absorbers (running) | 918 050 | 33–40 h (pilot bound: 65 h) |
| M20-00b layer axes to 1 µm (D0 a, open) | 822 528 | about 30 h |

Every further axis multiplies this. An absorber axis for the whole grid, a temperature axis
with ions, a mobility axis for the transport layers (FauxPV #61), a permittivity axis (M20-00b
D1 b): each is a few million devices, weeks on the server.

**Why a straight rewrite does not help.** SimSS is compiled code; a Rust port would run at the
same speed, plain Python slower. One device is too small for a GPU: about 400 grid points,
three unknowns per point, a block-tridiagonal Newton step of a few hundred kFLOP. A GPU loses
on one device to its kernel-launch latency.

**Why a batched one does.** The devices are independent. A GPU that solves 10⁵ of them in
lockstep has a block-tridiagonal system of 10⁵ × 400 × 3 unknowns per Newton step, sequential
along the 400 grid points and parallel across the batch. Back of the envelope, bandwidth-bound:
per grid point and Newton step the batch reads and writes about 100 bytes per device, 10 MB
for 10⁵ devices; at 1.5–3 TB/s (A100/H100 HBM) that is 3–7 µs per grid point, 3–6 ms per
Newton step with forward and backward sweeps. With about 20 Newton steps per voltage and 122
voltages per device (light and dark), 10⁵ devices take 10–15 s: **of the order of 10⁴
devices/s, a thousand times the server.** The estimate leaves out the launch overhead of a
scan over 400 points, the slowest device holding up its batch (§4.4) and the transcendental
functions in FP64. It may be off by a factor of 10–100 and still be worth it. M3 measures it.

**FP64 is not optional.** Carrier densities span 20 orders of magnitude, and the
Scharfetter–Gummel fluxes subtract nearly equal exponentials. A100 (9.7 TFLOP/s FP64, 19.5
with its FP64 tensor cores) and H100 (34 / 67) do FP64 at full rate; consumer GPUs at 1/64 —
for a bandwidth-bound solver less of a loss than the ratio suggests, but the exponentials are
not bandwidth-bound. Consumer GPUs are measured in M3 as well, not assumed.

## 1a M0 — measured (2026-11-07)

One chunk of 40 devices (80 SimSS runs) of FauxPV's M20-00b ion grid, one container, one core
of the laptop:

| Part | Before | After the fix below |
|---|---|---|
| container start and stop | 0.33 s per chunk | the same |
| writing the inputs (Python) | 4.5 s (0.11 s per device) | the same |
| SimSS | 94.6 s | 81.0 s |
| reading the outputs | 0.1 s | the same |
| disk written | 1.2 GB | 8 MB |

**Overhead is 5–6 %**: Docker and files are not the bottleneck, SimSS is. **The fix:** SimSS 5.36
wrote its internal variables to a file literally named `none` (14.7 MB per run) although
`varFile = none`; `outputRatio = 0` stops it, with bit-identical curves (FauxPV 7e483de). On
the server, where 128 workers wrote it at once, the saving is likely larger than the 14 % here.

**The time is in a few devices.** The 8 slowest of 80 runs took 74 % of the chunk's SimSS time,
up to 9.8 s each, all at `ℓ` 0.2–0.5 on 700–1000 nm absorbers; a run at `ℓ` ≥ 1 takes about
0.2 s (median 0.24 s). For `jaxdd` this is §3.4's lockstep problem in numbers: batches must
be sorted by difficulty (`ℓ`, absorber), or the hard tenth sets the pace of all. It also says
where the solver's robustness work goes (M1–M2).

## 2 Scope

### 2.1 The physics FauxPV's tables need (phase 1)

What `tools/simss_reference.py::simss_parameters` and `tools/dd_tables.py` ask of SimSS today,
and nothing more:

| Part | What |
|---|---|
| Layers | ETL, absorber, HTL; per layer thickness, permittivity, band edges, effective densities of states, mobilities, doping (donors in the ETL, acceptors in the HTL) |
| Interfaces | band offsets; thermionic transfer at a finite velocity (`nu_int`, 1e5 m/s) for the majority carrier; the minority carrier blocked by a 1 eV offset |
| Contacts | ohmic electrodes at given work functions (`W_R − W_L = V_bi`) |
| Recombination | bulk SRH through one neutral trap level (`C_n = C_p`, `E_t`), direct (net radiative) recombination; none in the transport layers |
| Generation | a profile from a file (Beer–Lambert of AM1.5G, light from either side) |
| Ions | anions and cations of equal density in the absorber only; settled at a pre-bias under light (`preCond`), then frozen for the scan (`fixIons`, with the patch FauxPV applies to SimSS 5.36 so that Poisson's equation also holds them); and the steady state with ions free at every voltage |
| Output | `J(V)` light and dark on a fixed voltage grid (−0.2 to 1.3 V in 25 mV), its error estimate, the absorber's mean QFLS; per device a convergence flag (SimSS: `failureMode`) |

### 2.2 Later phases, each its own decision

Transients (ZimT: scan-rate hysteresis, TRPL, impedance) for FauxPV's F3; charged and multiple
trap levels, interface traps; temperature-dependent parameters; tunnelling. None is needed for
the tables.

### 2.3 Not part of this

A line-by-line port of SIMsalabim (its one-device Gummel structure does not batch; §3.2,
§9); 2D/3D; a GUI; replacing SimSS as FauxPV's reference (SimSS
stays the yardstick the new solver is validated against).

## 3 Numerics

### 3.1 Equations and variables

Poisson's equation for the potential `ψ` with free carriers, dopants and ions; the continuity
equations of electrons and holes in steady state; the ions either fixed (frozen) or in
zero-flux equilibrium. Scaled throughout: lengths by the device thickness, potentials by
`V_T`, densities by a reference density, so the unknowns stay O(1) where possible.

The unknowns are a decision (D2): `(ψ, n, p)` with Scharfetter–Gummel fluxes, as SimSS and most
device codes do, or `(ψ, φ_n, φ_p)` with the quasi-Fermi potentials, where `n = N_c·exp(...)`
never goes negative and the dynamic range lives in the exponent. The first is the safer match
to SimSS; the second converges more robustly at high injection. M1 tries both on the same
devices.

### 3.2 The solver: coupled Newton on a block-tridiagonal system

A coupled Newton step (all three unknowns at once) gives a block-tridiagonal Jacobian with 3×3
blocks. It is solved by a block Thomas algorithm: a `lax.scan` over the grid points, with the
batch as the leading axis of every array, so each scan step is a batched 3×3 solve. A
decoupled Gummel iteration (SimSS's scheme) would need three scalar tridiagonal solves per
iteration (`jax.lax.linalg.tridiagonal_solve`, batched on GPU) but more iterations, and it
stalls at high injection; it is the fallback if the coupled step is fragile.

Damping: the classic potential-update limit (Bank–Rose, or a cap on `|Δψ|` of a few `V_T`).
Convergence: the update and the residual, both scaled, per device.

### 3.3 Frozen and free ions

**Free ions in steady state** carry no current, so their density is Boltzmann's in the
potential with a normalisation fixed by their total number: `n_a(x) = N·L·exp(ψ/V_T) /
∫exp(ψ/V_T)` (and the opposite sign for cations). The integral couples every grid point to a
scalar: a block-tridiagonal matrix bordered by one row and column per species. Solved with the
Sherman–Morrison–Woodbury identity on top of the block Thomas sweep — two extra sweeps, no
dense algebra.

**Frozen ions** are that profile at the pre-bias `V_pre`, computed once, then held as a fixed
charge for the scan.

### 3.4 Voltage continuation and the lockstep problem

Each device starts from its equilibrium (or `V_pre`) and steps through the voltage grid,
using the previous solution as the initial guess. The batch steps through the voltages
together; a device whose Newton iteration fails at a step halves its step locally (masked
sub-stepping) while the others wait.

Under `vmap`, a `lax.while_loop` runs until every lane has converged: **the batch costs what
its slowest device costs.** Mitigations, in order: a hard iteration cap per voltage with the
device flagged (SimSS skips such voltages too, about 1 % of them in FauxPV's tables); sorting
devices into batches of similar difficulty (by `ℓ`, ion density, doping — known in advance);
and a second pass over the failed ones with smaller steps.

## 4 The grid in space ("adaptive meshing", meaning 1)

JAX compiles for static shapes: every device in a batch has the same number of grid points.
The node **positions** can differ per device. Three steps of adaptivity, cheapest first:

1. **A priori graded mesh per device.** Geometric refinement towards every interface and
   contact, scaled by the local Debye length `L_D = √(ε·V_T/(q·N))` of the dopants and ions.
   That matters here: at 1e18 cm⁻³ of ions `L_D` is about 6 nm (ε 24.1), and FauxPV's tables show that
   the strongly screening ions are where the 0D lookup is worst (M20-00, M20-00b). SimSS
   grades its grid too; matching its resolution is part of the validation.
2. **r-adaptivity (moving nodes, fixed count).** After a first solve, redistribute the nodes
   to equidistribute a monitor function — the arc length of `ψ` and of `log n`, `log p` — and
   solve again. The count stays fixed, so it fits static shapes and batching.
3. **h-adaptivity (changing count)** only by bucketing: batches compiled at 200, 400, 800
   points, each device in the smallest bucket that meets its error estimate. More compile
   time and bookkeeping; only if M4 shows 1 and 2 are not enough.

M4's study: `J(V)` and `η` at 200, 400, 800, 1600 points, uniform against graded against
r-adaptive, on the hardest devices of FauxPV's tables (`ℓ` 0.1–0.2, ions 1e18, thick undoped
layers).

## 5 Fewer devices instead of faster ones ("adaptive", meaning 2)

What I called adaptive earlier was mostly this: **sampling the parameter space where the
result changes, instead of on a full tensor grid.** A fast solver makes the loop below
interactive; on the CPU server each iteration would take a day.

- **Axis refinement (keeps FauxPV's lookup).** FauxPV interpolates on a regular grid, so the
  grid must stay a tensor product, but each axis's points can be chosen by error: solve at
  the midpoints between neighbouring axis points across a sample of the other axes, and insert
  a point where the interpolation misses by more than the tolerance. This is how FauxPV found
  its doping and ion-density axes by hand (M19-14 E15, E22); automated, it replaces guessing.
- **Sparse grids** (Smolyak) or **scattered samples with a surrogate** (a GP or a small network,
  the surrogate already in FauxPV's `TODOS.md`). Far fewer devices in 7–9 dimensions, but
  FauxPV's lookup would change from interpolation to a model evaluation — a FauxPV decision,
  not this project's.

Either needs the solver to be callable from Python in a loop at low latency. That is a
requirement on the API (§6), not only on throughput.

## 6 Software

- **Package** `jaxdd`, Python ≥ 3.11, `jax` with `jax_enable_x64`; on the cluster
  `jax[cuda12]` in an Apptainer image, run under Slurm. Pure functions only: a device is a
  pytree of arrays, `solve(device, voltages) -> curves`, `vmap`ped and `jit`ted; no global state.
- **Batches across GPUs** with `jax.sharding` (`shard_map`) over the batch axis; the devices are
  independent, so no communication.
- **Memory.** About 30 arrays of 400 points in FP64 per device (~100 kB) and the Jacobian's
  3×3 blocks (~90 kB): about 200 kB, so 10⁵ devices in about 20 GB. An 80 GB card holds
  3–4·10⁵ per batch.
- **I/O.** FauxPV's table format (`simss_tables.npz`: `J_light`, `J_dark`, `errJ`, QFLS on the
  same axes), so `tools/dd_tables.py` gains a `--backend jaxdd` and nothing downstream changes.
- **FauxPV never imports it at run time** for F0 (its tables are data). If F3 adopts it later,
  as an optional extra beside the SIMsalabim subprocess of M20 D4.
- **Determinism.** Bit-identical on one GPU type and JAX version; across them within a stated
  tolerance (XLA's reductions differ). As FauxPV's M20 D3 for F3.

## 7 Validation

The reference is SimSS's own output, which FauxPV already holds: **367 200 devices in
`simss_tables.npz`**, 512 in `absorber_tables.npz`, soon 918 050 in `absorber_ion_tables.npz`,
the off-grid checks in `data/drift_diffusion/checks/`, and the benchmark under
`data/benchmarks/simss/`. Same device descriptions, so a comparison is a script, not a
project.

| Gate | Criterion |
|---|---|
| Ion-free devices | `η` within 1e-3 of SimSS at every voltage up to 1.0 V, `J_SC` within 0.1 %, on a random 1 000 of the table |
| Frozen ions | the same on 1 000 devices with ions, including the step of M19-14 E22 and `ℓ` 0.1 at 1e18 |
| Free ions | SimSS's unpatched steady state at 0 V, as M19-13 E15 checked it |
| Failures | where SimSS skips a voltage, `jaxdd` either converges (and is checked against SimSS at 800 points) or flags it; no silent wrong curve |
| Physics invariants | current continuity to 1e-9 relative; equilibrium at 0 V in the dark gives zero current and flat quasi-Fermi levels; `η` never above 1 + the generation's quadrature error |

Tolerances are a decision (D3); the ones above are FauxPV's off-grid tolerances divided by
30, so the solver is never the larger error.

## 8 Milestones and gates

| | Work | Where | Size | Gate to continue |
|---|---|---|---|---|
| **M0** | Profile one FauxPV chunk: container start, file I/O, SimSS's own time. **Done 2026-11-07** (§1a): overhead 5 %; the var-file fix took 14 % off a chunk | FauxPV | 1 day | — |
| **M1** | One device, CPU, x64: Poisson + continuity, SG, coupled Newton, contacts, offsets, thermionic interface, SRH + direct, generation profile; no ions. Both variable sets (D2) | laptop | 1–2 weeks | ion-free gate of §7 |
| **M2** | Ions: frozen (pre-bias, then fixed) and free (bordered system) | laptop | 1–2 weeks | both ion gates of §7 |
| **M3** | Batched: `vmap`, masked `while_loop`, continuation with sub-steps; on one A100 or H100 (RZ), and on a consumer GPU for comparison | RZ | 1 week | **go/no-go: ≥ 50× the server's 7.8 devices/s on one GPU, at the §7 accuracy** |
| **M4** | Mesh: graded, r-adaptive; the convergence study of §4 | RZ | 1 week | 400 points or fewer meet the gates on the hardest devices, or bucketing is justified |
| **M5** | FauxPV backend: `dd_tables.py --backend jaxdd`, the table format; a stored table reproduced to tolerance | FauxPV + RZ | 1 week | 48-point `reproduce` within §7 |
| **M6** | Adaptive axes (§5): error-driven axis refinement; first use on M20-00b's open layer axes (D0 a) | FauxPV + RZ | 1–2 weeks | the refined axes meet the off-grid tolerance with fewer devices than the hand-made ones |
| M7 | Optional: implicit gradients (`jax.custom_root`) for fitting; transients for F3 | — | open | own plan |

Total to the first useful table (M0–M5): **5–8 weeks** of focused work, most of it M1–M2
(physics and robustness), not the GPU part. M3 decides early, before M4–M6 are spent.

## 9 Licence and attribution

SIMsalabim is LGPL-3.0, which allows modified and translated versions on one condition: they
are LGPL-3.0 (or GPL-3.0) themselves. **`jaxdd` is released under LGPL-3.0** (decided
2026-11-07, open source like SIMsalabim), so it may read and carry over SIMsalabim's code
wherever that helps — boundary conditions, interfaces, ions, the convergence safeguards —
which takes most of the risk out of M1–M2. What that requires:

- `COPYING.LESSER` (LGPL-3.0) and `COPYING` (the GPL-3.0 it supplements) in the repository.
- Where code follows SIMsalabim's, the original copyright notice kept and a prominent note that
  it was translated and changed, with the date (GPL-3.0 §5 a). A file `NOTICE` lists which
  modules derive from which SIMsalabim units.
- Citation in the README and in every result: Koopmans, Le Corre, Koster, J. Open Source Softw.
  7, 3727 (2022), doi:10.21105/joss.03727, and the SIMsalabim repository with the version
  followed (5.36, FauxPV's).
- FauxPV is unaffected: it calls `jaxdd` from `tools/`, as it calls SimSS today, and LGPL is
  made for use by programs under other licences.

The structure is new regardless (§3.2: a coupled, batched Newton step instead of one device's
Gummel iteration), so the code is "after SIMsalabim", not a transliteration. Before the first
release, a word with the Koster group: they maintain pySIMsalabim, and a GPU backend may be of
use to them — at best a joint effort. Not legal advice; for a release with an institute's
name, the institute's legal office has the last word.

## 10 Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Convergence robustness | SimSS embodies years of tuning; the hard devices (`ℓ` 0.1, 1e18 ions, thick undoped layers) fail first | SIMsalabim's safeguards carried over (§9); M1–M2 on exactly those devices; Gummel fallback; flag, never guess |
| The slowest lane | one stubborn device holds a batch of 10⁵ | iteration caps, difficulty-sorted batches, a second pass |
| FP64 cost of `exp`/`log` | slower than the bandwidth estimate | Bernoulli function via series near 0 and cached exponentials; measure in M3 |
| `lax.scan` over 400 points | launch overhead per step dominates small batches | large batches; if needed a Pallas/Triton kernel for the block Thomas sweep |
| Scope creep towards a full SIMsalabim | months | §2.1 is the scope; every addition its own decision |
| RZ access and queues | M3 waits for GPU time | M1–M2 need no GPU; M3 needs hours, not days |

## 11 Decisions

| | Question | Options | Recommendation |
|---|---|---|---|
| D1 | Language | JAX (Python); Rust + CUDA; C++/CUDA | **JAX**: batching by `vmap`, gradients free, one language with FauxPV; a hand kernel only where M3 shows the scan is the bottleneck |
| D2 | Unknowns | `(ψ, n, p)`; `(ψ, φ_n, φ_p)` | try both in M1 on the same devices |
| D3 | Tolerances | §7's, or FauxPV's own off-grid tolerances | §7's (the solver never the larger error) |
| D4 | Licence of `jaxdd` | MIT; Apache-2.0; LGPL-3.0 | **LGPL-3.0, decided 2026-11-07** (§9): SIMsalabim's code may be used, open source like it |
| D5 | Where it lives | its own repo on Forgejo; `fauxpv/projects/` | **its own repo**: it is useful without FauxPV, and FauxPV only calls it from `tools/` |
| D6 | M0 first | yes; no | **yes**: a day, and it may halve the current runs without any of this |
