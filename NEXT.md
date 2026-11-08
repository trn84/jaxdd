# Where jaxdd stands, and what comes next

Hand-over note, kept current at the end of a working session. The full plan is
[`PLAN.md`](PLAN.md); where jaxdd departs from SIMsalabim on purpose, [`DIFFERENCES.md`](DIFFERENCES.md).

## State (2026-11-07, commit 5dde971)

| Milestone | State |
|---|---|
| M0 profile SimSS | done: overhead 5 %; SimSS 5.36 wrote a 14.7 MB var file per run (`outputRatio 0` fixes it, in FauxPV) |
| M1 ion-free steady state, one device | done; matches SimSS to 1–2e-6 of the current scale with its Bernoulli approximation |
| M2 ions (free; frozen after a pre-bias) | done; within 2e-5 |
| §7 gate, 1000 devices of FauxPV's table | met against a converged SimSS: every miss is SimSS stopping early (DIFFERENCES.md §1) |
| M3 batched (`jaxdd.batch.solve_batch`) | **done 2026-11-08: A100 1198 devices/s at batch 16384, 154× SimSS on 64 cores** (gate 390/s) |

**A100 results** (RunPod, `validation/bench.py`, batch 16384 unless noted): 143 → 265 → 323 →
341 → **1198 devices/s** (5dde971: the ions' constraint by `jax.jacrev` instead of `jacfwd`,
extrapolated starts, the quadratic stop). Breakdown of one Newton step (`breakdown.py`, 67 ms for
16384 devices): Jacobian blocks + ion columns 46 ms, sweep 15, the rest 6.

The Newton system is solved by the block Thomas sweep; parallel cyclic reduction and a
PCR–Thomas hybrid were built, measured slower everywhere, and taken out (PLAN.md §8a).

**A device in `bench.py` was one run (one curve); SimSS's 7.8 devices/s are FauxPV devices of
two runs each (light and dark).** Per run the A100 is 1198 / 15.6 = **77×** SimSS on 64 cores,
not 154× (`bench.py` corrected 2026-11-08; the gate of 50× still holds). The M20-00b ion grid is
1.84 M runs: about 26 min of solving, not 13.

## M5 (2026-11-08): the FauxPV side built; every miss of the gate is SimSS's

FauxPV's `tools/dd_tables.py --backend jaxdd` (FauxPV task M20-00c): the SimSS input files of
every grid point, read by `read_setup`/`build_device` in worker processes, solved by
`solve_batch` per grid size and voltage axis, into tables of their own. `reproduce --backend
jaxdd` on 48 grid points of `simss_tables.npz` (CPU, 30 s): median `η` gap 1.5e-5, 44 within
1e-3. The four that miss, each rerun with SimSS at `tolDens = tolPois = 1e-9` (FauxPV
`data/drift_diffusion/checks/m20_00c_simss_tight.{py,json}`, `m20_00c_prebias_ions.{py,json}`):

| FauxPV grid point | `ℓ` | `N_ion` cm⁻³ | `x_ion` | jaxdd − SimSS 1e-9, `η` | SimSS's pre-bias | jaxdd given SimSS's frozen ion numbers |
|---|---|---|---|---|---|---|
| p1_4_2_0_5_0_3 | 0.2 | 3e16 | 0.5 | 5.4e-6 | converged | — (SimSS's early stop in the scan, DIFFERENCES §1) |
| p0_10_1_3_4_2_3 | 0.1 | 1e18 | 0.25 | 8.1e-3 | not converged | 1.1e-5 |
| p1_10_4_1_2_4_2 | 0.2 | 1e18 | 1 | 3.7e-2 | not converged | 5.9e-6 |
| p2_8_2_3_0_4_2 | 0.5 | 3e17 | 0.5 | 4.0e-3 | not converged | 8.7e-5 |

The three are SimSS's: its pre-bias stopped at `maxItSS` and it froze ions off their number by
0.8–6.5e-4 of `N·L` (DIFFERENCES §1b). jaxdd needs no fix for them. (A first reading the same
day, commit 66c525f, had them as jaxdd's.)

## Next

1. **Use: the M20-00b ion grid on the A100**, once FauxPV decides which solver's grid is the
   reference (FauxPV M20-00c, D1). Before it, a larger FauxPV `reproduce --backend jaxdd`
   sample of the points with ions at `ℓ` ≤ 0.5, to size how many of SimSS's tables hold an
   unconverged pre-bias.
2. Speed, after that, the host first. Preparing a FauxPV grid point costs ~90 ms on one core
   (laptop, 2026-11-08: FauxPV writing SimSS's files 47 ms, `read_setup` 18 ms, `build_device`
   24 ms) against ~1.7 ms of A100 time; on fewer than ~32 cores the GPU waits. Planned (FauxPV
   M20-00c, "The device building without files"): **`read_setup` from a mapping of file names
   to texts** as well as from a directory — FauxPV hands over the very texts it would write, so
   the inputs stay SimSS's byte for byte, and the disk drops out (about two thirds); then
   `build_device`'s Python loops (`_sub_grid`, `_generation`) vectorised if the pod's profile
   still shows them. Not a second way to describe a device. Then the solver: assemble the
   Jacobian by edges (each interval's flux once) and in one `jacfwd`, estimated 1.5–2× (about a
   day); a fused kernel only if far more is needed. Hard devices hold a batch in lockstep (M0:
   the slowest tenth is 74 % of SimSS's time): sort batches by `ℓ` and absorber.
3. M4 (mesh study), M6 (adaptive axes).
4. Proposal PLAN.md §8b, for FauxPV and jaxdd together: a steric (Bikerman) limit on ion
   accumulation. Not started.

## Known limits

- **The Newton shortcuts cost no accuracy that matters** (`validation/shortcuts.py`, 2014
  devices: FauxPV's 1000-point sample light and dark, and the ion cases; 2026-11-08): against
  the same solver run strict (`solve_batch(..., strict=True)`, Newton to a 1e-10 step), the
  largest difference in J is 1.8e-6 of the generation current (the hard frozen case below),
  the 99th percentile 8.5e-9, the median 2e-10; no device beyond 1e-4. They save 27 % of the
  Newton steps (2.60 against 3.56 per voltage). Unconverged with them: 3 points, all in the
  hard frozen case in the dark (strict: 59 there, without the settled-current criterion).
  Every point carries its `converged` flag; nothing is cut off silently.

- l 0.2 with 1e18 cm⁻³ ions frozen at full accumulation: the potential keeps a few 1e-5 V_T of
  round-off noise in the carrier-free bulk; converged by the settled current (noise ≤ 3e-6 of
  the scale). SimSS skips a third of those voltages.
- `fixIons` without `preCond` is refused (ions held at their initial uniform density).
- Stage A only: refused by name otherwise (`simss_input.check_supported`).

## The A100 run (RunPod, fresh pod, CUDA 12 image with Python ≥ 3.11)

```bash
cd / && git clone https://github.com/trn84/jaxdd.git && cd jaxdd
python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]" "jax[cuda12]"
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # CudaDevice
.venv/bin/python -m pytest -q
.venv/bin/python validation/bench.py 16384 4096  2>&1 | tee bench_4096.txt
.venv/bin/python validation/bench.py 65536 16384 2>&1 | tee bench_16384.txt
cd validation && ../.venv/bin/python breakdown.py 16384 2>&1 | tee ../breakdown_16384.txt
```

## Working notes

- Local reference material outside the repository, next to it in `../jaxdd-ref/`: the
  SIMsalabim 5.36 source (read for M1–M2), SimSS cases (`cases/`, `ion_cases/`), the
  1000-device sample of FauxPV's table with its reference curves (`table_sample/`). The
  validation scripts take those paths as arguments.
- Validation scripts: `compare_simss.py` (one device against SimSS), `table_gate.py` (the §7
  gate), `batch_vs_single.py` (batch against one-device solver and SimSS), `bench.py`
  (throughput), `breakdown.py` (a Newton step part by part).
- On a CPU compare per core: XLA's own threads hurt this solver; for throughput run one process
  per core with `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`.
- The user's rule: where SimSS errs, document it in DIFFERENCES.md and take the physically
  correct answer, with a switch that reproduces SimSS's number.
