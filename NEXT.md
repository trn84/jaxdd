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

## M5 (2026-11-08): the FauxPV side built, the gate with frozen ions not met

FauxPV's `tools/dd_tables.py --backend jaxdd` (FauxPV task M20-00c): the SimSS input files of
every grid point, read by `read_setup`/`build_device` in worker processes, solved by
`solve_batch` per grid size and voltage axis, into tables of their own. `reproduce --backend
jaxdd` on 48 grid points of `simss_tables.npz` (CPU, 30 s): median `η` gap 1.5e-5, 44 within
1e-3. Of the four that fail, SimSS rerun at `tolDens = tolPois = 1e-9` (FauxPV
`data/drift_diffusion/checks/m20_00c_simss_tight.{py,json}`):

| FauxPV grid point | `ℓ` | `N_ion` cm⁻³ | `x_ion` | ETL, HTL (nm, cm⁻³) | jaxdd − SimSS 1e-9, `η` | where |
|---|---|---|---|---|---|---|
| p1_4_2_0_5_0_3 | 0.2 | 3e16 | 0.5 | 1/1e18, 1/1e17 | 5.4e-6 | SimSS's early stop (DIFFERENCES §1) |
| p0_10_1_3_4_2_3 | 0.1 | 1e18 | 0.25 | 100/3e17, 30/1e17 | **8.1e-3** | towards 1 V, light and dark (6.9e-3, 7.0e-3) |
| p1_10_4_1_2_4_2 | 0.2 | 1e18 | 1 | 10/1e16, 300/1e16 | **3.7e-2** | light only, from reverse bias: J_SC −8.41 vs −8.94 A/m²; dark within 3.5e-5 |
| p2_8_2_3_0_4_2 | 0.5 | 3e17 | 0.5 | 100/0, 300/1e16 | **4.0e-3** | towards 1 V, light 3.6e-3, dark 4.1e-4 |

`ℓ` is `L_diff/d` at short circuit; layers on FauxPV's axes (the absorber's permittivity, 500 nm
absorber). jaxdd converged at every voltage and gives the same with `strict=True`, with
`bernoulli_name="simss"` and with `max_iter=300`; SimSS at 1e-9 has `errJ` ≤ 8e-6 of `J_gen` and
skips nothing. So jaxdd and the converged SimSS solve something different there — every case
with ions frozen after a pre-bias, the worst the hard case of PLAN.md §10. The inputs: FauxPV
`m20_00c_simss_tight.py write` writes them (to the temp dir).

## Next

1. **The three frozen-ion cases above** — the gate of §7 for frozen ions, which until now was
   checked on the reference cases only, not on 1000 devices. Then FauxPV's `reproduce --backend
   jaxdd` again, and a sample with ions at `ℓ` ≤ 0.5. Only then the M20-00b ion grid on the A100
   (its SimSS run, still going, is the reference for it).
2. Speed, after that: assemble the Jacobian by edges (each interval's flux once) and in one
   `jacfwd`, estimated 1.5–2× (about a day); a fused kernel only if far more is needed. On a
   real grid the host may set the pace first: writing and reading SimSS's files is ~0.1 s per
   point, and hard devices hold a batch in lockstep (M0: the slowest tenth is 74 % of SimSS's
   time).
3. M4 (mesh study), M6 (adaptive axes).
3. Proposal PLAN.md §8b, for FauxPV and jaxdd together: a steric (Bikerman) limit on ion
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
