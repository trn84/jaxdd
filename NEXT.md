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

Linear solvers: `thomas` (block sweep; fastest so far everywhere, the default), `pcr` (81
devices/s on the A100: bandwidth-bound), `hybrid` (PCR 3 levels + sweep; 289 against the sweep's
323).

## Next

1. **Choose:** speed further or put it to use.
   - Speed: assemble the Jacobian by edges (each interval's flux once) and in one `jacfwd`,
     estimated 1.5–2× (about a day); a fused kernel only if far more is needed.
   - Use (M5): `jaxdd` as a backend of FauxPV's `tools/dd_tables.py` — the ion grid of M20-00b
     (918 050 devices, ~40 h on the CPU server) in ~13 min on one A100, and the table points
     SimSS's `errJ` flags recomputed.
2. M4 (mesh study), M6 (adaptive axes).
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
.venv/bin/python validation/bench.py 16384 4096 thomas  2>&1 | tee bench_thomas_4096.txt
.venv/bin/python validation/bench.py 65536 16384 thomas 2>&1 | tee bench_thomas_16384.txt
.venv/bin/python validation/bench.py 65536 16384 hybrid 2>&1 | tee bench_hybrid_16384.txt
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
