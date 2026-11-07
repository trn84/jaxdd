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
| M3 batched (`jaxdd.batch.solve_batch`) | built and right on CPU; **A100 gate (≥ 390 devices/s, 50× SimSS on 64 cores) open** |

**A100 results so far** (RunPod, `validation/bench.py`): 143 → 265 → 323 → 341 devices/s
(batch 16384, before the last commit). Commit 5dde971 removed 90 % of a Newton step (the ions'
constraint was differentiated by `jax.jacfwd`; now `jax.jacrev`) and cut the Newton steps by
40 % (extrapolated starts, quadratic stop): one CPU core went 1.8 → 19.2 devices/s. **Not yet
measured on the A100.**

Linear solvers: `thomas` (block sweep; fastest so far everywhere, the default), `pcr` (81
devices/s on the A100: bandwidth-bound), `hybrid` (PCR 3 levels + sweep; 289 against the sweep's
323).

## Next

1. **A100 run of 5dde971** (instructions below): `bench.py` for `thomas` and `hybrid`, and
   `validation/breakdown.py 16384`, which times each part of a Newton step. Expected: past the
   gate.
2. By the breakdown: if the sweep and the Jacobian blocks dominate, a fused kernel for them
   (Pallas, or CUDA through `jax.ffi`) — the user's question of 2026-11-07; the whole Newton step
   as one CUDA/HIP kernel only if jaxdd is to run millions of devices for good.
3. Then M4 (mesh study), M5 (`dd_tables.py --backend jaxdd` in FauxPV: recompute the table
   points SimSS's `errJ` flags, and the M20-00b ion grid), M6 (adaptive axes).
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
