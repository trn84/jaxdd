# Running on CPUs and GPUs

The same code runs on a CPU, an NVIDIA GPU, and (with an older JAX) an AMD MI50. How to get the
most out of each differs. All numbers below are J-V curves of 61 voltages per second, on the
benchmark devices of `validation/bench.py`; the README's figure has every machine measured.

| Machine | jaxdd | SimSS 5.36 on all cores of the same machine |
|---|---|---|
| A100-SXM4-40GB, batch 16384 | ~2750 | — |
| RTX 4070 Ti SUPER, batch 16384 | ~1050 | — |
| Ryzen 9 5950X (16 cores), CPU | ~640 | ~33 |
| one CPU core | ~58 | ~0.5 (2 s per curve) |

The desktop's rates depend on the machine's state as much as on the code: on 2026-11-12 the
same 5950X ran SimSS at 46 and jaxdd at 730, on 2026-11-18 at 33 and 640 with the same SimSS
image and the same benchmark code (the cores at 3.55 GHz under load, the power profile no
matter). The factors in the README are those of one day's measurement, each solver's.

## CPU: one small process per core

XLA's own threads do not help this solver. Its work is many tiny 3×3 operations in a sequential
sweep, and spread over threads they cost more in synchronisation than they gain. The fastest use
of a CPU is **one single-threaded process per core, each with a small batch**:

```bash
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
       XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
taskset -c 0 python my_sweep.py --part 0 &      # one per core, each pinned
```

- **Pin each process to its core.** The flags alone do not hold XLA to one thread: one process
  still used about three cores. `taskset` (or `os.sched_setaffinity` before JAX is imported, as
  `benchmarks/run.py` does) does.
- **Keep the batch per process small (1–4 devices).** Alone, a process runs at 20–27 curves/s
  whatever its batch. With every core busy, its working set must fit its share of the cache. On
  the 5950X with 16 processes, batch 32 gives 43 curves/s in total, batch 8 147, batch 4 313,
  batch 2 349, batch 1 375. Each process is then still at 100 % of its core, but waiting on
  memory.
- **Physical cores, not threads.** 32 processes on the 16-core 5950X were no faster than 16.

## NVIDIA GPUs

Install the CUDA jaxlib (`pip install "jax[cuda12]"`); JAX then uses the GPU by default. Use
large batches: the sweep along the grid is sequential, so a GPU is busy only when each step
covers thousands of devices. On an A100-SXM4-40GB batch 16384 gives about 2750 curves/s
(`benchmarks/run.py` and `validation/bench.py` alike), batch 4096 about 2150.

**Several GPUs in one process.** `solve_batch` splits a batch along its devices across every
GPU the process sees (`shard_map`; the devices are independent, so the GPUs do not talk), each
solving its share; a batch is padded to a whole number per GPU, and the results come back
through the host onto one device, so that nothing after a pass is an operation across the GPUs
(a scatter across them, run by XLA with collectives, hung both GPUs of a host with an old driver). `CUDA_VISIBLE_DEVICES` chooses
the GPUs, `solve_batch(..., devices=[...])` too. The batch is then the whole: 32768 on two GPUs
is 16384 on each. Not split where the batch or the voltages are traced (inside `jax.jit`,
`jax.jvp`, `jax.vmap`), nor in `jaxdd.transient`. On two A100-SXM4-40GB (`STEPS=gpus benchmarks/apptainer.sh`,
2026-11-15) the benchmark runs at 5500 curves/s with 16384 devices per GPU, 2.01× the 2739 of one
GPU, and at 4767 with 8192 per GPU (`benchmarks/run.py`, 2026-11-18: 5435 against 2748, 1.98×); the grid study's small batches gain less (1.24× at 1024
devices per GPU and 800 points, 1.45× at 512 and 1600), a GPU being busy only with thousands.

**Transients** (`validation/transient_bench.py 16384 4096`, `STEPS=transient
benchmarks/apptainer.sh`, 2026-11-17), on one A100-SXM4-40GB (`solve_transient` does not split a
batch across GPUs), 16384 ZimT runs in batches of 4096 on the graded grid, each device's trap,
ion and doping densities drawn at random:

| Workload | Transients/s | Time steps/s | Newton steps per time step: device, batch | One Newton step per device |
|---|---|---|---|---|
| traps, light on and off (241 steps, 3×3) | 576 | 139 000 | 2.32, 2.40 | 3.00 µs |
| mobile ions, a sweep and back (241 steps, 5×5) | 183 | 44 100 | 2.32, 3.14 | 7.22 µs |
| the photovoltage at the open circuit (151 steps) | 975 | 147 000 | 1.88, 2.06 | 3.29 µs |

A time step without ions costs about what a voltage of a J-V curve does at the same batch
(2140 curves of 61 voltages per second at 4096: 131 000 voltages/s). With ions a Newton step
costs 2.4× a 3×3 one, near their blocks' 25 entries against 9; the 5×5 blocks are inverted
elementwise (`solver._inverse5`) and go through the sweep kernels. The lockstep adds 35 %: the
batch pays the slowest device's 3.14 Newton steps per time step where a device needs 2.32. The
internal voltage's border at the open circuit costs 10 % per Newton step.

How the step with ions got there: 13.75 µs with LAPACK's inverse per point in the scan
(2026-11-15), 10.31 with the elementwise inverse in the scan (`JAXDD_SWEEP=scan`), 6.91 through
the kernel. The current in time over control volumes (docs/differences.md §13) and the densities
below the floor (docs/solver.md) then cost 13 % and 4 % more, 8.12 µs, in what every Newton
step formed anew: in the current it computes for its settled-current stop the integral form's
widths, chosen between ZimT's and the control volume by a traced flag (both computed), and in
its step the floor's test through `exp` at every point. Formed once per device, the choice made
when tracing and the test in the logarithm, it is 7.22 µs; the RTX 4070 Ti SUPER (at batch 256
44 µs, 85 in the scan, its FP64 rate the limit) showed none of it.
On that card at batch 64: 41, 4.5 and 69 transients/s with the LAPACK inverse.

**Time a solver from its third call on, or check that nothing compiles.** A benchmark that
excludes the first call excludes one compilation. `jaxdd.batch._solve_batch._cache_size()`
counts the compiled programs: it must not grow between the calls that are timed.

**Data-centre cards only, for speed.** jaxdd needs FP64. The A100 and H100 run it at half their
FP32 rate; consumer cards (GeForce, RTX) at 1/64. A RTX 4070 Ti SUPER reaches about 1050
curves/s, little more than a 16-core desktop CPU (640) and well under half an A100. It is fine
for development and for checking that everything runs on a GPU.

### A cloud A100

On a GPU pod (RunPod or alike: any image with an NVIDIA driver, `git` and `curl`; the image's
Python does not matter) `benchmarks/runpod.sh` does everything. It installs uv if missing, makes
`.venv` with Python 3.12 and current `jax[cuda12]` on the first start, and runs the tests,
`breakdown.py`, `bench.py` (with the Pallas sweep and again with the scan), `hard.py` and
`benchmarks/run.py` in the background, logging to `benchmarks/runpod.log`:

```bash
cd /workspace && git clone https://github.com/trn84/jaxdd.git && cd jaxdd   # /workspace survives a stop
benchmarks/runpod.sh                     # about 25 min on an A100; STEPS="bench scan" for a part
tail -F benchmarks/runpod.log            # to follow it again after Ctrl+C or a new terminal
```

A closed web terminal does not stop it. The CPU part of `run.py` gets the cores the pod's
cgroup grants (`CPU_PROCS` to override): a pod sees all of its host's cores, but may use only
some. The `run` step prints its `benchmarks/results/<label>.json` into the log, so the log
alone carries the result off the pod.

By hand, in any CUDA 12 image with Python ≥ 3.11:

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev,bench]" "jax[cuda12]"
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # a CudaDevice
.venv/bin/python validation/bench.py 65536 16384
```

`bench.py N batch` solves `N` variants of the committed test devices in batches and prints
curves per second after compilation. Those devices are easy; `validation/hard.py N batch` is a
hard workload (frozen ions from 1e15 to 4e18 cm⁻³ at every accumulation, absorbers and transport
layers over wide ranges, light and dark) and prints the Newton steps a batch pays besides the
rate: on 4096 of them one pass pays 5.4 steps per voltage, where the solver before the scan from
the pre-bias and the Jacobian by intervals paid 10.1 (up to 35 at the first voltage).
`breakdown.py batch` times each part of one Newton step on
its own (residual, Jacobian blocks, ion columns and constraints, the sweep, the current), which
shows where a step's time goes.

### A host with an old glibc or driver: Apptainer

A host with glibc 2.17 takes no current JAX wheel, and on NVIDIA driver 530 (CUDA 12.1) the
newest JAX that does install (0.5.3) grew to tens of GB of host memory on its first solve and
was killed; JAX 0.4.31 with the CUDA 12.1 tools runs, at a sixth of the speed. Current JAX runs
on such a host inside an Apptainer container with a current Ubuntu, on the host's driver, as
long as the image's CUDA compat libraries stay out (they need driver 535: `cuInit 803`).
`benchmarks/apptainer.sh` does all of it: it pulls the image, makes the container's venv on the
first start, and runs the tests, `breakdown.py`, `bench.py` and `benchmarks/run.py` in the
background, logging to `benchmarks/apptainer.log`:

```bash
git clone https://github.com/trn84/jaxdd.git ~/jaxdd && cd ~/jaxdd
benchmarks/apptainer.sh                          # STEPS="breakdown bench" for a part
STEPS=all benchmarks/apptainer.sh                # every step of a full measurement, ~40 min
```

`STEPS=all` adds to the default steps the bounded ions on the hard workload (`steric`:
`hard.py` with 1.21e28, 2e27 and 1e26 m⁻³ of sites against Boltzmann's, `STERIC_ARGS`) and the
transients (`transient`: `transient_bench.py`, traps, mobile ions and the open circuit).

`STEPS=sample` runs `validation/sample_measured.py` on the three measured samples instead
(`SAMPLES`, `SAMPLE_ARGS`; hours, not in the default steps), each chain saved to
`benchmarks/samples/` every 100 steps. With `UPLOAD=https://HOST/s/TOKEN`, a Nextcloud or
ownCloud public folder link with upload rights, the host uploads the log and the chains there
every ten minutes and at the end, so a session that ends keeps what is done. The three samples
at the default 128 walkers × 3000 steps took 4.2 h on an A100-SXM4-40GB.

On an A100-SXM4-40GB it runs at 2750 curves/s in the container at batch 16384, all tests
passing; a Newton step of 16384 devices there is 20 ms, 9 of it the assembly and 7 the block
sweep's kernel.

### Why there is no fused kernel

The fastest the A100 could go would be the whole Newton step in one hand-written CUDA kernel
called through the XLA FFI: one thread per device, the blocks assembled in registers where the
sweep uses them instead of written to memory and read back, a mode beside the JAX solver (as
`JAXDD_SWEEP`) with the JAX solver as its reference and fallback. Implicit differentiation
would need no adjoint of it ([solver.md](solver.md)). From the A100's bandwidth it would run
an estimated 10⁴–3·10⁴ curves/s, 4–10× today's 2750 (uncertain by 3×), and the devices would
wait for each other per warp of 32 instead of per batch (on the 250 nm ion grid the batch pays
4.59 Newton steps per voltage where 2.76 are needed).

It was decided against (2026-11-15), for three reasons:

- **No planned use needs it.** Of the uses in [overview.md](overview.md#what-the-speed-is-for)
  only drift-diffusion inside a larger simulation would, and none is planned.
- **The transients with ions are no longer its case.** With the 5×5 sweep kernel a Newton step
  with ions is 2.4× a 3×3 one (it was 4.5×), close to the ratio of their blocks' entries
  (25/9). What is left is the memory traffic every workload shares.
- **Its cost.** The solver's core would be written twice (800–1200 lines of CUDA, the Jacobian
  by hand where JAX takes it by autodiff), and it would need compiled wheels per CUDA (and ROCm)
  version where jaxdd is one pure-Python wheel. Consumer cards would gain little, because their
  FP64 rate is the limit.

It comes back when an application needs 10⁴ curves/s or more, or transients inside another
simulation. Then the first measurement, on the A100, is the Newton step of Stage A's physics
without ions as one kernel against the JAX step. Two cheaper steps come before it, each measured
on the A100:

1. **The blocks' structural zeros.** 24 of their 36 entries per grid point are non-zero (`A`
   and `C` 5 of 9 each, the right-hand sides 5 of 9). A sweep kernel that reads only those 24,
   as separate arrays, saves a third of the blocks' writes and reads. The assembly takes 9.1 of
   the 20 ms of a Newton step at 16 384 devices.
2. **The assembly inside the Pallas sweep kernel** (untried): the blocks are never in memory,
   the residual is still the JAX one traced into the kernel (its derivatives by `jax.jvp`, if
   Pallas lowers them), and no compiled wheels are needed.

## AMD Instinct MI50 (gfx906)

gfx906 left ROCm's supported list with ROCm 6.3, and from JAX 0.5 on XLA compiles through
Triton, which refuses gfx906. What runs is JAX 0.4.31 in the image
`rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10`, with `ROCM_PATH=/opt/rocm` set. jaxdd
supports JAX ≥ 0.4.31 for this reason: the CI runs the tests on JAX 0.4.31 with NumPy 1.26 and
Python 3.11 beside the newest JAX, and the curves agree with the current JAX's to 9e-7 of the
current scale.

```bash
git clone https://github.com/trn84/jaxdd.git && cd jaxdd
rocm/mi50.sh                                  # IMAGE=... rocm/mi50.sh for another ROCm image
```

`rocm/mi50.sh` starts the container and runs `rocm/probe.sh` in it: versions, the cards, a jitted
float64 batch, the tests on the GPU, every card alone, the batch sizes 16 GB hold, the benchmark.
Everything is logged to `rocm/mi50_probe.log`. Fallback images with the same JAX:
`rocm6.1.3-jax0.4.31-py3.11.10`, `rocm6.2.3-jax0.4.31-py3.11.10`. Several cards are used as
several processes, one per card (`HIP_VISIBLE_DEVICES`).

`HSA_OVERRIDE_GFX_VERSION` does not help here: gfx906 already is 9.0.6, and claiming another
architecture runs code compiled for a different instruction set.

## Measuring a machine

`benchmarks/run.py` measures, on the same devices, jaxdd on the GPU (if JAX has one), jaxdd on
every CPU core (one pinned process each), and SimSS on every core (native binaries or a Docker
image), and compares jaxdd's curves with SimSS's:

```bash
.venv/bin/pip install -e ".[dev,bench]"
docker build -t jaxdd/simss:5.36 simss         # SimSS with both binaries (simss/README.md)
.venv/bin/python benchmarks/run.py --label "my machine" --simss-docker jaxdd/simss:5.36
.venv/bin/python benchmarks/run.py --label "my laptop" --simss simss --simss-fixed simss_fixed_ions
.venv/bin/python benchmarks/plot.py           # the README's figure and table
```

SimSS needs the release's `simss` and, for the frozen-ion case, the patched `simss_fixed_ions`;
[`simss/`](../simss/README.md) builds an image with both. For stable
SimSS numbers give it enough runs (`--simss-runs 1024`): the Docker start, about a second, is
counted in its time. Each run writes `benchmarks/results/<label>.json`; commit it to add the
machine to the figure.
