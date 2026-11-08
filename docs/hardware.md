# Running on CPUs and GPUs

The same code runs on a CPU, an NVIDIA GPU, and (with an older JAX) an AMD MI50. How to get the
most out of each differs. All numbers below are J-V curves of 61 voltages per second, on the
benchmark devices of `validation/bench.py`; the README's figure has every machine measured.

| Machine | jaxdd | SimSS 5.36 on all cores of the same machine |
|---|---|---|
| A100 80 GB PCIe, batch 16384 | ~1200 | — |
| Ryzen 9 5950X (16 cores), CPU | ~360 | ~45 |
| RTX 4070 Ti SUPER, batch 16384 | ~300 | — |
| one CPU core | ~20 | ~0.5 (2 s per curve) |

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
covers thousands of devices. On an A100, batch 4096 gives about 950–1060 curves/s and 16384
about 1160–1200.

**Data-centre cards only, for speed.** jaxdd needs FP64. The A100 and H100 run it at half their
FP32 rate; consumer cards (GeForce, RTX) at 1/64. A RTX 4070 Ti SUPER reaches about 300
curves/s, less than a 16-core desktop CPU. It is fine for development and for checking that
everything runs on a GPU.

### A cloud A100

Any CUDA 12 image with Python ≥ 3.11:

```bash
git clone https://github.com/trn84/jaxdd.git && cd jaxdd
python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -e ".[dev]" "jax[cuda12]"
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # a CudaDevice
.venv/bin/python -m pytest -q
.venv/bin/python validation/bench.py 65536 16384 2>&1 | tee bench_16384.txt
cd validation && ../.venv/bin/python breakdown.py 16384 2>&1 | tee ../breakdown_16384.txt
```

`bench.py N batch` solves `N` variants of the committed test devices in batches and prints
curves per second after compilation. `breakdown.py batch` times each part of one Newton step on
its own (residual, Jacobian blocks, ion columns and constraints, the sweep, the current), which
shows where a step's time goes.

## AMD Instinct MI50 (gfx906)

gfx906 left ROCm's supported list with ROCm 6.3, and from JAX 0.5 on XLA compiles through
Triton, which refuses gfx906. What runs is JAX 0.4.31 in the image
`rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10`, with `ROCM_PATH=/opt/rocm` set. jaxdd
supports JAX ≥ 0.4.30 for this reason; its tests pass on 0.4.31 (with NumPy 1.26), and its
curves agree with the current JAX's to 9e-7 of the current scale.

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
