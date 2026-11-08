<h1 align="center"><img src="assets/logo/jaxdd-logo-1-bands.svg" alt="JAXDD" width="440"></h1>

A one-dimensional drift-diffusion solver for perovskite solar cells, written in JAX to solve
hundreds of thousands of devices at once on a data-centre GPU (A100/H100). It follows the device
model of [SIMsalabim](https://github.com/kostergroup/SIMsalabim) and is validated against its
output: anyone who sweeps SimSS over a parameter grid can run the same input files in batches.

It reads SIMsalabim's input files and solves one device or a batch of thousands
(`jaxdd.batch.solve_batch`), with ions free or frozen after a pre-bias, on CPU and GPU. Against
SimSS 5.36 it agrees to 1e-6 of the current scale with SimSS's own Bernoulli approximation;
where the two differ by more, it has so far been SimSS's error, documented in
[`docs/differences.md`](docs/differences.md). On one A100 about 1200 J-V curves per second, on
a 16-core desktop CPU about 360: 25× and 8× SimSS on all cores of that desktop.

**How it works and how to use it:** the handbook in [`docs/`](docs/README.md) (overview,
tutorial, model, solver with flowcharts, batching, hardware, validation).
**What is planned:** [`PLAN.md`](PLAN.md).

## Try it

```bash
git clone https://github.com/trn84/jaxdd.git && cd jaxdd
python -m venv .venv && .venv/bin/pip install -e ".[dev]"     # add "jax[cuda12]" on an NVIDIA GPU
.venv/bin/python -m pytest -q                                   # against SimSS's own curves
.venv/bin/python validation/bench.py 64 64                      # J-V curves per second
```

## Speed

<!-- benchmark:begin -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="benchmarks/speedup-dark.svg">
  <img src="benchmarks/speedup-light.svg" alt="J-V curves per second: SimSS and jaxdd on CPU and GPU, per machine">
</picture>

The same devices for every solver (`benchmarks/run.py`: the three committed SimSS cases, traps, ions and doping drawn at random). Run it on a machine of yours and send the JSON in `benchmarks/results/`; `benchmarks/plot.py` redraws this. Factors over SimSS on Ryzen 9 5950X desktop (32 processes, 45.3 curves/s).

| Machine | Solver | Processes / batch | Curves per second | × reference SimSS | jaxdd against SimSS (max ΔJ / scale) |
|---|---|---|---|---|---|
| A100 80GB PCIe (RunPod) (NVIDIA A100 80GB PCIe · 128 logical cores) | jaxdd, GPU | batch 16384 | 1,160 | 25.6 |  |
| Ryzen 9 5950X desktop (NVIDIA GeForce RTX 4070 Ti SUPER · 32 logical cores) | SimSS 5.36, all cores | 32 processes | 45.3 | 1.0 | 1.5e-05 on 48 devices |
| Ryzen 9 5950X desktop (NVIDIA GeForce RTX 4070 Ti SUPER · 32 logical cores) | jaxdd, all CPU cores | 32 processes | 357 | 7.9 |  |
| Ryzen 9 5950X desktop (NVIDIA GeForce RTX 4070 Ti SUPER · 32 logical cores) | jaxdd, GPU | batch 16384 | 302 | 6.7 |  |
<!-- benchmark:end -->

**Measure a machine** (each part runs where the machine has it, the same devices for every
solver; `--help` lists the sizes):

```bash
.venv/bin/pip install -e ".[dev,bench]"                     # "jax[cuda12]" too on an NVIDIA GPU
.venv/bin/python benchmarks/run.py --label "A100 pod"                       # jaxdd: GPU and CPU
docker build -t jaxdd/simss:5.36 simss                                      # SimSS, see simss/
.venv/bin/python benchmarks/run.py --label "my server" --simss-docker jaxdd/simss:5.36
.venv/bin/python benchmarks/run.py --label "laptop" --simss simss --simss-fixed simss_fixed_ions
.venv/bin/python benchmarks/plot.py                                         # figure and table
```

SimSS needs the release's `simss` and, for the frozen-ion case, the patched `simss_fixed_ions`;
[`simss/`](simss/README.md) builds an image with both. On a CPU jaxdd runs one single-threaded
process per core, its fastest use of a CPU ([`docs/hardware.md`](docs/hardware.md)). Each run
writes `benchmarks/results/<label>.json`; commit it to add the machine.

## Licence

GNU LGPL version 3 ([`COPYING.LESSER`](COPYING.LESSER), supplementing the GNU GPL version 3 in
[`COPYING`](COPYING)), as SIMsalabim. What derives from SIMsalabim is listed in
[`NOTICE`](NOTICE).

## Citing

If you use jaxdd, cite SIMsalabim, whose model it implements:

> M. Koopmans, V. M. Le Corre, L. J. A. Koster, *SIMsalabim: An open-source drift-diffusion
> simulator for semiconductor devices*, J. Open Source Softw. 7, 3727 (2022),
> doi:[10.21105/joss.03727](https://doi.org/10.21105/joss.03727)
