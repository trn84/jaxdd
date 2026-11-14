# SimSS, the reference

jaxdd is validated against SimSS 5.36, the steady-state program of
[SIMsalabim](https://github.com/kostergroup/SIMsalabim) (LGPL-3.0), and its transients against
ZimT 5.36, the same project's transient program. This directory builds both:

```bash
docker build -t jaxdd/simss:5.36 simss
```

The image has three binaries:

- `simss`: the release, unchanged. Used for every device except the one below.
- `simss_fixed_ions`: the release with [`fixed_ions.patch`](fixed_ions.patch), for devices with
  ions held: settled at a pre-bias and then frozen (`preCond = 1`, `fixIons = 1`), or held from
  the start where SimSS puts them, each layer's own density (`fixIons = 1` alone). In SimSS 5.36,
  `fixIons` holds the ions in the continuity step, but `Solve_Poisson` still rescales them with
  every potential update, so the "frozen" ions drift with each iteration and the curve depends
  on the voltage step. The patch makes `Solve_Poisson` leave them alone while they are fixed.
  jaxdd solves the frozen case as the patched SimSS defines it.
- `zimt`: ZimT, the release, unchanged: `--entrypoint zimt` with a ZimT `simulation_setup.txt`
  and its `tVGFile`.

`benchmarks/run.py --simss-docker jaxdd/simss:5.36` uses both, choosing per device. The patch is
a modification of SIMsalabim and is under SIMsalabim's licence, LGPL-3.0.
