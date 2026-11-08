# SimSS, the reference

jaxdd is validated against SimSS 5.36, the steady-state program of
[SIMsalabim](https://github.com/kostergroup/SIMsalabim) (LGPL-3.0). This directory builds it:

```bash
docker build -t jaxdd/simss:5.36 simss
```

The image has two binaries:

- `simss`: the release, unchanged. Used for every device except the one below.
- `simss_fixed_ions`: the release with [`fixed_ions.patch`](fixed_ions.patch), for devices with
  ions settled at a pre-bias and then frozen (`preCond = 1`, `fixIons = 1`). In SimSS 5.36,
  `fixIons` holds the ions in the continuity step, but `Solve_Poisson` still rescales them with
  every potential update, so the "frozen" ions drift with each iteration and the curve depends
  on the voltage step. The patch makes `Solve_Poisson` leave them alone while they are fixed.
  jaxdd solves the frozen case as the patched SimSS defines it.

`benchmarks/run.py --simss-docker jaxdd/simss:5.36` uses both, choosing per device. The patch is
a modification of SIMsalabim and is under SIMsalabim's licence, LGPL-3.0.
