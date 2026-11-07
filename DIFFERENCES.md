# Where jaxdd differs from SIMsalabim, and why

jaxdd solves SIMsalabim 5.36's discrete equations (the same grid, the same Scharfetter–Gummel
discretisation in the generalised potentials, the same integral current). Where SIMsalabim's
result is not the solution of those equations, or where SIMsalabim approximates something jaxdd
can compute exactly, jaxdd takes the physically correct answer by default and says so here.
Each difference has a way to reproduce SIMsalabim's number, so a comparison can always tell the
two apart.

## 1 Convergence: SimSS stops early, jaxdd does not

**What.** SimSS iterates Poisson and the two continuity equations in turn (Gummel) with
successive under-relaxation, and stops when the relative change of the densities falls below
`tolDens` (1e-5 by default, `convVar = 1`). A damped iteration that changes slowly passes that
test before it has converged. The sign is SimSS's own `errJ`: the spread of the current across
the device, which a converged steady state has at zero.

**How large.** On 1000 devices without ions drawn from FauxPV's drift-diffusion table
(`validation/table_gate.py`, 2026-11-07), 76 differ from SimSS by more than 1e-3 in the
collection `η = (J_dark − J_light)/J_gen` up to 1.0 V:

- 52 only at the first voltage of the scan (−0.2 V), where SimSS starts from its crude initial
  guess. The worst: a device at `L_diff/d` 0.1 with a 300 nm ETL at 1e18 cm⁻³, where SimSS
  reports **+136 A/m² in the dark at −0.2 V** (± 0.40) — a current a dark diode in reverse
  cannot carry; jaxdd gives −2e-5 A/m². One voltage later SimSS agrees with jaxdd.
- 18 more within three times SimSS's own `errJ` at the voltage in question.
- The remaining 6 (gaps of 1.7–2.8e-3 near 1 V): SimSS rerun with `tolDens` and `tolPois`
  1e-9 lands on jaxdd to 3e-9 and 1e-8. The default-tolerance curve is SimSS's error.

All 76 are at `L_diff/d` ≤ 2, most with a transport layer doped at 1e17–1e18 cm⁻³.

**jaxdd.** Newton's method on all unknowns at once, to a step of 1e-10 (potential in `V_T`,
densities relatively) or to the round-off floor (a step below 1e-6 that has stopped falling).
There is no switch to stop early; to compare with SimSS, run SimSS tighter
(`tolDens = tolPois = 1e-9`), or discard the voltages where SimSS's `errJ` is large.

## 2 The Bernoulli function

**What.** SimSS evaluates `B(x) = x/(eˣ − 1)` by Taylor series of order 1 below |x| = 0.05,
2 below 0.45 and 4 below 1.5 (`Units/DDRoutines.pas`, `B`). The error reaches 2e-4 relative
at the first switch and 8e-4 at the last, and it enters every Scharfetter–Gummel flux.

**How large.** Up to 3e-4 of the current scale at high forward bias on FauxPV's devices
(`validation/compare_simss.py` with and without `--bernoulli=simss`).

**jaxdd.** The exact function by default (`bernoulli`: the full expression, its series only
below |x| = 1e-2 to second order, with finite derivatives everywhere). `bernoulli_name="simss"`
reproduces SimSS's approximation; with it, jaxdd and SimSS agree to 1–2e-6 of the scale.

## 3 The ions' steady profile

**What.** SimSS builds the zero-current ion profile point by point as a product of Bernoulli
ratios `B(±Δ)/B(∓Δ)` (`Calc_Ion_Distribution_Steady_State`), which is `exp(Δ/V_T)` exactly only
with the exact `B`.

**jaxdd.** The Boltzmann profile `exp(±V/V_T)`, normalised to the layer's number of ions with
SimSS's trapezoidal rule. With SimSS's approximate `B`, the two differ by `O(Δ³)` per step; the
largest effect seen is 1.9e-4 of the current scale (1e18 cm⁻³ free ions, doped layers).

## Not differences

Reproduced on purpose, because they are part of the discrete equations SimSS defines:

- **The total length** is the sum of the layers, not the last grid point: SimSS equalises the
  spacing across each interface after building the grid, and `Σh` is then slightly off 1. The
  generation profile is stretched to the layers' sum, and every scale uses it.
- **The current across an interface** is set to the one before it (the current through
  interface traps, none in Stage A).
- SimSS's constants (`q`, `k`, `ε₀` in five digits), not CODATA's.
