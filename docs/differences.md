# Where jaxdd differs from SIMsalabim, and why

jaxdd solves SIMsalabim 5.36's discrete equations (the same grid but for §5, the same Scharfetter–Gummel
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

**How large.** On 1000 devices without ions drawn from a parameter table of perovskite cells
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

## 1b Frozen ions: SimSS freezes an unconverged pre-bias, off its ion number

**What.** With `preCond` and `fixIons`, SimSS solves the pre-bias with the ions free, then holds
them for the scan. If the pre-bias does not converge within `maxItSS` (its log: "No
convergence for Vint=" at `Vpre`, the first solve), SimSS goes on and freezes the ions of its
last iteration. Those are off their number: each iteration of `Main_Solver` sets the profile
from the potential and rescales it to `N·L` (`Calc_Ion_Distribution_Steady_State`,
`Rescale_Ion_Density`), then `Solve_Poisson` multiplies the mobile ions by `exp(±ΔV/V_T)` and
does not rescale; only at a fixed point is that factor 1. Nothing in `JV.dat` shows it: `errJ`
is the scan's, and the scan solves the frozen ions cleanly. A tighter `tolDens` does not help.

**How large.** On 48 random points of a parameter grid of perovskite cells with ions
(2026-11-08): at three of them SimSS's frozen anions and cations were off `N·L` by
0.8–6.5e-4, opposite for the two species — at 1e18 cm⁻³ up to 1e-4 C/m² of net ionic charge,
about 0.25 V across a 500 nm absorber — and `η` off by 4e-3 to 3.7e-2 (`J_SC` 6 % at
`L_diff/d` 0.2). Given SimSS's frozen ion numbers (the device's `anion_total`, `cation_total`),
jaxdd reproduces SimSS's curves to 9e-5 of `η`: the whole gap is the ions' number.

**jaxdd.** The pre-bias by Newton with the ions' number as a constraint, held to round-off
(1e-15 of `N·L`); a pre-bias that does not converge is flagged (`converged` false at every
voltage of the scan), never frozen silently. To reproduce SimSS's number, set the device's ion
totals to SimSS's frozen ones.

## 2 The Bernoulli function

**What.** SimSS evaluates `B(x) = x/(eˣ − 1)` by Taylor series of order 1 below |x| = 0.05,
2 below 0.45 and 4 below 1.5 (`Units/DDRoutines.pas`, `B`). The error reaches 2e-4 relative
at the first switch and 8e-4 at the last, and it enters every Scharfetter–Gummel flux.

**How large.** Up to 3e-4 of the current scale at high forward bias on perovskite cells
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

## 4 The start of each voltage

**What.** SIMsalabim starts each voltage of a scan from the last two solutions extrapolated
linearly in the voltage (`Extrapolate_Solution`).

**jaxdd.** The batched solver does the same, but holds the extrapolated change of each unknown
at each point to one Newton step's caps (10 `V_T` in the potential, 5 in `ln n`, `ln p` and the
ion logarithms), and extrapolates only from voltages that converged. A density far below
anything that matters (in the dark, over much of a thick device with a thick doped transport
layer) has an undetermined logarithm, and extrapolating it unclipped put the start e^100 off
and lost every voltage after short circuit ([solver.md](solver.md), "Why the extrapolated
change is clipped"). Where the change is within
the caps, which is everywhere on the benchmark devices and on 1000 devices of a perovskite
parameter table, the start is SIMsalabim's and the curves are bit for bit the same. The
one-device `solve_jv` starts from the last converged solution. This is a difference in the
path, not in the result: a converged voltage is the solution of the same equations either way.

## 5 The grid's share of points: SimSS's rounding on a whole number

**What.** `Make_Grid` gives each layer `FLOOR((NP − n_thin·5)·L/(Ltot − L_thin))` points
beyond its first, at least five. Where that share is a whole number in exact arithmetic — every
layer but one thin, so that `L/(Ltot − L_thin)` is 1, or thicknesses in simple ratios
(100/500/200 nm, 30/240/30 nm) — `FLOOR` is decided by the last bit of the arithmetic. SimSS's
10-byte reals land just below the whole number on some of those stacks and give the layer one
point less: 0.7/350/0.7 nm with `NP` 400 gets 389 absorber points instead of 390, the last
layer one more. float64 lands below it on other stacks. The grid is then one node off at an
interface: the absorber's last node is the next layer's first.

**How large.** Where ions pile up at that interface (frozen at accumulation), the node moves
the peak: on a parameter grid of perovskite cells with ions (2026-11-11), four of 284 sampled
curves that missed SimSS by more than 1e-3 in `η` were this, 1.3–1.9e-3 up to 1.0 V, all on
stacks with both transport layers thin (0.7 and 1.4 nm). With SimSS's split jaxdd met SimSS
there to 4–6e-5. It is a discretisation error of the grid, not of either solver.

**jaxdd.** The share in exact rational arithmetic from the input's decimal strings
(`device.layer_points`, `split="exact"`, the default): the number of points SIMsalabim's
formula means, on every platform. `build_device(setup, grid_split="simss")` repeats SimSS's
operations in its order, each rounded to an 80-bit extended value (independent of the
platform's `long double`); it gave SimSS's points on all eight stacks checked against SimSS's
`Var.dat` (its `lid` column), among them SIMsalabim's own `Tests/test_13`, and agrees with x87
arithmetic on 24 000 random stacks.

## Not differences

Reproduced on purpose, because they are part of the discrete equations SimSS defines:

- **The total length** is the sum of the layers, not the last grid point: SimSS equalises the
  spacing across each interface after building the grid, and `Σh` is then slightly off 1. The
  generation profile is stretched to the layers' sum, and every scale uses it.
- **The current across an interface** is set to the one before it (the current through
  interface traps, none in Stage A).
- SimSS's constants (`q`, `k`, `ε₀` in five digits), not CODATA's.
