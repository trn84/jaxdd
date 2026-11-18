# How jaxdd is validated

The reference is SimSS's own output. jaxdd solves SimSS's discrete equations, so a converged
jaxdd solution and a converged SimSS solution of the same input must agree to round-off (with
the same Bernoulli function) or to the size of the documented differences (without). A gap
beyond that is a bug in one of the two, and the work is to find out which.

## The criteria

| Check | Criterion |
|---|---|
| Ion-free devices | `η` within 1e-3 of SimSS at every voltage up to 1.0 V, `J_SC` within 0.1 %, on a random 1000 devices of a perovskite parameter table |
| Frozen ions | the same on devices with ions, including 1e18 cm⁻³ at `L_diff/d` 0.1 |
| Free ions | SimSS's unpatched steady state |
| Failures | where SimSS skips a voltage, jaxdd either converges (and is checked against a SimSS run at tighter tolerances or a finer grid) or flags it; never a silent wrong curve |
| Invariants | the current constant across the device; zero current and flat quasi-Fermi levels at 0 V in the dark; `J_SC` never above the generation current |

The 1e-3 in `η` is chosen well below the interpolation error of a table built from such curves,
so that the solver is never the larger error in what is computed from it.

## What the comparisons show

- **Without ions**, with SimSS's Bernoulli approximation (`bernoulli_name="simss"`), jaxdd and
  SimSS agree to 1–2e-6 of the current scale; with SimSS converged tighter, to 1e-8. With the
  exact Bernoulli function the gap is up to 3e-4, which is SimSS's approximation
  ([differences.md](differences.md) §2).
- **With ions** (free, and frozen after a pre-bias), within 2e-5.
- **On 1000 devices of a perovskite parameter table** every device beyond 1e-3 was SimSS stopping early at its
  default tolerance, most often at the first voltage of a scan; SimSS rerun with
  `tolDens = tolPois = 1e-9` lands on jaxdd ([differences.md](differences.md) §1).
- **On grid points with frozen ions**, the remaining gaps were SimSS freezing a pre-bias
  that had not converged, off its number of ions; given SimSS's ion numbers, jaxdd reproduces
  its curves ([differences.md](differences.md) §1b).
- **On a large parameter grid of perovskite cells with ions** (2026-11-11): 367 200 devices on
  a 500 nm absorber and 915 000 on four more (250–1000 nm; `L_diff/d` 0.1–30, ion densities
  1e15–1e18 cm⁻³ frozen at five accumulations, transport layers 1–300 nm doped 0–1e18 cm⁻³),
  light and dark, solved by the batched solver on an A100 and compared curve by curve with
  SimSS's run of the same inputs. The median gap in `η` is 2–3e-6; 10.1 % and 8.0 % of the
  curves miss by more than 1e-3 somewhere up to 1.0 V, nine in ten at `L_diff/d` ≤ 2 and two
  thirds or more from 2e17 cm⁻³ of ions. A stratified sample of 284 of those misses (by grid, `L_diff/d`, ion
  density and size of the gap) was rerun with SimSS at its default tolerances, which gave the
  stored curves back to 1.5e-7, and at `tolDens = tolPois = 1e-9`: 145 were SimSS stopping
  early (the tight run within 1e-3 of jaxdd), 135 SimSS freezing a pre-bias that did not
  converge, at the tight tolerances too (§1b), and 4 SimSS's share of the grid points (§5).
  None was jaxdd's.
- **jaxdd's own failures on that grid**, with its transport layers extended to 1 µm (4.7 M
  devices): every curve with an unconverged voltage (2260) solved again with the clipped scan
  start ([solver.md](solver.md)) kept 5019 of its 45 051, under 2e-5 of all voltages, each
  flagged; where both runs converged the curves agree to 4e-5 in `η`. The rest were all in the
  dark, from 0.775 V up, with ions frozen at full accumulation and at least 5e17 cm⁻³ on the
  device: the round-off floor of the residual assembled row by row. Assembled by intervals
  ([solver.md](solver.md)), a sample of 16 384 devices of the 250 nm grid (light and dark) has
  no unconverged voltage, the six hardest converge without the shortcuts too, and the curves
  agree with the row form's to 9e-6 of the generation current where both converged.
- **The batched solver** matches the one-device solver to 1e-10, and its shortcuts change `J` by
  at most 1.2e-10 of the generation current ([solver.md](solver.md)).

- **Interface traps** (2026-11-13): a donor and an acceptor level of 3e14 m⁻² at the two
  interfaces of the 1 nm stack, within 1.5e-6 of SimSS's scale; the same with free ions, 5.4e-6;
  a p-i-n stack with 10 and 20 nm transport layers and 5e13 and 4.6e14 m⁻² of donor traps,
  4e-7 against SimSS at tolerances of 1e-9. At its default tolerances SimSS stopped early there
  by 1.2e-4, and from 1.0 V on it does not converge at either; jaxdd converges to 1.25 V. The
  traps move these curves by up to two thirds of their scale.
- **Charged bulk traps and trap files** (2026-11-14): the 1 nm stack's absorber traps (3.9e21
  m⁻³ at mid-gap) as acceptors, and as three donor levels from a trap file with two acceptor
  levels from an interface trap file at the left interface and one donor level at the right;
  within 2.5e-10 and 1.0e-10 of SimSS's scale at its tolerances of 1e-9, at most six Newton steps
  per voltage. The traps' charge moves these curves by 4–5 % of their scale.
- **Finite surface recombination** (2026-11-14): 20 m/s for the electrons at the left contact
  and 50 m/s for the holes at the right one on the 1 nm stack, within 5e-12 of SimSS's scale at
  its tolerances of 1e-9, at every voltage but 1.05 V, where SimSS takes the contact's branch
  from the voltage before ([differences.md](differences.md) §7). At the other two ends, where
  SimSS 5.36 has the sign of `S` wrong, the device turned around gives the same curve to 1e-11.
- **Langevin recombination** (2026-11-14): in all three layers of the 1 nm stack, moving the
  curve by 2 % of its scale: within 1.4e-10 of SimSS with its mobilities, 3.7e-5 with each
  point's own layer's, the default ([differences.md](differences.md) §8).
- **The field-dependent mobility** (2026-11-14): on the 1 nm stack, γ from 0 to 3e-4 (m/V)^½
  for electrons and holes in the three layers, the electrons' constant in the last, moving the
  curve by 32 % of its scale: within 1.1e-10 of SimSS's scale at its tolerances of 1e-9, in at
  most 16 Newton steps per voltage, as without. With Langevin recombination in all three layers
  as well (the rate in the field of its interval, moving the curve by another 2.5 %): within
  1.5e-10 with SimSS's mobilities, 6.2e-5 with each point's own layer's
  ([differences.md](differences.md) §8).
- **Field-dependent generation** (2026-11-14): Braun's dissociation in all three layers of the
  1 nm stack, a Gaussian distance distribution in the absorber (`P0` 0.1), an exponential and a
  delta function in the transport layers, which halves J_sc: within 2.1e-10 of SimSS's scale
  with its Romberg integration and Bessel series, 6.1e-9 with the exact ones, the default
  ([differences.md](differences.md) §9), at most 16 Newton steps per voltage.
- **Series and shunt resistance** (2026-11-13): 2e-4 and 0.3 Ω m² on the 1 nm stack, the
  series resistance moving the voltage by 1.8 V at the top of the scan: `J_ext` within 2.2e-6 of
  SimSS's scale, `V_ext` within 4 µV.
- **The transfer-matrix profile** (2026-11-13): on the 1 nm stack between glass, 110 nm ITO
  and gold, and on a 20/500/10 nm stack between glass, 150 nm ITO and silver, the profile on the
  grid is SimSS's within 4e-12 and the curves within 2e-7 and 6e-7 of SimSS's scale; a uniform
  profile (`genProfile = none`) within 2.4e-6.
- **The current in time** (2026-11-15): ZimT's integral form sums each density's change over
  the interval after its point; summed over the point's control volume
  (jaxdd's default, [differences.md](differences.md) §13) it is the differential form's current
  of the same solution to 5e-11 of the scale at 400, 1600 and 6400 points (light on and off
  with and without traps and with finite surface recombination, sweeps with and without
  interface traps and behind a series resistance; the photovoltage at the open circuit to
  3.5e-12 of its excursion), and with mobile ions (`zimt_ions_fast_light`,
  `zimt_ions_slow_light`, `zimt_region_slow_light`) to 8.5e-8, 3.1e-10 and 2.7e-11 at 6400
  points, where ZimT's sums are 4.8e-6, 2.6e-8 and 2.7e-9 off (at fewer points the ions' own
  error in space is the larger with either: 6.0e-4, 3.4e-4 and 1.2e-4 at 400). Against the differential form
  on the graded grid at 6400 points ZimT's sums are, at 400 / 1600 / 6400 graded points,
  2.0e-3 / 6.0e-4 / 1.9e-4 of the scale off through the light on and off, 3.0e-2 / 7.8e-3 /
  2.1e-3 with surface recombination and 3.9e-3 / 7.9e-4 / 1.9e-4 through the sweep, where the
  default is 5.4e-5 / 1.9e-6, 4.0e-5 / 2.4e-6 and 9.2e-6 / 5.7e-7 off (the sweep with charged
  interface traps: 3.0e-3 / 6.2e-4 / 1.5e-4 against 5.1e-6 / 4.1e-7); on SIMsalabim's grid
  2.8e-3 / 1.5e-3 / 3.6e-4, 2.7e-2 / 7.6e-3 / 2.2e-3 and 2.9e-3 / 5.9e-4 / 1.6e-4 against
  8.3e-4 / 9.4e-4 / 1.9e-4, 8.5e-4 / 9.6e-4 / 1.9e-4 and 2.5e-4 / 1.9e-4 / 3.2e-5. The
  comparisons with ZimT below take ZimT's sums (`zimt_current=True`).
- **The grid on hard devices** (2026-11-15, `validation/grid.py` on an A100): `hard.py`'s 4096
  devices (frozen ions up to 4e18 cm⁻³, absorbers of 250–1000 nm, transport layers of 1–300 nm),
  each in the light and the dark, against the criteria (η within 1e-3 up to 1.0 V, `J_SC` within
  0.1 %), on both grids against the graded one at 3200 points:

  | `NP` | graded: meet both | η: median, 99 %, max | `J_SC`: median, max | SIMsalabim's: meet both | η median |
  |---|---|---|---|---|---|
  | 200 | 89.3 % | 3.4e-4, 2.1e-3, 6.5e-3 | 2.3e-4, 4.0e-2 | 4.6 % | 3.9e-3 |
  | 300 | 98.1 % | 1.2e-4, 1.0e-3, 2.9e-3 | 7.8e-5, 1.6e-2 | 11.7 % | 2.1e-3 |
  | 400 | 99.6 % | 6.0e-5, 6.0e-4, 1.7e-3 | 3.7e-5, 8.1e-3 | 27.4 % | 1.5e-3 |
  | 800 | 100 % | 1.2e-5, 1.8e-4, 4.6e-4 | 6.9e-6, 1.9e-3 | 71.1 % | 6.0e-4 |
  | 1600 | 100 % | 2.3e-6, 5.2e-5, 1.4e-4 | 1.3e-6, 4.4e-4 | 88.3 % | 2.6e-4 |
  | 3200 | | | | 95.0 % | 1.2e-4 |

  The graded grid's error falls by about 4 per doubling; SIMsalabim's by 2–3, and its own 3200
  points are up to 9e-3 in η off: it keeps a layer whose share is below five points at five at
  any `NP` (differences.md §12), and the devices with a 1 nm transport layer never converge on
  it. Both converge to the same curves (SIMsalabim's grid with `grad = 8` at 3200 points within
  1e-5 of the graded one in the median, 512 devices). One voltage of 499 712 did not converge on
  the graded grid at 1600 and at 3200 points, none at fewer. The same on 512 of the devices on
  the RTX 4070 Ti SUPER: 99.8 % at 400 points; the one device that misses is 30/1000/300 nm with
  a diffusion length far below the absorber and accumulated ions, its `J_SC` off by 2.1e-3. The
  grid's parameters are a choice on those 512 (β 2–16 and `d₀` 0.3 pm–0.1 nm tried: 93–99.8 %
  at 400 points; the first attempt, boundary layers a few Debye lengths of the doping and ions
  wide, met them on 22 % at most): the hard devices' worst errors were at interfaces with
  accumulated ions, whose fields make the densities change on picometres. With the same points it costs Newton nothing
  more: 2.82 Newton steps per voltage on both grids at 400 points, the batch paying 4.30.

  The committed SimSS cases (25, `tests/data`) on both grids against the graded one at 3200
  points: the graded one at 400 points within 1e-5–3e-4 of the curve's scale on all but one;
  SIMsalabim's at 3200 within 3e-5–1e-4 where the transport layers are thick (the `pin_` and
  `region_` cases), 7e-4–1.6e-3 where they are 1 nm (its five points). The one,
  `frozen_1e17_x1_thin_light` (ions frozen at full accumulation, 1 nm layers), converges slower
  on both: up to 1.0 V the graded grid at 400 points is within 6e-4 of `J_SC` (3.3e-3 of the
  scale at 1.3 V, in high injection), SIMsalabim's within 2e-2 and not monotonically (800 points
  better than 1600); at 6400 points the two are 1.7e-4 apart. In time (ZimT's cases, against
  the differential current on the graded grid at 6400 points), the sweep with mobile ions
  (`zimt_ions_fast_light`): graded 400 points 6.0e-4 of the scale, SIMsalabim's 4.0e-3 at 400
  and 2.2e-3 at 3200; the light on and off with traps: 5.2e-5, 8.2e-4 and 6.6e-4; the same
  Newton steps per time step on both.
- **Mobile ions in time** (2026-11-15, Stage C): sweeps to a voltage and back under light with
  free ions, against ZimT 5.36 at tolerances of 1e-9: ions in the absorber of
  `steady_1e17_thin_light` over 2 × 5 s to 1.2 V (`tests/data/zimt_ions_slow_light`) and over
  2 × 50 ms to 0.9 V (`zimt_ions_fast_light`), ions in the region of two layers of
  `region_etl_light` over 2 × 5 s to 0.9 V (`zimt_region_slow_light`) and with the
  differential current over 2 × 50 ms (`zimt_region_diff_light`): within 1.8e-9, 4.0e-8,
  3.3e-9 and 3.8e-9 of ZimT's scale, where the hysteresis is 9.5e-2, 4.2e-2, 2.5e-2 and about
  1e-2 of the scale; in one batch with devices without ions, which keep their accuracy, at
  3–4 Newton steps per time step. Beyond about 0.93 V, near the open circuit, ZimT's Gummel
  iteration stops converging with ions (its densities moving by up to 99 % at its limit of
  2000 iterations), at 1e-9 and at 1e-12 alike and whatever its acceleration; jaxdd agrees
  with it up to its first unconverged step, after which ZimT carries unconverged states.
- **The open circuit and series resistance in time** (2026-11-15, Stage C), the internal voltage
  an unknown of each step's Newton solve, against ZimT 5.36 at `tolPois = tolDens = tolVint =
  1e-12`: the photovoltage of `thin_l1_light` at the open circuit through a pulse of 1.2 suns
  over 1 µs and 1 ms after it at 1 sun (`tests/data/zimt_tpv_light`, 151 steps, 1.0283 V to
  1.0368 V) within 7.4e-12 V, at 1.8 Newton steps per time step (13 for the first, from 0 V);
  `resist_light` (`R_series = 2e-4`, `R_shunt = 0.3` Ω m²) swept to 1.2 V and back in 2 × 100 ns
  (`zimt_rseries_sweep_light`, `Vint` up to 0.65 V from `Vext`): `Jext` within 1.3e-10 of
  ZimT's scale and `Vint` within 4.1e-11 V, at 2.7 Newton steps per time step, where ZimT runs
  Brent on whole solves at each step. The open circuit at `t = 0` is SimSS 5.36's `Voc`
  (1.0282617 V; SimSS at 1e-12 on 0.1 mV steps around it, a cubic through them,
  `thin_l1_voc_light`) to 1.1e-13 V.
- **Gradients through time** (2026-11-15, Stage C): the tangents of a transient in its
  parameters against central differences (a step of 1e-3 in the logarithm): the absorber's trap
  density through the light switched on within 2.7e-7 of their largest, the holes' capture in
  the photovoltage at the open circuit within 8e-6 (`Vint`'s Newton tolerance in the
  differences), the anions' mobility through a sweep within 6e-6 (at a step of 1e-4). Fits of a
  transient from its own curve, every parameter started at twice or half its value
  (`validation/fit_transient.py`, 60 steps, RTX 4070 Ti SUPER): the trap density, the
  electrons' capture and the holes' mobility from the light switched on, to 7e-9, in 12
  Jacobians and 11 s; the trap density and the holes' capture from the photovoltage, to 4e-13;
  the anions' mobility and density from the sweep's hysteresis, to 7e-10, in 7 Jacobians and
  40 s (compiling 14–42 s more). The direct recombination is not determined by the
  photovoltage of a device whose traps dominate (a fit took it to 0 at a misfit of 5e-8 V).
- **ZimT's output, its automatic stop and failures** (2026-11-15, Stage C). Every column of the
  `tJFile` of the fourteen ZimT cases in `tests/data`: the currents and the recombination as
  currents (per layer, at each interface, the minority carriers' at the contacts, the ions',
  the displacement current) within 1e-7 of the scale, `errJ` within 3e-6, with
  `specialOutput = spv` each layer's voltage drop within 1e-9 V. The QFLS within 5e-11 V under
  light; within 3.5e-6 V in the dark after the light, where ZimT's Gummel overshoots and comes
  back (at 2 ns after the light ZimT's falls to −7.1e-7 V and returns to −2.37e-7 V, where
  jaxdd's falls monotonically to the same −2.37e-7 V); with mobile ions within 3.2e-7 V over
  the time steps and 7e-7 V in the steady state at t = 0. In two of ZimT's steady states at
  t = 0 its minority carriers in a transport layer are not converged: with interface traps the
  first layer's QFLS is 2.1e-2 V above jaxdd's, 0.8684 V where one step later, 0.8 ns and 10 mV
  on, ZimT's own is 0.8479 V and jaxdd's 0.8479 V; with a trap file 1.8e-5 V off in the dark at
  0 V (jaxdd's −3.5e-7 V, the equilibrium's 0 to SIMsalabim's Bernoulli function). jaxdd's
  values there do not change with Newton's floor 1e-18 times lower. Before the densities below
  that floor were stepped in the density (docs/solver.md, "When is a voltage converged?") the
  QFLS was up to 5e-3 V off in the dark after the light and 9e-4 V with surface recombination.
  The automatic stop on the photovoltage after a pulse, against the steady state before it
  (`Track` 2) and step by step (`Track` 3): ZimT stops after 118 and 119 of 182 steps, jaxdd a
  step before each, where the states' difference crosses `tolPois` (1e-12 V) and ZimT's own
  error, about as large, decides; `Vint` within 7.4e-12 V up to there. `failureMode` 0, 1 and
  2 on steps made to fail. Two ion regions per species (`zimt_regions2_light`): ZimT keeps
  only the last region's current (differences.md §11); with `zimt_ion_currents` jaxdd's
  `Jext`, `Jnion`, `Jpion` within 1e-8 of the scale.
- **Traps in time** (2026-11-15, Stage C): light on and off at short circuit with the
  absorber's traps neutral (`tests/data/zimt_traps_photo_light`, `thin_l1_light` itself), charged
  as acceptors (`zimt_acceptor_photo_light`) and three levels from a trap file with two at an
  interface (`zimt_trapfiles_photo_light`), and a sweep to 1.2 V and back with charged interface
  traps (`zimt_iface_sweep_light`), against ZimT 5.36 at tolerances of 1e-12: within 1.2e-10,
  4.1e-10, 1.1e-10 and 3.7e-11 of ZimT's scale, single and in one batch with the devices without
  traps. Traps that followed their steady state at once would be 5.3e-3, 5.9e-2, 4.3e-2 and
  (the interface traps alone) 3.5e-2 of the scale off: the occupations' dynamics are what these
  cases test. At short circuit the interface traps' dynamics move the current by 5e-6 only,
  hence the sweep for them.
- **Transients of electrons and holes** (2026-11-15, Stage C): `thin_l1_light` without its
  traps, against ZimT 5.36 at tolerances of 1e-12 (at 1e-9 ZimT itself moves by 1.7e-7 of the
  scale in the first picoseconds), 240 steps each: light on and off at short circuit over six
  decades of time (`tests/data/zimt_photo_light`, and with finite surface recombination
  `zimt_photo_surface_light`), a sweep to 1.2 V and back in 2 × 100 ns whose displacement
  current is half the scale (`zimt_sweep_light`, `zimt_sweep_diff_light` with the differential
  current): within 1.1e-10, 3.1e-10, 1.0e-10 and 1.5e-11 of ZimT's scale, and as one batch,
  each device with its own steps, within the same.
- **ZimT's input and its steady states** (2026-11-15): a ZimT 5.36 input (`tests/data/
  zimt_steady_light`, `thin_l1_light` in ZimT's format) read with its `tVGFile` (`read_tvg`, ZimT's
  checks) and built into a device; its seven steady states (`t = 0` on every line, at two light
  intensities) within 1.6e-10 of ZimT's scale by the steady-state solver: ZimT's `t = 0` is
  SimSS's steady state.
- **Two mobile regions of a species** (2026-11-14): anions and cations mobile in both
  transport layers of a p-i-n cell, the absorber closed to them (`tests/data/two_regions_light`),
  each region its own normalisation: within 6.8e-7 of SimSS's scale where it converged, by both
  solvers, in a batch beside a device of one region per species, which keeps its curve to
  4.1e-7; the gradient in one region's ion density against finite differences.
- **Ions across layers and held from the start** (2026-11-14): one ionic region of two layers,
  free ions entering the 20 nm ETL of a p-i-n cell (`tests/data/region_etl_light`) and ions
  entering its 10 nm HTL, frozen after a pre-bias (`region_htl_frozen_light`): within 1.7e-7
  and 3.9e-6 of SimSS's scale where it converged, by both solvers; the region moves the curve
  by 0.7 % of the scale. Ions held from the start (`fixIons` without `preCond`, a net charge,
  `held_thin_light`): within 8.1e-11 of the patched SimSS (the unpatched one 40 % off).
- **SimSS's voltages and its differential current** (2026-11-14): `simss_voltages` gives the
  voltages of every SimSS run in `tests/data`, a logarithmic downward scan (`Vdist = 2`,
  `Vscan = -1`) and `expJV`'s with `useExpData`; on the logarithmic scan the curve is within
  2.0e-6 of SimSS's scale. With `currDiffInt = 1` and SimSS at tolerances of 1e-9 the curves are
  within 2.5e-11 (`thin_l1_light`) and 1.7e-11 (with interface traps) of SimSS's, 1.2e-9 in a
  batch beside devices with the integral form; the anode-left p-i-n cell with free ions within
  6.9e-6, as with the integral form.
- **The left electrode as the anode** (`leftElec = 1`, 2026-11-13): a p-i-n cell in its own
  orientation (glass, 150 nm ITO, 10 nm TaTm, 500 nm perovskite with free ions, 20 nm C60,
  100 nm Ag), interface traps at both interfaces, the transfer-matrix profile and a shunt: within
  6.9e-6 of SimSS's scale at its tolerances of 1e-9, the short-circuit current to ten digits.
- **Gradients** agree with central finite differences and between forward and reverse mode,
  in the batch's arrays, the voltages and the input's own parameters (mobility, thickness, band
  edge, ion density, temperature, work function), without ions, with free and with frozen ions
  (tests; [solver.md](solver.md#derivatives)).
- **A fit** (`validation/fit.py`, 2026-11-13): the absorber's trap density, direct
  recombination and both mobilities and the left work function, started 1.4–1.7× (and 0.025 eV)
  off, fitted to SimSS's curve of `thin_l1_light` by least squares with the Jacobian from
  `jax.jacfwd`: 25 curves and 19 Jacobians, 17 s on the RTX 4070 Ti SUPER. The trap density comes
  back to 4e-5, the mobilities to 3e-4, the work function to 2e-6 eV, `k_direct` to 4 %: the
  curve barely determines it (its singular value is 2.5e-6 of the largest), and the misfit there
  (2.1e-7 of the scale) is already below the one at the true parameters (3.2e-7, jaxdd against
  SimSS). With the ions of `frozen_1e17_x1_thin_light` fitted too, the ion density comes back to
  7e-4, the rest as well. What a single light curve does not settle: from twice as far off (3×,
  0.05 eV) the fit stops in another minimum, 2.8e-4 of the scale off (about 0.07 A/m²), with
  `k_direct` 38× too high and the electron mobility 2.9×; and with noise of 1e-3 of the scale
  `k_direct` runs to its bound and the others land 20–160 % off, 5–10 times their linearised
  standard errors. A curve determines the trap-assisted recombination and the transport well,
  the direct recombination hardly; a fit of real data wants several light intensities or a dark
  curve, several starts, and its uncertainty by sampling rather than by linearisation.

- **A fit to measured curves** (`validation/fit_measured.py`, 2026-11-13): the five light
  intensities of each of the three samples of Heester et al. (2025; `validation/data/heester2025`)
  at once, the paper's free parameters within its bounds plus the generation's scale, from the
  middle of the bounds. The device is the paper's stack with interface traps; what the paper does
  not state is assumed (uniform generation instead of its optics, the interface traps' energy,
  capture coefficients and type, the transport layers' outer bands), and the intensities are the
  measured short-circuit currents' ratios (0.126 where the files say 0.10, 0.357 for 0.38). The
  misfit, rms over the 1-sun short-circuit current: 0.77 % (0.1 Cl), 0.55 % (0.2 Cl), 2.7 %
  (0.6 Cl; 2.4 % from the paper's values), against a spread of about 2 % between pixels of one
  sample; the 0.2 Cl sample's curves are matched within 0.2–0.7 A/m² on average at 187 A/m².
  Cold and warm starts end in the same fit for 0.1 and 0.2 Cl. On the RTX 4070 Ti SUPER a fit is
  2–10 minutes, 100–400 evaluations of the five curves and as many Jacobians (the paper's
  Bayesian optimisation ran about 25 000 simulations per sample).
  What the curves determine: the
  Jacobian's singular values spread over seven to nine decades, three to seven of the fourteen
  parameters end at a bound, and `k_direct` ends within a factor of three of its lower bound in
  all six fits (two starts per sample), so
  that with these assumptions the curves settle about three to five combinations of the
  parameters, not the parameters one by one — the paper's own point. The 0.6 Cl sample, with the
  largest band step to the ETL in the paper's fit, is the one the assumptions do not reach.
  With the cell in its own orientation and the transfer-matrix profile instead
  (`--device anode`), the misfits are no better. With the paper's n,k data for the perovskite
  (its supporting information names Werner et al., ACS Energy Lett. 3, 742 (2018); of their
  Table S4 the composition at the samples' band gap of 1.65–1.68 eV, CsBr10%-1:2, read from a
  file by `--nk`, the data not redistributable) they are 0.81, 0.75 and 2.44 %, with SIMsalabim's
  `nk_peroTripleCat` as a stand-in 0.86, 0.75 and 2.4 %; the generation's scale ends at 1.04,
  0.99 and 1.04 (the stack's 188 A/m² in the absorber at 1 sun against 187–196 A/m² measured,
  the stand-in's 210). The back electrode is Ag in the paper's text and Al in its supporting
  information; Al (SIMsalabim's `nk_Al`, `--back`) generates 177 A/m², the scale ends 5 % higher
  and the misfits are the same to 1e-4. The fitted parameters move by up to a factor of 20
  between a uniform and a computed generation (0.2 Cl: the ETL interface's traps 1.1e14 against
  3.2e13 m⁻² with Werner's data, 2.4e13 with the stand-in's), little between the two n,k data.
  What these curves say about a single parameter depends on the unstated assumptions as much as
  on the data.
- **Many starts at once** (`fit_measured.py --starts 16`, `jaxdd.fit.least_squares_many`,
  2026-11-14): the same fit from 16 starts (the cold one, the paper's values and 14 of a Sobol
  sequence in the box), every start scipy's `least_squares` as alone, their curves and
  Jacobians solved as one batch: 10.5 min per sample on the RTX 4070 Ti SUPER (400 batched
  evaluations of 16 × 5 curves and 14 tangents, 1.6 s each), the time of two fits from one
  start. The starts end in 6–10 minima. For 0.2 Cl the best is the cold start's (rms 0.553 %,
  6 starts). For 0.1 and 0.6 Cl it is lower than the cold and the paper's start find, in
  another part of the box: 0.701 % against 0.773 % (4 starts; the absorber's electron mobility
  2e-5 against 1e-3 m²/Vs, the ions 1e22 against 1e21 m⁻³) and 1.93 % against 2.69 % and 2.42 %
  (2 starts; the cations at their bound of 1e23 m⁻³, the holes' mobility at 1e-6). With the
  sampler's σ the lower minima are e⁵⁹ and e¹⁵⁸ more likely, and the posteriors below are
  sampled around them. 64 starts for 0.6 Cl (24 min, 1.6× the time of 16 per evaluation for 4×
  the starts) find the same best, from 8 starts.
  With the computed generation and Werner's n,k data (`--device anode --nk …`, 2026-11-15,
  12 min per sample) the best of 16 starts are 0.658, 0.552 and 1.96 %, where the uniform
  generation's are 0.701, 0.553 and 1.93 %: the optics do not make the curves easier to fit.
  The best minima lie elsewhere in the box again (0.6 Cl: the cations at their bound of
  1e23 m⁻³, the HTL interface's traps 4.5e14 m⁻²).

- **The fit's uncertainty, by sampling** (`validation/sample_measured.py`, 2026-11-14): a posterior
  of the same fourteen parameters, by an ensemble sampler whose walkers' five intensities are
  solved as one batch (differential evolution, the walkers started around each sample's best
  minimum: for 0.2 Cl the cold start's, for 0.1 and 0.6 Cl the best of 16 starts, above,
  `--from-multistart`; the prior uniform in the fitted coordinates within the bounds, the
  likelihood Gaussian with σ the fit's rms misfit). Two independent runs per sample (seeds 0
  and 1). 0.2 Cl: on an A100-SXM4-40GB, 128 walkers and 3000 steps, the first 1000 discarded, 77
  and 75 min (0.77 and 0.75 s per batch of 64 × 5 curves), acceptance 0.05, autocorrelation
  times 127–199 steps, 1338 and 1287 independent samples.
  0.1 and 0.6 Cl: on the RTX 4070 Ti SUPER, 64 walkers
  and 3000 steps, the first 1000 discarded (1.7–1.9 h each, 0.99–1.13 s per batch of 32 × 5),
  acceptance 0.05–0.07, autocorrelation times 115–189 steps, so the 2000 steps kept hold about 11
  of the longest and 680–750 independent samples per run: quantiles to a few percent, the tails
  rougher. The two runs give the same posteriors: every median within 0.11 half-widths of the
  other run's, every half-width within 17 % (the largest ratios 0.1 Cl's ETL mobility and 0.6
  Cl's hole mobility, both at their bounds, and 0.6 Cl's electron mobility; in 0.2 Cl within
  0.10 half-widths and 11 %, the cation density at its bound), with 1382, 2626 and 1507
  independent samples between them (`benchmarks/samples/compare_seeds.py`). Median and 16–84 %
  half-width, both runs together:

  | | 0.1 Cl | 0.2 Cl | 0.6 Cl | bounds |
  |---|---|---|---|---|
  | `mu_n` (m²/Vs) | 2.06e-5 ×1.06 | 1.03e-4 ×1.09 | 6.0e-6 ×1.07 | 1e-6 – 1e-3 |
  | `mu_p` | 9.1e-4 ×1.12 (↑ 63 %) | 9.4e-4 ×1.07 ↑ | 1.02e-6 ×1.02 ↓ | 1e-6 – 1e-3 |
  | `N_t` (m⁻³) | 2.8e19 ×2.4 | 5.5e19 ×2.6 | 1.15e19 ×1.16 (↓ 62 %) | 1e19 – 1e23 |
  | `Nti_htl` (m⁻²) | 5.5e12 ×1.14 | 1.0e11 ×1.02 ↓ | 1.06e14 ×1.07 | 1e11 – 1e15 |
  | `Nti_etl` | 1.8e11 ×1.7 | 1.08e14 ×1.02 | 1.04e11 ×1.05 ↓ | 1e11 – 1e15 |
  | `k_direct` (m³/s) | 1.9e-16 ×11 | 5.3e-18 ×3.7 | 1.49e-17 ×1.14 | 1e-18 – 1e-15 |
  | `R_shunt` (Ω m²) | ≳ 15 | 1.42 ×1.22 | 0.43 ×1.4 | 1e-2 – 1e3 |
  | `mu_htl` | 6.8e-8 ×11 | 1.5e-7 ×12 | 6.8e-6 ×1.43 (↑ 21 %) | 1e-8 – 1e-5 |
  | `mu_etl` | 1.04e-8 ×1.05 ↓ | 6.4e-6 ×1.06 | 1.21e-7 ×1.25 | 1e-8 – 1e-5 |
  | `Ev_htl` (eV) | 5.405 ±0.005 (↓ 40 %) | 5.417 ±0.007 | 5.442 ±0.004 | 5.4 – 5.6 |
  | `Ec_etl` (eV) | 4.063 ±0.003 | 3.900 ±0.000 ↓ | 4.015 ±0.005 | 3.9 – 4.3 |
  | `N_cation` (m⁻³) | 1.14e22 ×1.05 | 1.02e20 ×1.02 ↓ | 9.9e22 ×1.01 ↑ | 1e20 – 1e23 |
  | `N_anion` | 7.6e21 ×1.05 | 1.53e21 ×1.19 | 8.2e22 ×1.01 | 1e20 – 1e23 |
  | generation scale | 0.998 | 1.000 | 1.06 | 0.7 – 1.3 |

  ↑ and ↓: at least 79 % of the samples within 2 % of the box's width from the upper or lower
  bound; in brackets the share where it is 20–79 %. One to four of the fourteen parameters are
  held by a bound in each sample, and not the same ones: the hole mobility leans on the upper
  bound in 0.1 and 0.2 Cl and sits at the lower in 0.6 Cl, the cation density at the lower bound in
  0.2 Cl and at the upper in 0.6 Cl, the interface traps at the lower bound on the HTL's side in
  one sample and on the ETL's in another. There the posterior is the prior's statement, not the
  data's; those parameters want the bounds widened or the model changed, not an error bar.
  `k_direct` is settled in 0.2 and 0.6 Cl (below 2.4e-17, and 1.5e-17 within 14 %); in 0.1 Cl its
  16–84 % range covers two of the box's three decades, as the HTL's mobility does. The shunt is
  measured in 0.2 and 0.6 Cl; in 0.1 Cl the curves say it is above about 15 Ω m², and the
  posterior above that is the prior's, log-uniform up to the bound. `N_t` is settled to a factor
  of 2.4–2.6 in 0.1 and 0.2 Cl, at 3–6e19 m⁻³; in 0.6 Cl it leans on the lower bound.

  The linearised `σ²(JᵀJ)⁻¹` at the best minimum is nowhere narrower than the posterior. In 0.6
  Cl it is of the posterior's order for all but one (multiplicative half-widths ×1.04–×3.3
  against ×1.01–×1.25, ±0.024 eV against ±0.005 in the band edges), and wider by four decades
  for the HTL's mobility, whose fit sits at its bound. In 0.1 Cl it is wider by up to seventeen
  decades (the hole mobility, `k_direct`, the HTL's mobility, the fit of each at its bound) or
  unbounded (`N_t`, the ETL's interface traps); in 0.2 Cl wider by four decades for the HTL's
  mobility and unbounded for `k_direct`. Away from the fit the
  misfit grows faster than its curvature there says, and the bounds cut the rest. The
  remaining spread lies along a few strongly correlated directions: in 0.1 Cl the HTL's band
  edge against its interface traps (+0.99) and the two ion densities (+0.89), both against the
  ETL's mobility (−0.81, −0.72); in 0.2 Cl the HTL's band edge against the anion density (−0.95)
  and the hole mobility (−0.93); in 0.6 Cl the two ion densities (+0.95), the HTL's band edge
  against its interface traps (+0.94), the ETL's band edge against its mobility (−0.93). The
  posteriors are narrow, but the parameters differ between the samples in ways that are not a
  trend in the chlorine content (the hole mobility from one bound to the other, the cation
  density at 1e22, 1e20 and 1e23 m⁻³, the shunt from 0.4 to above 15 Ω m²), so what they
  measure is the fit under these assumptions as much as the cells. With σ the rms misfit the widths are a lower bound: the misfit is systematic and smooth
  in the voltage, not independent noise, and where the posterior is close to Gaussian its widths
  scale with σ (`--sigma`).
- **Ions bounded by their sites** (`validation/hard.py 1024 256 N_max`, 2026-11-15, RTX 4070
  Ti SUPER): `hard.py`'s 1024 devices (1e15–4e18 cm⁻³ of ions frozen at one of five
  accumulations), the absorber's two species bounded by the halide sites' 1.21e28 m⁻³, by
  IonMonger's 2e27 and by 1e26 (Driftfusion's in ChargeTransport.jl). Every voltage of every
  device converges, at the Newton steps of Boltzmann's ions (2.81 per voltage needed, 3.95 paid);
  174 curves/s against 227 without sites, the start's two more solves (docs/solver.md). The
  curves' largest difference from Boltzmann's over their scale, median / 99 % / most: 5.8e-7 /
  2.1e-4 / 1.6e-3 at 1.21e28, 3.5e-6 / 1.3e-3 / 9.6e-3 at 2e27, 7.1e-5 / 2.5e-2 / 0.17 at 1e26.
  On 16 384 of these devices (`validation/hard.py 16384 16384 N_max`, `STEPS=steric
  benchmarks/apptainer.sh`, 2026-11-16, A100-SXM4-40GB) the tail is longer: every voltage
  converges at 1.21e28 and 2e27, the differences 5.5e-7 / 3.4e-4 / 1.7e-2 and 3.3e-6 / 2.0e-3 /
  0.10; at 1e26 one device does not converge at three voltages (reported), the others differ by
  6.7e-5 / 3.9e-2 / 1.6, and one pass pays 9.66 Newton steps per voltage against 6.4 without
  sites (60 at most at one voltage). 1230, 1221 and 1191 curves/s in two passes against 1533.
  At the lattice's sites the bound is a correction of 1e-3 at most to these devices; as the
  parameter the codes take it is the model. On the 1 nm stack with 1e23 m⁻³ of each species
  (`tests/data/steady_1e17_thin_light`, Boltzmann's ions up to 3.9e25 m⁻³), free at every
  voltage from −0.5 to 1.4 V, every voltage converges with 1.2e23, 3e23, 8e23, 2e24 and 1e26 m⁻³
  of sites; with 5e23 the three from −0.5 to −0.4 V do not (reported unconverged).

When jaxdd and SimSS disagree, check SimSS first: its `errJ` at that voltage, its log for "No
convergence", and a rerun at tight tolerances. The cases where SimSS was wrong so far are all in
[differences.md](differences.md).

## A table with refined axes

`validation/axes.py` builds a table of a perovskite cell with ions frozen after a pre-bias
(`tests/data/frozen_1e17_x1_thin_light`) over four axes: the absorber's trap density (three
decades), its ion density (1e15–4e18 cm⁻³), the share `x_ion` of the built-in voltage the ions
screen, and its thickness (250–1000 nm). Each point holds `η(V) = (J_dark − J_light)/J_gen` at
29 voltages from −0.2 to 1.2 V. Three ways from five points per axis, their error against the
solver at 4096 random points between the table's:

| `tol` | Way | Points per axis | Points | Estimate | Error, largest | 99th percentile |
|---|---|---|---|---|---|---|
| 5e-2 | refined | 19, 21, 14, 9 | 50 274 | 4.9e-2 | 3.75e-2 | 2.09e-2 |
| | even, each axis halved until it passes | 17, 17, 17, 9 | 44 217 | 6.2e-2 | 5.30e-2 | 2.61e-2 |
| | equal on every axis | 17⁴ | 83 521 | 5.3e-2 | 4.95e-2 | 2.44e-2 |
| 1e-1 | refined | 9, 13, 10, 7 | 8 190 | 9.3e-2 | 8.09e-2 | 5.50e-2 |
| | even, each axis halved until it passes | 9, 17, 9, 9 | 12 393 | 8.6e-2 | 8.41e-2 | 5.16e-2 |
| | equal on every axis | 9⁴ | 6 561 | 1.1e-1 | 9.76e-2 | 6.76e-2 |

At 5e-2 the refined table is within the tolerance with 1.7 times fewer devices than seventeen
points on every axis, which is at it; the even axes do not pass their own test at 44 217 points
and the next halving is beyond 100 000. At 1e-1 nine points on every axis are at the tolerance
on these test points (1.25e-1 on another 4096), and the refined table, within it, is no
smaller. The estimate is the table's error within a third in every row.

Why the gain is small here, and what the error is:

- **The function is smooth.** Along lines of 257 points through the twelve worst test points
  the linear interpolation's error falls by 3 to 4 per halving on every axis: second order, no
  kink or jump. At nine points per axis the transitions are one to three intervals wide.
- **It is a front that moves.** `η(V)` falls from its plateau in one flank, between about
  1.2 V where the ions do not screen and about the pre-bias where they screen fully; the change
  happens between 1e17 and 1e18 cm⁻³ of ions and only for `x_ion > 0`. At nine points per axis,
  at one voltage and on one line of the other axes, 25 % (trap density), 20 % (ion density), 13 % (`x_ion`) and 0.6 %
  (thickness) of the intervals miss by more than an eighth of 5e-2; over all voltages and
  lines every interval of every axis does. A tensor product needs the points wherever the
  front passes.
- **No other voltage axis helps.** The curves of a cell's corners shifted along `V` before
  they are averaged, by the pre-bias, by `V_oc` or by the voltage where `η` is half its
  plateau, or scaled by `V_oc`: 1.1e-1 to 4.1e-1 at nine points per axis where the fixed
  voltages give 7.2e-2 (512 test points). The curve changes its form: its zero near 1.2 V stays
  while the flank moves, `V_oc` (0.85–1.38 V) does not follow the flank (0.03–1.58 V), and the
  flank's place interpolates no better than `η` (up to 0.2 V off).
- **What the refinement saves is the axes' worst places not being the same.** The sum of each
  axis's largest miss is 1.7e-1 at 9⁴ and 9.9e-2 at 17⁴, the largest sum in one cell 1.1e-1
  and 5.3e-2, the error 9.8e-2 and 5.0e-2. A test of each axis against a share of the
  tolerance refines every axis for another's worst place: the thickness from 9 to 17 points
  doubles the table for 5.0e-2 against 4.8e-2. The sum in each cell
  (`refine_axes`' test) leaves it at 9 and puts the points between 2e17 and 2e18 cm⁻³ of ions
  and at the highest trap densities.
- **The largest error needs many test points.** It is in under 1 % of the volume (at 17⁴,
  0.7 % of the points are above 2.5e-2): 512 points give 7.2e-2 at 9⁴ and 3.7e-2 at 17⁴ where
  4096 give 1.0e-1 to 1.3e-1 and 4.8e-2 to 5.0e-2.

## The tests

The tests compare with SimSS's and ZimT's runs on their grid (`grid="simss"`, a wrapper of
`build_device` and `device_parameters` in `tests/test_solver.py`), ZimT's with its sums in
the integral current (`zimt_current=True`, the wrapper `zimt_transient`); the graded grid's
own tests name it. `pytest` runs on whatever JAX has (CPU or GPU), in parallel, one worker per physical core
(pytest-xdist; `-n 0` for one process): on a 16-core desktop about 75 s, and 30 s once JAX's
compilation cache (`.pytest_cache/jax`, `tests/conftest.py`) holds the programs; in one process
it was 400 s. What is left is tracing, which JAX does not cache, and the gradients' finite
differences, about 12 s each for the slowest. The CI (`.github/workflows/ci.yml`) runs them on
the CPU for a push whose commit message says `[test]` (and always for a tag, a pull request
and a run started by hand), twice: with the newest JAX on Python 3.13 and with the oldest that
jaxdd supports (JAX 0.4.31, NumPy 1.26, Python 3.11). The main ones:

| Test | What it checks |
|---|---|
| `test_jv_reproduces_simss` | the J-V curve of `tests/data/thin_l1_light` against SimSS's `JV.dat` |
| `test_jv_with_ions_reproduces_simss` | the same for frozen and free ions |
| `test_batch_matches_the_one_device_solver` | `solve_batch` against `solve_jv` |
| `test_a_dark_scan_with_a_density_far_below_the_floor_converges` | a dark scan with densities far below the convergence floor: every voltage converges, the current monotone and zero at 0 V, SimSS's curve (the extrapolated start, [solver.md](solver.md)) |
| `test_free_ions_keep_their_number` | the ions' constraint holds to round-off |
| `tests/test_steric.py` | ions bounded by their sites: Boltzmann's curve to the bit without sites and to 1e-10 with 1e35 m⁻³; at 10–100× the mean density no density above the sites, one electrochemical potential over the layer, the number kept; the start's normalisation; the blocks against autodiff in saturation; batch against one device, frozen against free at the pre-bias; the gradient in `N_max`; a hard device that cycled from Poisson's start converges from Boltzmann's; refused in time, and below the mean density |
| `test_the_block_sweep_solves_the_block_tridiagonal_system` | the sweep against a dense solve |
| `test_short_circuit_current_never_exceeds_the_generation` | an invariant |
| `test_bernoulli_*` | finite value and derivative at 0, `B(−x) = B(x) + x`, SimSS's series within its own error |
| `test_grid_spacing_*` | SIMsalabim's grid, including its interface quirk |
| `test_grid_split_*` | each layer's share of the points where it is a whole number: exact, and as SimSS's 10-byte reals give it ([differences.md](differences.md) §5) |
| `test_unsupported_input_is_refused_by_name` | `check_supported` |
| `test_the_differential_current_reproduces_simss` | `currDiffInt = 1` (`tests/data/diff_iface_light`) to 1e-9 of SimSS's scale, single and in a batch with the integral form |
| `test_ions_held_from_the_start_reproduce_simss` | `fixIons` without `preCond` against the patched SimSS, to 1e-9 of its scale |
| `test_ions_in_a_region_across_layers_reproduce_simss` | an ionic region of two layers, free and frozen, against SimSS |
| `test_two_mobile_regions_of_a_species_reproduce_simss` | two regions per species (four normalisations) against SimSS, single and in a padded batch |
| `test_ionic_regions_as_simss_finds_them` | `ion_regions` as SimSS's `Init_Ionic_Region`, and what is refused (differences.md §10) |
| `test_voltages_are_simss_own` | `simss_voltages` against the voltages of every SimSS run in `tests/data` |
| `test_logarithmic_downward_scan_reproduces_simss` | `Vdist = 2`, `Vscan = -1` (`tests/data/logv_light`): SimSS's voltages and curve, to 2e-6 of its scale |
| `test_voltages_from_experimental_data_and_simss_checks` | `useExpData` (`expJV`'s voltages, in the file's order) and SimSS's checks on the voltage parameters |
| `test_transients_of_electrons_and_holes_reproduce_zimt` | the four ZimT cases without traps, to 1e-9 of ZimT's scale |
| `test_transients_with_traps_reproduce_zimt` | the four ZimT cases with traps (neutral, charged, a trap file, interface), to 1e-9 |
| `test_transients_in_a_batch_reproduce_zimt` | all eight as one batch, each device with its own steps; every `tJFile` column, the QFLS to 1e-9 V under light and 4e-6 V in the dark after it, but in ZimT's two unconverged steady states |
| `test_transients_with_mobile_ions_reproduce_zimt` | the four ZimT cases with mobile ions, to 1e-7 of ZimT's scale |
| `test_transients_with_and_without_ions_in_a_batch` | devices with mobile ions beside devices with traps and with neither, in one batch of 5×5 blocks; every `tJFile` column, the QFLS to 4e-6 V |
| `test_ions_in_two_regions_per_species_against_zimt` | two ion regions per species: ZimT's currents with `zimt_ion_currents`, every region's without |
| `test_the_integral_current_in_time_is_the_one_the_equations_balance` | the integral form in time on the graded grid: the differential form's current to 1e-9 of the scale; ZimT's sums (`zimt_current`) more than 1e-2 off (differences.md §13) |
| `test_layer_voltages_as_zimt_writes_them` | `specialOutput = spv`: each layer's voltage drop, with every other column |
| `test_automatic_stop_reproduces_zimt` | `Track` 2 and 3 on the photovoltage after a pulse: ZimT's stop within one step |
| `test_states_compare_as_zimt_compares_them` | `States_Are_Equal` under each `convVar` |
| `test_tangents_through_time_are_finite_differences` | a transient's tangents (traps, the open circuit, mobile ions) against central differences |
| `test_a_fit_through_time_recovers_its_parameters` | two parameters recovered from a transient by its gradients |
| `test_failure_modes_as_zimt_treats_them` | `failureMode` 0, 1, 2 on steps made to fail, `Δt` from the last step kept |
| `test_the_ions_blocks_in_time_are_autodiff` | `_assemble_ions`' 5×5 blocks and slot columns against `jax.jacfwd`, over a step and in the steady state |
| `test_photovoltage_at_open_circuit_reproduces_zimt` | the photovoltage at the open circuit through a light pulse, `Vint` to 1e-10 V of ZimT's |
| `test_open_circuit_in_the_steady_state_is_simss_voc` | `oc` at `t = 0` against SimSS's `Voc` (`tests/data/thin_l1_voc_light`), to 1e-10 V |
| `test_series_resistance_in_time_reproduces_zimt` | a sweep with series resistance and shunt: `Jext` to 1e-9 of ZimT's scale, `Vint` to 1e-9 V |
| `test_series_resistance_beside_given_voltages_and_ions_in_a_batch` | the internal voltage's row in a batch with a device whose voltage is given and one with mobile ions |
| `test_zimt_steady_states_reproduce_zimt` | a ZimT input read and built, its steady states (`tests/data/zimt_steady_light`) to 1e-9 of ZimT's scale |
| `test_tvg_as_zimt_reads_it` | `read_tvg` as ZimT reads its `tVGFile`, and ZimT's checks on it |
| `test_benchmark.py` | `benchmarks/run.py` hands SimSS exactly the devices jaxdd solves |
| `test_free_surface.py` | free ends, where SimSS is no reference: a uniform film against its algebraic balance (1e-10), ambipolar diffusion to a recombining surface against the cosh solution (1e-5), one free end independent of the voltage and the contact's role, a charged HTL/absorber/ETL neutral with no field at its ends and its implied V_oc converged in the grid, a blocking contact (SimSS's nearest device) 0.1 V off and moving with its work function and voltage, in time a uniform film's decay against implicit Euler in closed form (1e-9), a stack with mobile ions and charged traps through a decay converging at first order in the step, one free end not feeling its contact's voltage, Braun's generation and Langevin's rate in a field-dependent mobility in a film without field against their zero-field balance (1e-8), Braun's in a charged stack converged in the grid, mobile ions conserved in their layer with the film neutral and the same frozen after a pre-bias, a batch per device equal to `solve_state`, the implied V_oc's derivative in a surface's velocity against central differences ([model.md](model.md#free-surfaces-jaxdds-w_l-or-w_r--free)) |
| `test_axes.py` | `jaxdd.axes` on functions whose answer is known: the interpolation exact on multilinear functions; refined axes within the tolerance with fewer points than even ones on a peak; an axis refined for its part of a cell's error, not for a share; a feature the start misses is not found; the limits and NaN |

`tests/data/` holds SimSS 5.36 runs, inputs and `JV.dat` unchanged, each described in
[its README](../tests/data/README.md); the first four are a perovskite cell without ions, the
same with 1e17 cm⁻³ ions frozen after a pre-bias (SimSS with
[`simss/fixed_ions.patch`](../simss/README.md)), the same with the ions free, and a thick cell
in the dark with a thick doped transport layer and ions frozen near flat band.

## The validation scripts

Run from the repository root with `.venv/bin/python`. Several read reference data kept outside
the repository (next to it in `../jaxdd-ref/`: SIMsalabim's source, SimSS case directories, a
1000-device sample of a perovskite parameter table with SimSS's curves); they take the paths
as arguments. SimSS itself is built by [`simss/`](../simss/README.md).

| Script | What |
|---|---|
| `validation/compare_simss.py CASE_DIR …` | one device against SimSS: inputs and `JV.dat` in each directory |
| `validation/table_gate.py SAMPLE [simss\|exact]` | the 1000-device check, in `η`; writes `validation/table_gate_<bernoulli>.json` |
| `validation/batch_vs_single.py CASE_DIR …` | the batched solver against the one-device solver and SimSS |
| `validation/shortcuts.py CASE_DIR …` | the batched solver with its shortcuts against `strict=True` |
| `validation/bench.py [N] [batch]` | curves per second ([hardware.md](hardware.md)) |
| `validation/breakdown.py [batch]` | one Newton step's time, part by part |
| `validation/hard.py [N] [batch] [N_max]` | a hard workload, frozen ions over a wide range ([batching.md](batching.md)); with `N_max` the ions bounded by their sites, against Boltzmann's |
| `validation/fit_measured.py [SAMPLE …] [--start cold\|paper] [--device mirrored\|anode] [--nk FILE] [--back FILE] [--starts N]` | a fit to the measured curves of Heester et al. (2025), five intensities at once; the absorber's and back electrode's n,k data from files outside the repository; from N starts at once |
| `validation/sample_measured.py [SAMPLE] [--move de\|stretch] [--walkers W] [--steps S] [--sigma F] [--out FILE] [--from-multistart]` | the fit's posterior by an ensemble sampler over the batch; the chain to an `.npz`; around the best of `fit_measured.py --starts N` |
| `validation/axes.py [tol] [chunk] [test_points] [max_size]` | a table of a perovskite cell over four axes: refined axes against even ones, the error at random points |
| `validation/fit.py CASE_DIR [--start S] [--fix NAME …] [--noise σ]` | a fit by gradients to the case's SimSS curve, from a perturbed start; what the curve determines (singular values) |

## Running SimSS for a comparison

Build SimSS from [`simss/`](../simss/README.md) (`docker build -t jaxdd/simss:5.36 simss`): the
release's `simss`, and `simss_fixed_ions` for devices with ions frozen after a pre-bias. On one
CPU core a J-V curve of 61 voltages takes SimSS about 2 s. `benchmarks/run.py` runs SimSS and
jaxdd on the same devices on the machine at hand and compares their speed and their curves
([hardware.md](hardware.md)).

Set `outputRatio = 0` in SimSS's setup: SimSS 5.36 otherwise writes its internal variables to a
file literally named `none` (14.7 MB per run) although `varFile = none`. The curves are
bit-identical either way; the disk traffic is not.
