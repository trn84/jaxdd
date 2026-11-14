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
- **Series and shunt resistance** (2026-11-13): 2e-4 and 0.3 Ω m² on the 1 nm stack, the
  series resistance moving the voltage by 1.8 V at the top of the scan: `J_ext` within 2.2e-6 of
  SimSS's scale, `V_ext` within 4 µV.
- **The transfer-matrix profile** (2026-11-13): on the 1 nm stack between glass, 110 nm ITO
  and gold, and on a 20/500/10 nm stack between glass, 150 nm ITO and silver, the profile on the
  grid is SimSS's within 4e-12 and the curves within 2e-7 and 6e-7 of SimSS's scale; a uniform
  profile (`genProfile = none`) within 2.4e-6.
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
  Bayesian optimisation ran about 25 000 simulations per sample). What the curves determine: the
  Jacobian's singular values spread over seven to nine decades, three to seven of the fourteen
  parameters end at a bound, and `k_direct` ends within a factor of three of its lower bound in
  all six fits (two starts per sample), so
  that with these assumptions the curves settle about three to five combinations of the
  parameters, not the parameters one by one — the paper's own point. The 0.6 Cl sample, with the
  largest band step to the ETL in the paper's fit, is the one the assumptions do not reach.
  With the cell in its own orientation and the transfer-matrix profile instead
  (`--device anode`, a stand-in perovskite's n,k data), the misfits are 0.86, 0.75 and 2.4 %:
  no better, and the fitted parameters move by up to a factor of 20 between the two assumptions
  about the generation (0.2 Cl: the ETL interface's traps 1.1e14 against 2.4e13 m⁻²). What these
  curves say about a single parameter depends on the unstated assumptions as much as on the data.

- **The fit's uncertainty, by sampling** (`validation/sample_measured.py`, 2026-11-14): a posterior
  of the same fourteen parameters, by an ensemble sampler whose walkers' five intensities are
  solved as one batch (differential evolution, the walkers started around the least-squares fit;
  the prior uniform in the fitted coordinates within the bounds, the likelihood Gaussian with σ the
  fit's rms misfit). All three samples, 128 walkers and 3000 steps each, the first 1000
  discarded: 4.2 h on an A100-SXM4-40GB (0.75–0.96 s per batch of 64 × 5 curves). Acceptance
  0.05–0.07, autocorrelation times 103–246 steps, so the 2000 steps kept hold 8–10 of the longest
  and 1000–1300 independent samples: quantiles to a few percent, the tails rougher. The 0.2 Cl
  sample run on the RTX 4070 Ti SUPER with 64 walkers (1.5 h, 0.92 s per batch of 32 × 5)
  gives the same medians and widths to within that noise. Median and 16–84 % half-width:

  | | 0.1 Cl | 0.2 Cl | 0.6 Cl | bounds |
  |---|---|---|---|---|
  | `mu_n` (m²/Vs) | 9.7e-4 ×1.04 ↑ | 1.03e-4 ×1.09 | 9.9e-4 ×1.01 ↑ | 1e-3 above |
  | `mu_p` | 1.04e-4 ×1.09 | 9.3e-4 ×1.07 ↑ | 4.4e-4 ×1.02 | 1e-3 above |
  | `N_t` (m⁻³) | 3.1e19 ×2.5 | 5.6e19 ×2.6 | 3.1e19 ×2.5 | |
  | `Nti_htl` (m⁻²) | 2.4e11 ×2.0 | 1.0e11 ×1.02 ↓ | 2.4e13 ×1.07 | 1e11 below |
  | `Nti_etl` | 7.7e13 ×1.10 | 1.08e14 ×1.02 | 9.9e14 ×1.01 ↑ | 1e15 above |
  | `k_direct` (m³/s) | 2.5e-17 ×10 | 5.4e-18 ×3.7 | 2.9e-17 ×10 | 1e-18 – 1e-15 |
  | `R_shunt` (Ω m²) | ≳ 15 | 1.42 ×1.22 | ≳ 15 | up to 1e3 |
  | `mu_htl` | 1.0e-6 ×5.7 | 1.6e-7 ×12 | 1.5e-8 ×2.0 (↓ 23 %) | 1e-8 below |
  | `mu_etl` | 1.01e-8 ×1.02 ↓ | 6.4e-6 ×1.06 | 2.9e-6 ×2.0 | 1e-8 below |
  | `Ev_htl` (eV) | 5.599 ±0.001 ↑ | 5.417 ±0.007 | 5.401 ±0.001 ↓ | 5.4 – 5.6 |
  | `Ec_etl` (eV) | 4.071 ±0.003 | 3.900 ±0.000 ↓ | 4.139 ±0.003 | 3.9 – 4.3 |
  | `N_cation` (m⁻³) | 1.11e21 ×1.06 | 1.02e20 ×1.02 ↓ | 4.35e21 ×1.04 | 1e20 below |
  | `N_anion` | 5.9e20 ×1.18 | 1.53e21 ×1.19 | 1.22e22 ×1.01 | |
  | generation scale | 0.998 | 1.000 | 1.02 | |

  ↑ and ↓: at least 79 % of the samples within 2 % of the box's width from the upper or lower
  bound. Three or four of the fourteen parameters are held by a bound in every sample, and
  not the same ones: the electron and the hole mobility take turns at the upper bound, the
  HTL's band edge sits at the upper bound in one sample and at the lower in another. There the
  posterior is the prior's statement, not the data's; those parameters want the bounds widened or
  the model changed, not an error bar. `k_direct` is settled only in the 0.2 Cl sample (below
  2.4e-17); in the other two its 16–84 % range covers two of the box's three decades. The shunt
  is measured only in the 0.2 Cl sample; in the other two the curves say it is above about
  15 Ω m², and the posterior above that is the prior's, log-uniform up to the bound. `N_t` is
  settled to a factor of 2.5 in all three, at 3–6e19 m⁻³.

  The linearised `σ²(JᵀJ)⁻¹` at the fit is nowhere narrower than the posterior and is wider by
  up to seven decades (0.1 Cl's `mu_n`, whose fit sits at its bound), or unbounded (`k_direct`
  in all three; `N_t` and the shunt in 0.1 and 0.6 Cl; in 0.1 Cl also the HTL's mobility and
  interface traps): away from the fit the misfit grows
  faster than its curvature there says, and the bounds cut the rest. The remaining spread lies
  along a few strongly correlated directions, different in each sample: 0.1 Cl the ETL's band
  edge against its interface traps (−0.96); 0.2 Cl the HTL's band edge against the anion density
  (−0.95) and the hole mobility (−0.93); 0.6 Cl the cation density against the ETL's band edge
  (+0.96). The posteriors are narrow, but the parameters differ between the samples in ways
  that are not a trend in the chlorine content (the mobilities swapping, the cation density at
  1e20, 1e21, 4e21), so what they measure is the fit under these assumptions as much as the
  cells. With σ the rms misfit the widths are a lower bound: the misfit is systematic and smooth
  in the voltage, not independent noise, and where the posterior is close to Gaussian its widths
  scale with σ (`--sigma`).

When jaxdd and SimSS disagree, check SimSS first: its `errJ` at that voltage, its log for "No
convergence", and a rerun at tight tolerances. The cases where SimSS was wrong so far are all in
[differences.md](differences.md).

## The tests

`pytest` runs on whatever JAX has (CPU or GPU) in under a minute:

| Test | What it checks |
|---|---|
| `test_jv_reproduces_simss` | the J-V curve of `tests/data/thin_l1_light` against SimSS's `JV.dat` |
| `test_jv_with_ions_reproduces_simss` | the same for frozen and free ions |
| `test_batch_matches_the_one_device_solver` | `solve_batch` against `solve_jv` |
| `test_a_dark_scan_with_a_density_far_below_the_floor_converges` | a dark scan with densities far below the convergence floor: every voltage converges, the current monotone and zero at 0 V, SimSS's curve (the extrapolated start, [solver.md](solver.md)) |
| `test_free_ions_keep_their_number` | the ions' constraint holds to round-off |
| `test_the_block_sweep_solves_the_block_tridiagonal_system` | the sweep against a dense solve |
| `test_short_circuit_current_never_exceeds_the_generation` | an invariant |
| `test_bernoulli_*` | finite value and derivative at 0, `B(−x) = B(x) + x`, SimSS's series within its own error |
| `test_grid_spacing_*` | SIMsalabim's grid, including its interface quirk |
| `test_grid_split_*` | each layer's share of the points where it is a whole number: exact, and as SimSS's 10-byte reals give it ([differences.md](differences.md) §5) |
| `test_unsupported_input_is_refused_by_name` | `check_supported` |
| `test_benchmark.py` | `benchmarks/run.py` hands SimSS exactly the devices jaxdd solves |

`tests/data/` holds four SimSS 5.36 runs, inputs and `JV.dat` unchanged: a perovskite cell
without ions, the same with 1e17 cm⁻³ ions frozen after a pre-bias (SimSS with
[`simss/fixed_ions.patch`](../simss/README.md)),
the same with the ions free, and a thick cell in the dark with a thick doped transport layer
and ions frozen near flat band.

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
| `validation/fit_measured.py [SAMPLE …] [--start cold\|paper] [--device mirrored\|anode]` | a fit to the measured curves of Heester et al. (2025), five intensities at once |
| `validation/sample_measured.py [SAMPLE] [--move de\|stretch] [--walkers W] [--steps S] [--sigma F] [--out FILE]` | the fit's posterior by an ensemble sampler over the batch; the chain to an `.npz` |
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
