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
  flagged; where both runs converged the curves agree to 4e-5 in `η`. The rest are all in the dark, from 0.775 V up, with ions frozen at full
  accumulation and at least 5e17 cm⁻³ on the device: single voltages scattered through the
  scan, the strong frozen charge of [tutorial.md](tutorial.md), "Known limits".
- **The batched solver** matches the one-device solver to 1e-10, and its shortcuts change `J` by
  at most 1.8e-6 of the generation current ([solver.md](solver.md)).

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

## Running SimSS for a comparison

Build SimSS from [`simss/`](../simss/README.md) (`docker build -t jaxdd/simss:5.36 simss`): the
release's `simss`, and `simss_fixed_ions` for devices with ions frozen after a pre-bias. On one
CPU core a J-V curve of 61 voltages takes SimSS about 2 s. `benchmarks/run.py` runs SimSS and
jaxdd on the same devices on the machine at hand and compares their speed and their curves
([hardware.md](hardware.md)).

Set `outputRatio = 0` in SimSS's setup: SimSS 5.36 otherwise writes its internal variables to a
file literally named `none` (14.7 MB per run) although `varFile = none`. The curves are
bit-identical either way; the disk traffic is not.
