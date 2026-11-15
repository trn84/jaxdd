"""jaxdd against SimSS, and the properties of its pieces."""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.device import build_device as _build_device
from jaxdd.device import device_parameters as _device_parameters
from jaxdd.device import make_grid
from jaxdd.simss_input import (
    Setup,
    check_supported,
    read_setup,
    read_tvg,
    setup_from_texts,
    simss_voltages,
)
from jaxdd.solver import bernoulli, bernoulli_simss, solve_jv

DATA = Path(__file__).parent / "data"


def build_device(setup, *args, grid="simss", **kwargs):
    """`device.build_device` on SimSS's grid unless told otherwise: the tests compare with SimSS's
    and ZimT's runs point by point (jaxdd's default is its graded grid)."""
    return _build_device(setup, *args, grid=grid, **kwargs)


def device_parameters(setup, *args, grid="simss", **kwargs):
    """`device.device_parameters` on SimSS's grid unless told otherwise, as `build_device`."""
    return _device_parameters(setup, *args, grid=grid, **kwargs)


def test_bernoulli_is_finite_with_a_finite_derivative_at_zero() -> None:
    """The double where: neither B nor dB/dx sees 0/0."""
    assert float(bernoulli(0.0)) == 1.0
    assert float(jax.grad(bernoulli)(0.0)) == pytest.approx(-0.5)


@pytest.mark.parametrize("x", [-30.0, -1.0, -1e-3, 1e-3, 0.3, 2.0, 30.0])
def test_bernoulli_identity(x: float) -> None:
    """Invariant: B(−x) = B(x) + x, the identity behind Scharfetter–Gummel's symmetry."""
    assert float(bernoulli(-x)) == pytest.approx(float(bernoulli(x)) + x, rel=1e-12, abs=1e-14)


def test_simss_bernoulli_matches_the_exact_one_to_its_own_error() -> None:
    x = jnp.linspace(-5.0, 5.0, 2001)
    assert float(jnp.max(jnp.abs(bernoulli_simss(x) / bernoulli(x) - 1))) < 1e-3


def test_grid_spacing_sums_to_the_device_but_for_the_interfaces() -> None:
    """Invariant: one point more than NP, monotone, and Σh within 1e-3 of 1 (SIMsalabim
    equalises the spacing across each interface, which moves the sum slightly)."""
    x, h, i0, i1, _ = make_grid([1e-9, 500e-9, 1e-9], 400, 4.0)
    assert len(x) == 402 and np.all(np.diff(x) > 0)
    assert abs(h[:-1].sum() - 1.0) < 1e-3
    assert i0 == [0, 6, 397] and i1 == [5, 396, 401]


#: Stacks (m) whose share of the grid is a whole number in exact arithmetic, with the absorber's
#: points as SimSS 5.36 gave them (its `Var.dat`, column `lid`; 2026-11-11), and exactly.
SPLIT_EDGES = [
    (("7e-10", "3.5e-07", "7e-10"), 389, 390),
    (("1.4e-09", "7e-07", "1.4e-09"), 389, 390),
    (("1e-07", "5e-07", "2e-07"), 250, 250),
    (("1e-06", "7e-07", "5e-07"), 127, 127),
]


@pytest.mark.parametrize(("stack", "simss", "exact"), SPLIT_EDGES)
def test_grid_split_on_a_whole_share(stack: tuple[str, ...], simss: int, exact: int) -> None:
    """Where FLOOR meets a whole number, "simss" lands where SimSS's 10-byte reals did and
    "exact" on the whole number; both give every point."""
    for split, absorber in (("simss", simss), ("exact", exact)):
        _, _, i0, i1, lid = make_grid(stack, 400, 4.0, split)
        assert i1[1] - i0[1] == absorber
        assert i1[-1] == 401 and np.all(np.diff(lid) >= 0)


def test_the_graded_grid_gives_every_layer_its_ends() -> None:
    """`grid="graded"` (`device._graded_layout`): `NP + 2` points, increasing, the layers in order
    and each spanning its thickness; spacings growing from the smallest at each layer's end
    towards its middle; a 1 nm layer beside 500 nm and 30 nm about the same points at 400 as at
    1600 (its ends' e-folds), where SIMsalabim's grid gives it 5 at any `NP`."""
    from jaxdd.device import GRADED_SMALLEST, _graded_layout, _spacing

    stack = ["1e-09", "5e-07", "3e-08"]
    L = np.array([float(x) for x in stack])
    for NP in (400, 1600):
        i0, i1, lid, raw, norm = _graded_layout(stack, NP)
        x, h = _spacing(L, L.sum(), i0, lid, raw, norm)
        assert len(x) == NP + 2 and np.all(np.diff(x) > 0)
        assert i0[0] == 0 and i1[-1] == NP + 1
        assert all(b + 1 == a for a, b in zip(i0[1:], i1[:-1], strict=True))
        for k in range(3):  # each layer's points span it (its first to the next layer's first)
            end = x[i0[k + 1]] if k < 2 else L.sum()
            assert abs((end - x[i0[k]]) - L[k]) < 1e-3 * L[k] + 2 * h[i1[k]] * L.sum()
        absorber = raw[i0[1] : i1[1] + 1]
        assert absorber[1] < absorber[len(absorber) // 2] / 100  # graded towards its ends
        assert raw[i0[1]] < 10 * GRADED_SMALLEST
    simss = make_grid(stack, 1600, 4.0)
    assert simss[3][0] - simss[2][0] == 5  # SIMsalabim's thin layer: 5 points beyond its first
    assert i1[0] - i0[0] > 30  # the graded one's at 1600


def test_the_graded_grid_converges_where_simss_does_not() -> None:
    """A hard device with a 1 nm ETL (`validation/hard.py`, the one SIMsalabim's grid is worst
    on): η on the graded grid at 400 points within 1e-3 of the graded grid at 1600, where
    SIMsalabim's grid at 1600 is not (its 1 nm layer keeps 5 points)."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
    from hard import V, hard_setups

    from jaxdd.batch import prepare, solve_batch

    s = hard_setups(91)[90]
    light, dark = (Setup(raw=dict(s.raw, G_frac=g), layers=s.layers, directory=s.directory)
                   for g in ("1", "0"))  # fmt: skip

    def solve(NP, grid):  # J_dark − J_light, and the generation current on that grid
        setups = [Setup(raw=dict(x.raw, NP=str(NP)), layers=x.layers, directory=x.directory)
                  for x in (light, dark)]  # fmt: skip
        batch = prepare([build_device(x, grid=grid) for x in setups])
        J, ok, _ = solve_batch(batch, V)
        assert np.asarray(ok).all()
        J = np.asarray(J)
        return (J[1] - J[0])[V <= 1.0 + 1e-9], float(batch["J_scale"][0])

    ref, scale = solve(1600, "graded")  # η over the reference's generation current, as grid.py
    assert np.max(np.abs(solve(400, "graded")[0] - ref)) < 1e-3 * scale
    assert np.max(np.abs(solve(1600, "simss")[0] - ref)) > 2e-3 * scale


def test_the_graded_grid_is_differentiable() -> None:
    """The graded grid's points are the input's structure, its spacing scaled by the thicknesses
    as SIMsalabim's is: `device_parameters(grid="graded")` builds `batch.device_tree` in JAX, the
    same device as `build_device`, and its derivative in a thickness is finite."""
    from jaxdd.batch import device_tree

    setup = read_setup(DATA / "thin_l1_light" / "simulation_setup.txt")
    structure, p = device_parameters(setup, grid="graded")
    tree = device_tree(structure, {k: jnp.asarray(v) for k, v in p.items()})
    dev = build_device(setup, grid="graded")
    assert np.allclose(np.asarray(tree["x"]), dev.x, rtol=1e-12, atol=0)

    def last(Lk):
        q = {k: jnp.asarray(v) for k, v in p.items()}
        q["L"] = q["L"].at[1].set(Lk)
        return device_tree(structure, q)["x"][-1]

    assert np.isfinite(float(jax.grad(last)(jnp.asarray(p["L"][1]))))


def test_grid_split_first_layer_on_a_whole_share() -> None:
    """SimSS gave the 100 nm layer of 100/500/200 nm 49 points beyond its first, not 50."""
    assert make_grid(("1e-07", "5e-07", "2e-07"), 400, 4.0, "simss")[3][0] == 49
    assert make_grid(("1e-07", "5e-07", "2e-07"), 400, 4.0, "exact")[3][0] == 50


def test_a_setup_from_texts_builds_the_same_device() -> None:
    """`setup_from_texts` (file names to texts, no files) gives the device `read_setup` gives;
    a file the setup names but the mapping lacks is refused by name."""
    import dataclasses

    from jaxdd.simss_input import setup_from_texts

    case = DATA / "frozen_1e17_x1_thin_light"
    files = {f.name: f.read_text() for f in case.iterdir() if f.suffix == ".txt"}
    a = build_device(read_setup(case / "simulation_setup.txt"))
    b = build_device(setup_from_texts(files))
    for field in dataclasses.fields(a):
        assert np.array_equal(getattr(a, field.name), getattr(b, field.name)), field.name
    profile = setup_from_texts(files).raw["genProfile"]
    without = {k: v for k, v in files.items() if k != profile}
    with pytest.raises(FileNotFoundError, match="genProfile"):
        build_device(setup_from_texts(without))


def test_jv_reproduces_simss() -> None:
    """The discrete equations are SimSS's: with its Bernoulli approximation, its curve to its
    own convergence tolerance at every voltage."""
    case = DATA / "thin_l1_light"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-5 * scale


def test_short_circuit_current_never_exceeds_the_generation() -> None:
    """Invariant: no device collects more than it generates (the generation's current)."""
    case = DATA / "thin_l1_light"
    dev = build_device(read_setup(case / "simulation_setup.txt"))
    res = solve_jv(dev, np.array([0.0]))
    from jaxdd.constants import Q

    generated = Q * np.sum(dev.h[:-1] * dev.G[:-1]) / np.sum(dev.h[:-1]) * dev.L
    assert -res.J[0] <= generated * (1 + 1e-9)


def test_unsupported_input_is_refused_by_name(tmp_path) -> None:
    case = DATA / "thin_l1_light"
    for f in case.iterdir():
        (tmp_path / f.name).write_text(f.read_text())
    setup_file = tmp_path / "simulation_setup.txt"
    text = setup_file.read_text()
    assert "currDiffInt = 2 " in text
    setup_file.write_text(text.replace("currDiffInt = 2 ", "currDiffInt = 3 ", 1))
    with pytest.raises(NotImplementedError, match="currDiffInt"):
        check_supported(read_setup(setup_file))


def test_the_differential_current_reproduces_simss() -> None:
    """`currDiffInt = 1` (`Calc_Curr_Diff`: each interval's Scharfetter–Gummel flux, across an
    interface the current before it plus the current through its traps), on `iface_charged_light`
    with SimSS at tolerances of 1e-9: its curve to 1e-9 of its scale, by both solvers, in a batch
    with the same device in the integral form (which keeps its own curve)."""
    from jaxdd.batch import prepare, solve_batch

    devices, refs = [], []
    for case in ("diff_iface_light", "iface_charged_light"):
        setup = read_setup(DATA / case / "simulation_setup.txt")
        check_supported(setup)
        devices.append(build_device(setup))
        refs.append(np.genfromtxt(DATA / case / "JV.dat", names=True))
    assert devices[0].curr_diff and not devices[1].curr_diff
    V = refs[0]["Vint"]
    assert np.array_equal(V, refs[1]["Vint"])
    single = solve_jv(devices[0], V, bernoulli_name="simss")
    J, ok, _ = solve_batch(prepare(devices), V, bernoulli_name="simss")
    assert single.converged.all() and np.asarray(ok).all()
    for got, ref, tol in ((single.J, refs[0], 1e-9), (np.asarray(J)[0], refs[0], 1e-8),
                          (np.asarray(J)[1], refs[1], 1e-5)):  # fmt: skip
        assert np.max(np.abs(got - ref["Jext"])) < tol * np.max(np.abs(ref["Jext"]))


def test_ions_held_from_the_start_reproduce_simss() -> None:
    """`fixIons` without `preCond`: the ions held where SIMsalabim starts them, each layer's own
    density (8e22 anions and 1e23 cations per m³ in the absorber, a net charge), as fixed
    ions; both solvers against SimSS with `simss/fixed_ions.patch` at tolerances of 1e-9."""
    from jaxdd.batch import prepare, solve_batch

    setup = read_setup(DATA / "held_thin_light" / "simulation_setup.txt")
    check_supported(setup)
    assert setup.i("fixIons") and not setup.i("preCond")
    dev = build_device(setup)
    assert not dev.anion_mask.any() and not dev.cation_mask.any()
    assert dev.anion_fixed.max() == 8e22 and dev.cation_fixed.max() == 1e23
    ref = np.genfromtxt(DATA / "held_thin_light" / "JV.dat", names=True)
    scale = np.max(np.abs(ref["Jext"]))
    single = solve_jv(dev, ref["Vint"], bernoulli_name="simss")
    J, ok, _ = solve_batch(prepare([dev]), ref["Vint"], bernoulli_name="simss")
    assert single.converged.all() and np.asarray(ok).all()
    assert np.max(np.abs(single.J - ref["Jext"])) < 1e-9 * scale
    assert np.max(np.abs(np.asarray(J)[0] - ref["Jext"])) < 1e-9 * scale


@pytest.mark.parametrize(
    ("case", "layers", "tol"),
    [("region_etl_light", (0, 1), 1e-6), ("region_htl_frozen_light", (1, 2), 2e-5)],
)
def test_ions_in_a_region_across_layers_reproduce_simss(case, layers, tol) -> None:
    """Mobile ions in a region of two layers (`ionsMayEnter` in both, SIMsalabim's
    `Init_Ionic_Region`), one Boltzmann profile across the interface normalised to both layers'
    ions: free ions entering the 20 nm ETL from the absorber, and ions entering the 10 nm HTL,
    settled at a pre-bias and frozen (SimSS with `simss/fixed_ions.patch`); both solvers
    against SimSS at tolerances of 1e-9, where it converged."""
    from jaxdd.batch import prepare, solve_batch
    from jaxdd.simss_input import ion_regions

    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    assert ion_regions(setup, "anion") == [(*layers, True)]
    dev = build_device(setup)
    assert dev.anion_total == pytest.approx(setup.layers[1].f("L") * setup.layers[1].f("N_anion"))
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    scale = np.max(np.abs(ref["Jext"]))
    single = solve_jv(dev, ref["Vint"], bernoulli_name="simss")
    J, ok, _ = solve_batch(prepare([dev]), ref["Vint"], bernoulli_name="simss")
    assert single.converged.all() and np.asarray(ok).all()
    assert np.max(np.abs(single.J - ref["Jext"])) < tol * scale
    assert np.max(np.abs(np.asarray(J)[0] - ref["Jext"])) < tol * scale


def test_two_mobile_regions_of_a_species_reproduce_simss() -> None:
    """Anions and cations each mobile in the ETL and in the HTL, the absorber closed to them:
    two regions per species, each its own normalisation (`solver._slots`, four in all); both
    solvers against SimSS at tolerances of 1e-9, where it converged, the batch beside a device
    of one region per species (padded with empty slots), which keeps its own curve."""
    from jaxdd.batch import prepare, solve_batch

    setup = read_setup(DATA / "two_regions_light" / "simulation_setup.txt")
    check_supported(setup)
    dev = build_device(setup)
    assert dev.anion_slots.shape[0] == 2 and dev.cation_slots.shape[0] == 2
    ref = np.genfromtxt(DATA / "two_regions_light" / "JV.dat", names=True)
    other = build_device(read_setup(DATA / "pin_iface_light" / "simulation_setup.txt"))
    oref = np.genfromtxt(DATA / "pin_iface_light" / "JV.dat", names=True)[: ref.size]
    assert np.array_equal(oref["Vint"], ref["Vint"])
    single = solve_jv(dev, ref["Vint"], bernoulli_name="simss")
    J, ok, _ = solve_batch(prepare([dev, other]), ref["Vint"], bernoulli_name="simss")
    assert single.converged.all() and np.asarray(ok).all()
    for got, r in ((single.J, ref), (np.asarray(J)[0], ref), (np.asarray(J)[1], oref)):
        assert np.max(np.abs(got - r["Jext"])) < 2e-6 * np.max(np.abs(r["Jext"]))


def test_the_gradient_in_one_regions_ions_is_finite_differences() -> None:
    """With two mobile regions per species, the curve's derivative in the ETL's anion and the
    HTL's cation density, through `device_tree` and the converged solution, against central
    finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "two_regions_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 0.9 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        p = dict(params)
        p["N_anion"] = jnp.asarray(params["N_anion"]).at[0].multiply(jnp.exp(a))
        p["N_cation"] = jnp.asarray(params["N_cation"]).at[2].multiply(jnp.exp(-0.7 * a))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-5
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_ionic_regions_as_simss_finds_them(tmp_path) -> None:
    """`ion_regions` as SIMsalabim's `Init_Ionic_Region`, and what is refused: SimSS 5.36's two
    quirks with regions of several layers (docs/differences.md §10); mobile ions in a layer
    they may not enter are SimSS's input error."""
    from jaxdd.simss_input import ion_region_problems, ion_regions

    case = DATA / "pin_iface_light"
    files = {f.name: f.read_text() for f in case.iterdir() if f.suffix == ".txt"}

    def setup(**layers):
        new = dict(files)
        for name, values in layers.items():
            lines = []
            for line in files[name].splitlines():
                key = line.split("=", 1)[0].strip()
                lines.append(f"{key} = {values[key]} * edited" if key in values else line)
            new[name] = "\n".join(lines)
        return setup_from_texts(new)

    l1, l3 = "L1_parameters.txt", "L3_parameters.txt"
    enter = {"ionsMayEnter": 1, "mu_anion": 1e-12, "mu_cation": 1e-12}
    assert ion_regions(setup(), "anion") == [(1, 1, True)]
    assert ion_regions(setup(**{l1: enter}), "anion") == [(0, 1, True)]
    assert ion_regions(setup(**{l1: enter, l3: enter}), "cation") == [(0, 2, True)]
    assert ion_region_problems(setup(**{l1: enter, l3: enter})) == []
    # the HTL lets them in but they do not move there: SimSS holds the whole region
    last = setup(**{l3: {"ionsMayEnter": 1}})
    assert ion_regions(last, "anion") == [(1, 2, False)]
    assert any("mobility 0 in its last layer" in p for p in ion_region_problems(last))
    # immobile anions in the ETL before the absorber's: SimSS starts the next region there
    start = setup(**{l1: {"N_anion": 3e21}})
    assert ion_regions(start, "anion") == [(0, 0, False), (0, 1, True)]
    assert any("takes in layer 1" in p for p in ion_region_problems(start))
    # mobile anions in the ETL and in the HTL, the absorber between closed to them: two regions
    closed = {"ionsMayEnter": 0, "mu_anion": 0, "N_anion": 0, "mu_cation": 0, "N_cation": 0}
    own = {**enter, "N_anion": 1e21}
    two = setup(**{l1: own, "L2_parameters.txt": closed, l3: own})
    assert ion_regions(two, "anion") == [(0, 0, True), (2, 2, True)]
    assert ion_region_problems(two) == []
    with pytest.raises(NotImplementedError, match="mobility 0"):
        check_supported(last)
    with pytest.raises(NotImplementedError, match="takes in layer 1"):
        build_device(start)
    with pytest.raises(ValueError, match="ionsMayEnter"):
        ion_regions(setup(**{l1: {"N_anion": 1e21, "mu_anion": 1e-12}}), "anion")


def test_voltages_are_simss_own() -> None:
    """`simss_voltages` gives the voltages of every reference run, in SimSS's order (where
    SimSS stopped converging, its `JV.dat` is the beginning of them)."""
    cases = [d for d in sorted(DATA.iterdir()) if (d / "JV.dat").exists()]
    assert any(read_setup(d / "simulation_setup.txt").i("Vdist") == 2 for d in cases)
    for case in cases:
        V = simss_voltages(read_setup(case / "simulation_setup.txt"))
        ref = np.atleast_1d(np.genfromtxt(case / "JV.dat", names=True)["Vint"])
        assert V.size >= ref.size, case.name
        assert np.max(np.abs(V[: ref.size] - ref)) < 1e-10, case.name


def test_logarithmic_downward_scan_reproduces_simss() -> None:
    """`Vdist = 2`, `Vscan = -1`: 25 voltages accumulating towards `Vacc` = 1.4 V, from 1.3 V
    down; SimSS's curve to 1e-5 of its scale at its own voltages."""
    case = DATA / "logv_light"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    V = simss_voltages(setup)
    assert V[0] == pytest.approx(1.3) and V[-1] == pytest.approx(-0.2)
    assert np.all(np.diff(np.abs(np.diff(V))) > 0)  # the steps grow away from Vacc
    res = solve_jv(build_device(setup), V, bernoulli_name="simss")
    assert res.converged.all()
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-5 * np.max(np.abs(ref["Jext"]))


def test_voltages_from_experimental_data_and_simss_checks(tmp_path) -> None:
    """With `useExpData` the voltages are `expJV`'s `Vext` column in the file's order (as SimSS
    5.36 runs them); the voltage parameters SimSS refuses raise `ValueError`."""
    case = DATA / "thin_l1_light"
    files = {f.name: f.read_text() for f in case.iterdir() if f.suffix == ".txt"}
    text = files["simulation_setup.txt"]

    def setup(**values):
        lines = []
        for line in text.splitlines():
            name = line.split("=", 1)[0].strip()
            lines.append(f"{name} = {values[name]}" if name in values else line)
        exp = "Vext Jext\n0.9 10\n0.3 -200\n0.55 -150\n-0.1 -250\n"
        new = {"simulation_setup.txt": "\n".join(lines), "expJV.csv": exp}
        return setup_from_texts({**files, **new})

    assert np.array_equal(simss_voltages(setup(useExpData=1)), [0.9, 0.3, 0.55, -0.1])
    assert np.allclose(simss_voltages(setup(Vscan=-1, Vstep=0.5)), [1.3, 0.8, 0.3, -0.2])
    bad = [
        ({"Vdist": 3}, "Vdist"),
        ({"Vscan": 0}, "Vscan"),
        ({"Vmin": 1.5}, "Vmin"),
        ({"Vstep": 0}, "Vstep"),
        ({"Vdist": 2, "Vacc": 1.0}, "Vacc"),
        ({"Vdist": 2, "Vacc": 2, "Vmin": 0.5, "Vmax": 0.5}, "Vmin = Vmax"),
    ]
    for values, match in bad:
        with pytest.raises(ValueError, match=match):
            simss_voltages(setup(**values))


def test_zimt_steady_states_reproduce_zimt() -> None:
    """ZimT 5.36's input read (`read_tvg`, a ZimT setup built into a device) and its steady
    states (`t = 0` at each line, a return to the steady state each time) solved by the
    steady-state solver: ZimT's `Jext` to 1e-9 of its scale, at two light intensities."""
    import dataclasses

    case = DATA / "zimt_steady_light"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    tvg = read_tvg(setup)
    assert np.all(tvg.t == 0) and np.all(tvg.dti == 0) and not tvg.oc.any()
    ref = np.genfromtxt(case / "tj.dat", names=True)
    assert np.array_equal(ref["Vext"], tvg.Vext) and np.array_equal(ref["G_frac"], tvg.G_frac)
    dev = build_device(setup)
    J = [solve_jv(dataclasses.replace(dev, G_frac=g), np.array([v]), bernoulli_name="simss").J[0]
         for v, g in zip(tvg.Vext, tvg.G_frac, strict=True)]  # fmt: skip
    assert np.max(np.abs(np.array(J) - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))


def test_tvg_as_zimt_reads_it() -> None:
    """`read_tvg`: lines before the header skipped, `oc`, the time steps (forward, or back to
    0), the end at the first empty line, extra columns ignored; ZimT's checks as `ValueError`."""

    def tvg(text):
        return read_tvg(Setup(raw={"T": "295", "tVGFile": "tVG.txt"}, layers=(),
                              files={"tVG.txt": text}))  # fmt: skip

    r = tvg("a comment\nt\tVext  G_frac Track\n0 oc 1 0\n1e-6 0.5 1 1 extra\n3e-6 OC 0 3\n"
            "0 0 1 2\n\n5e-6 0 1 0\n")  # fmt: skip
    assert np.array_equal(r.t, [0, 1e-6, 3e-6, 0])
    assert np.array_equal(r.oc, [True, False, True, False]) and r.Vext[1] == 0.5
    assert np.isnan(r.Vext[0]) and np.array_equal(r.track, [0, 1, 3, 2])
    assert np.allclose(r.dti, [0, 1e6, 5e5, 0]) and np.array_equal(r.G_frac, [1, 1, 0, 1])
    bad = [
        ("0 0 1 0\n", "header"),
        ("t Vext G_frac Track\n1e-6 0 1 0\n", "steady-state"),
        ("t Vext G_frac Track\n", "steady-state"),
        ("t Vext G_frac Track\n0 0 1 0\n2e-6 0 1 0\n1e-6 0 1 0\n", "positive"),
        ("t Vext G_frac Track\n0 0 1 0\n1e-6 0 1 4\n", "reading"),
        ("t Vext G_frac Track\n0 0 1 0\n1e-6 x 1 0\n", "reading"),
        ("t Vext G_frac Track\n0 0 1\n", "reading"),
        ("t Vext G_frac Track\n0 0 1 0\n1e-6 0 1 2\n", "Track=1"),
        ("t Vext G_frac Track\n0 0 1 1\n1e-6 0 1 2\n", "Track=1"),  # the first line's forgotten
        ("t Vext G_frac Track\n0 1e3 1 0\n", "too large"),
    ]
    for text, match in bad:
        with pytest.raises(ValueError, match=match):
            tvg(text)


ZIMT_CASES = ["zimt_photo_light", "zimt_sweep_light", "zimt_sweep_diff_light",
              "zimt_photo_surface_light"]  # fmt: skip
ZIMT_TRAP_CASES = ["zimt_traps_photo_light", "zimt_acceptor_photo_light",
                   "zimt_trapfiles_photo_light", "zimt_iface_sweep_light"]  # fmt: skip
ZIMT_ION_CASES = ["zimt_ions_slow_light", "zimt_ions_fast_light", "zimt_region_slow_light",
                  "zimt_region_diff_light"]  # fmt: skip


def _zimt_case(case):
    setup = read_setup(DATA / case / "simulation_setup.txt")
    return setup, read_tvg(setup), np.genfromtxt(DATA / case / "tj.dat", names=True)


def _check_columns(r, k, ref, qfls=True, spv=False):
    """Device `k`'s `tJFile` columns (`transient.tj_columns`) against ZimT's: every one ZimT
    writes; the currents and recombination as currents to 1e-7 of the scale (the larger of
    `Jext` and the photocurrent), `errJ` to 1e-5, the QFLS (where `qfls`) and voltages to 1e-9 V."""
    from jaxdd.transient import tj_columns

    c = tj_columns(r, k, spv=spv)
    assert set(c) | {"t", "G_frac"} == set(ref.dtype.names)
    scale = max(np.max(np.abs(ref["Jext"])), np.max(ref["JphotoL2"]))
    for name, v in c.items():
        err = np.max(np.abs(v - ref[name]))
        if name.startswith("QFLS"):
            assert not qfls or err < 1e-9, name
        elif name.startswith("V"):
            assert err < 1e-9, name
        else:
            assert err < (1e-5 if name == "errJ" else 1e-7) * scale, name


@pytest.mark.parametrize("case", ZIMT_CASES)
def test_transients_of_electrons_and_holes_reproduce_zimt(case: str) -> None:
    """Stage C's first step (`jaxdd.transient`): ZimT 5.36's `Jext` through 240 steps to 1e-9 of
    its scale (ZimT at tolerances of 1e-12): light on and off at short circuit over six decades
    of time, and a sweep to 1.2 V and back in 2 × 100 ns whose displacement current is half
    the scale, in the integral and the differential form, with finite surface recombination."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case(case)
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    assert np.max(np.abs(r.J_ext[0] - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))


@pytest.mark.parametrize("case", ZIMT_TRAP_CASES)
def test_transients_with_traps_reproduce_zimt(case: str) -> None:
    """Traps in time: ZimT 5.36's `Jext` to 1e-9 of its scale through light on and off at short
    circuit with the absorber's traps neutral, charged (acceptors), and three levels from a
    trap file with two at an interface, and a sweep to 1.2 V and back with charged interface
    traps. Each case's traps following their steady state at once would be 5e-3 to 6e-2 of the
    scale off (the occupations' dynamics are what is tested)."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case(case)
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    assert np.max(np.abs(r.J_ext[0] - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))


@pytest.mark.parametrize("case", ZIMT_ION_CASES)
def test_transients_with_mobile_ions_reproduce_zimt(case: str) -> None:
    """Mobile ions in time (5×5 blocks, each region's rescaling a slot): ZimT 5.36's `Jext` to
    1e-7 of its scale through a sweep and back, ions in the absorber over 2 × 5 s to 1.2 V and
    over 2 × 50 ms to 0.9 V, ions in a region of two layers over 2 × 5 s and, with the
    differential current, 2 × 50 ms to 0.9 V (ZimT's Gummel iteration fails beyond, near the
    open circuit). The hysteresis is 1e-2 to 1e-1 of the scale."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case(case)
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    assert np.max(np.abs(r.J_ext[0] - ref["Jext"])) < 1e-7 * np.max(np.abs(ref["Jext"]))


def test_layer_voltages_as_zimt_writes_them() -> None:
    """`specialOutput = spv`: each layer's voltage drop (`VL1` …) through the sweep to 1.2 V
    and back, with every other column, as ZimT 5.36 writes them."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case("zimt_sweep_spv_light")
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    _check_columns(r, 0, ref, spv=True)


def test_ions_in_two_regions_per_species_against_zimt() -> None:
    """Anions and cations in both transport layers, two regions per species, through a sweep to
    0.9 V and back in 2 × 50 ms (`tests/data/zimt_regions2_light`): ZimT 5.36 keeps each
    species' last region's current alone (docs/differences.md §11), jaxdd every region's by
    default. With `zimt_ion_currents` ZimT's `Jext`, `Jnion` and `Jpion` to 1e-8 of the scale;
    without, the ions' currents those of both regions, 60 and 14 times ZimT's here."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient, tj_columns

    setup, tvg, ref = _zimt_case("zimt_regions2_light")
    batch = prepare([build_device(setup)])
    scale = np.max(np.abs(ref["Jext"]))
    as_zimt = solve_transient(batch, tvg, bernoulli_name="simss", zimt_ion_currents=True)
    assert as_zimt.converged.all()
    c = tj_columns(as_zimt)
    for name in ("Jext", "Jnion", "Jpion"):
        assert np.max(np.abs(c[name] - ref[name])) < 1e-8 * scale, name
    every = tj_columns(solve_transient(batch, tvg, bernoulli_name="simss"))
    for name in ("Jnion", "Jpion"):
        assert np.max(np.abs(every[name])) > 10 * np.max(np.abs(ref[name])), name


def _first_steps(tvg, n):
    fields = ("t", "Vext", "oc", "G_frac", "track", "dti")
    return dataclasses.replace(tvg, **{k: getattr(tvg, k)[:n] for k in fields})


TANGENT_CASES = [("zimt_traps_photo_light", "N_t_bulk", 40), ("zimt_tpv_light", "C_p_bulk", 40),
                 ("zimt_ions_fast_light", "mu_anion", 20)]  # fmt: skip


@pytest.mark.parametrize(("case", "name", "steps"), TANGENT_CASES)
def test_tangents_through_time_are_finite_differences(case: str, name: str, steps: int) -> None:
    """Stage C's gradients (`jax.jvp` through `solve_transient` from `batch.device_tree`): each
    step's tangent by the implicit function at its solution, carried by the state. The
    absorber's trap density through the light switched on (traps filling), the holes' capture
    in the photovoltage at the open circuit (`Vint`'s border), and the anions' mobility through
    a sweep (5×5 blocks): central differences to 1e-5 of their largest (at a step of 1e-3 in
    the logarithm, where the differences' own error is smallest: 2.7e-7 in the first, 8e-6 in
    the photovoltage, whose Newton tolerance enters them)."""
    from jaxdd.batch import device_tree
    from jaxdd.transient import solve_transient

    setup, tvg, _ = _zimt_case(case)
    tvg = _first_steps(tvg, steps)
    structure, p0 = device_parameters(setup)
    oc = bool(tvg.oc.any())

    def curve(x):
        p = {k: jnp.asarray(v) for k, v in p0.items()}
        p[name] = p[name].at[1].set(jnp.exp(x))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        r = solve_transient(batch, tvg, bernoulli_name="simss", ions=name == "mu_anion",
                            vint=oc)  # fmt: skip
        return r.V_int[0] if oc else r.J_ext[0]

    x0 = jnp.log(jnp.asarray(p0[name])[1])
    _, tangent = jax.jvp(curve, (x0,), (jnp.ones_like(x0),))
    h = 1e-3
    fd = (curve(x0 + h) - curve(x0 - h)) / (2 * h)
    assert np.max(np.abs(tangent - fd)) < 1e-5 * np.max(np.abs(fd))


def test_a_fit_through_time_recovers_its_parameters() -> None:
    """A transient's parameters recovered from it (`validation/fit_transient.py`): the light
    switched on, the absorber's trap density and electron capture from twice and half their
    values, by `least_squares` with the Jacobian through time, to 1e-6."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
    from fit_transient import problem
    from scipy.optimize import least_squares

    residuals, jacobian, true = problem("zimt_traps_photo_light", [("N_t_bulk", 1),
                                                                  ("C_n_bulk", 1)], 40)  # fmt: skip
    start = true + np.log([2.0, 0.5])
    out = least_squares(lambda t: np.asarray(residuals(jnp.asarray(t))), start,
                        jac=lambda t: np.asarray(jacobian(jnp.asarray(t))))  # fmt: skip
    assert np.max(np.abs(out.x - true)) < 1e-6


def test_transients_with_and_without_ions_in_a_batch() -> None:
    """Devices with mobile ions beside devices with traps and with neither, in one batch of 5×5
    blocks (the others' ion rows held): each within 1e-7 of ZimT's scale, the `tJFile`'s
    columns (the ions' currents among them) but the QFLS as `_check_columns`."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    cases = [_zimt_case(c) for c in ("zimt_ions_slow_light", "zimt_region_diff_light",
                                     "zimt_photo_light", "zimt_traps_photo_light")]  # fmt: skip
    batch = prepare([build_device(setup) for setup, _, _ in cases])
    r = solve_transient(batch, [tvg for _, tvg, _ in cases], bernoulli_name="simss")
    assert r.converged.all()
    for k, (_, _, ref) in enumerate(cases):
        assert np.max(np.abs(r.J_ext[k] - ref["Jext"])) < 1e-7 * np.max(np.abs(ref["Jext"]))
        _check_columns(r, k, ref, qfls=False)  # the ions' currents among them


def test_transients_in_a_batch_reproduce_zimt() -> None:
    """The ZimT cases as one batch, each device with its own steps (the integral form beside
    the differential one, infinite surface recombination beside finite, devices with traps of
    one level and of several beside devices without): each within 1e-9, and every column of
    ZimT's `tJFile` (`_check_columns`)."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    cases = [_zimt_case(c) for c in ZIMT_CASES + ZIMT_TRAP_CASES]
    batch = prepare([build_device(setup) for setup, _, _ in cases])
    r = solve_transient(batch, [tvg for _, tvg, _ in cases], bernoulli_name="simss")
    assert r.converged.all()
    for k, (name, (_, _, ref)) in enumerate(zip(ZIMT_CASES + ZIMT_TRAP_CASES, cases, strict=True)):
        assert np.max(np.abs(r.J_ext[k] - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))
        # The QFLS only where every carrier stays above the floor of convergence (the sweeps
        # under light): in the dark, after the light, it is not converged (docs/model.md).
        _check_columns(r, k, ref, qfls=name in ("zimt_sweep_light", "zimt_sweep_diff_light"))


def test_photovoltage_at_open_circuit_reproduces_zimt() -> None:
    """The open circuit (`Vext = oc`), the internal voltage an unknown of each step's Newton
    solve: ZimT 5.36's photovoltage through a light pulse of 1 µs and 1 ms after, 151 steps,
    to 1e-10 V (ZimT's Brent to 1e-12 V); the current stays 0 to 1e-10 of the photocurrent."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case("zimt_tpv_light")
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    assert np.max(np.abs(r.V_int[0] - ref["Vint"])) < 1e-10
    assert np.array_equal(r.V_ext, r.V_int)  # no series resistance
    assert np.max(np.abs(r.J_ext)) < 1e-10 * np.max(ref["JphotoL2"])


def test_open_circuit_in_the_steady_state_is_simss_voc() -> None:
    """`oc` at `t = 0`: SimSS 5.36's `Voc` (its curve at 0.1 mV steps around it, a cubic
    through them) to 1e-10 V."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, _ = _zimt_case("zimt_tpv_light")
    first = dataclasses.replace(tvg, **{k: getattr(tvg, k)[:1] for k in
                                        ("t", "Vext", "oc", "G_frac", "track", "dti")})  # fmt: skip
    r = solve_transient(prepare([build_device(setup)]), first, bernoulli_name="simss")
    jv = np.genfromtxt(DATA / "thin_l1_voc_light" / "JV.dat", names=True)
    roots = np.roots(np.polyfit(jv["Vext"] - jv["Vext"].mean(), jv["Jext"], 3))
    roots = roots[np.abs(roots.imag) == 0].real + jv["Vext"].mean()
    voc = roots[(roots > jv["Vext"].min()) & (roots < jv["Vext"].max())]
    assert r.converged.all() and voc.size == 1
    assert abs(r.V_int[0, 0] - voc[0]) < 1e-10


def test_series_resistance_in_time_reproduces_zimt() -> None:
    """`R_series` with a shunt in time, `Vint` the unknown whose `Vint + Jext·R_series` is the
    given `Vext`: ZimT 5.36's `Jext` through the sweep to 1.2 V and back in 2 × 100 ns to 1e-9
    of its scale, `Vint` to 1e-9 V (`Vint` differs from `Vext` by up to 0.65 V)."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    setup, tvg, ref = _zimt_case("zimt_rseries_sweep_light")
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    assert r.converged.all()
    assert np.max(np.abs(r.J_ext[0] - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(r.V_int[0] - ref["Vint"])) < 1e-9
    assert np.max(np.abs(r.V_ext[0] - tvg.Vext)) < 1e-9


@pytest.mark.parametrize("case", ["zimt_track_tpv_light", "zimt_track_tpv3_light"])
def test_automatic_stop_reproduces_zimt(case: str) -> None:
    """ZimT's automatic stop (`Track`, five equal states in a row) on the photovoltage after a
    pulse: each step compared with the steady state before the pulse (`Track` 1 there, 2 after
    it), stopped after 118 of 182 steps, and each with the one before (`Track` 3), after 119,
    as ZimT 5.36, within one step: the states compare within ZimT's own tolerance (`tolPois`
    1e-12 V), so where their difference crosses it ZimT's own error decides the step. `Vint`
    of the steps kept to 1e-10 V."""
    from jaxdd.batch import prepare
    from jaxdd.transient import STOP_STATIC, solve_transient

    setup, tvg, ref = _zimt_case(case)
    r = solve_transient(prepare([build_device(setup)]), tvg, bernoulli_name="simss")
    rows = int(r.accepted[0].sum())
    assert r.stop[0] == STOP_STATIC and abs(rows - len(ref)) <= 1 and rows < len(tvg.t)
    assert r.accepted[0, :rows].all() and r.converged[0, :rows].all()
    rows = min(rows, len(ref))
    assert np.max(np.abs(r.V_int[0, :rows] - ref["Vint"][:rows])) < 1e-10


def test_states_compare_as_zimt_compares_them() -> None:
    """`_states_equal` as ZimT's `States_Are_Equal`: never at another `G_frac` or with the
    potentials apart by `tolPois`; then by `convVar` the densities (1), the current within
    `tolCurr` or the smaller `errJ` (2), either (3) or both (4)."""
    from jaxdd.transient import _states_equal

    base = {"V": jnp.zeros(4), "n": jnp.ones(4), "p": jnp.ones(4), "a": jnp.zeros(4),
            "c": jnp.zeros(4), "J": 1.0, "err": 0.0, "G": 1.0}  # fmt: skip
    dens_off = dict(base, n=jnp.ones(4) * (1 + 1e-5))  # densities apart, current equal
    curr_off = dict(base, J=1.1)  # the current apart, densities equal
    ctl = {"tol_pois": 1e-6, "tol_dens": 1e-6, "tol_curr": 1e-3}
    expect = {1: (False, True), 2: (True, False), 3: (True, True), 4: (False, False)}
    for cv, (with_dens_off, with_curr_off) in expect.items():
        c = dict(ctl, conv_var=cv)
        assert bool(_states_equal(base, base, c))
        assert bool(_states_equal(base, dens_off, c)) == with_dens_off, cv
        assert bool(_states_equal(base, curr_off, c)) == with_curr_off, cv
        assert not _states_equal(base, dict(base, G=0.5), c)
        assert not _states_equal(base, dict(base, V=jnp.full(4, 1e-6)), c)
    loose = dict(curr_off, err=0.2)  # within both states' errJ
    assert bool(_states_equal(dict(base, err=0.3), loose, dict(ctl, conv_var=2)))


def test_failure_modes_as_zimt_treats_them() -> None:
    """`failureMode`, with steps made to fail (two Newton steps at most): 1 keeps every step,
    0 stops at the first failed one (not kept), 2 skips each failed step (not a steady state)
    and takes the next one's Δt from the last step kept, so that its steps kept are those of a
    run through them alone; and stops where the step kept before a failed one is the
    reference (`Track` 1)."""
    from jaxdd.batch import prepare
    from jaxdd.transient import STOP_FAILED, solve_transient

    setup, tvg, _ = _zimt_case("zimt_photo_light")
    batch = prepare([build_device(setup)])

    def run(tvg_, **kw):
        return solve_transient(batch, tvg_, max_iter=2, bernoulli_name="simss", **kw)

    keep_all = run(dataclasses.replace(tvg, failure_mode=1))
    failed = ~keep_all.converged[0]
    assert 0 < failed.sum() < len(failed) - 1 and keep_all.accepted.all() and keep_all.stop[0] == 0
    first = int(np.argmax(failed))
    stop = run(dataclasses.replace(tvg, failure_mode=0))
    assert stop.stop[0] == STOP_FAILED
    assert stop.accepted[0, :first].all() and not stop.accepted[0, first:].any()
    skip = run(dataclasses.replace(tvg, failure_mode=2))
    kept = skip.accepted[0]
    assert skip.stop[0] == 0 and np.array_equal(kept, skip.converged[0] | (tvg.t == 0))
    assert (~kept).sum() > 0
    fields = ("t", "Vext", "oc", "G_frac", "track", "dti")
    alone = run(dataclasses.replace(tvg, **{k: getattr(tvg, k)[kept] for k in fields}))
    assert np.array_equal(alone.converged[0], skip.converged[0, kept])
    assert np.max(np.abs(alone.J[0] - skip.J[0, kept])) < 1e-9 * np.max(np.abs(alone.J))
    k = int(np.flatnonzero(~kept)[0])  # the step before the first one skipped as a reference
    track = tvg.track.copy()
    track[k - 1] = 1
    ref_stop = run(dataclasses.replace(tvg, failure_mode=2, track=track))
    assert ref_stop.stop[0] == STOP_FAILED and not ref_stop.accepted[0, k:].any()


def test_series_resistance_beside_given_voltages_and_ions_in_a_batch() -> None:
    """The internal voltage an unknown of every device where one device needs it: the series
    resistance beside a device with the voltage given (its row `Vint = Vext`) and one with
    mobile ions (5×5 blocks with the voltage's border), each as it is alone."""
    from jaxdd.batch import prepare
    from jaxdd.transient import solve_transient

    cases = [_zimt_case(c) for c in ("zimt_rseries_sweep_light", "zimt_photo_light",
                                     "zimt_ions_fast_light")]  # fmt: skip
    batch = prepare([build_device(setup) for setup, _, _ in cases])
    r = solve_transient(batch, [tvg for _, tvg, _ in cases], bernoulli_name="simss")
    assert r.converged.all()
    for k, ((_, _, ref), tol) in enumerate(zip(cases, (1e-9, 1e-9, 1e-7), strict=True)):
        assert np.max(np.abs(r.J_ext[k] - ref["Jext"])) < tol * np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(r.V_int[1:] - np.stack([t.Vext for _, t, _ in cases[1:]]))) < 1e-12


@pytest.mark.parametrize("case", ["frozen_1e17_x1_thin_light", "steady_1e17_thin_light"])
def test_jv_with_ions_reproduces_simss(case: str) -> None:
    """Frozen after a pre-bias, and free at every voltage: SimSS's curve to 2e-5 of its scale."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < 2e-5 * scale


@pytest.mark.parametrize(
    ("case", "tol"),
    [("iface_charged_light", 1e-5), ("iface_ions_light", 2e-5), ("pin_iface_light", 1e-5)],
)
def test_jv_with_interface_traps_reproduces_simss(case: str, tol: float) -> None:
    """Stage B, interface traps: a donor and an acceptor level (3e14 m⁻² each) at the two
    interfaces of the 1 nm stack, the same with free ions, and a p-i-n stack with 10 and 20 nm
    transport layers and 5e13 and 4.6e14 m⁻² of donor traps (SimSS at tolerances of 1e-9; at its
    defaults it stopped early there by 1.2e-4, and from 1.0 V on it does not converge at all). The
    traps move these curves by up to two thirds of their scale."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < tol * scale


@pytest.mark.parametrize("case", ["bulk_acceptor_light", "trapfiles_light"])
def test_jv_with_charged_bulk_traps_and_trap_files_reproduces_simss(case: str) -> None:
    """Stage B, bulk traps: the absorber's 3.9e21 m⁻³ at mid-gap as acceptors, and three donor
    levels from a trap file with two acceptor levels from an interface trap file at one
    interface and one donor level at the other (SimSS at tolerances of 1e-9). The traps' charge
    moves these curves by 4–5 % of their scale; jaxdd is within 1e-9 of it."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all() and res.iterations.max() <= 8
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-9 * scale


def test_finite_surface_recombination_reproduces_simss() -> None:
    """Stage B, the contacts: 20 m/s for the electrons at the left electrode and 50 m/s for the
    holes at the right one (SimSS at tolerances of 1e-9), finite only while extracted. Within
    1e-9 of SimSS's scale at every voltage but one: at 1.05 V the electrons at the left contact
    turn from extracted to injected, and SimSS takes the contact's branch from the voltage before
    (docs/differences.md); there jaxdd's current is the one whose branch holds at its solution."""
    setup = read_setup(DATA / "surface_light" / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / "surface_light" / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    lagged = np.isclose(ref["Vint"], 1.05)
    assert np.max(np.abs(res.J - ref["Jext"])[~lagged]) < 1e-9 * scale
    assert res.J[lagged][0] > ref["Jext"][lagged][0] + 50.0  # 110.0 against SimSS's 43.6 A/m²


def _texts(case: str) -> dict[str, str]:
    return {f.name: f.read_text() for f in (DATA / case).iterdir() if f.suffix == ".txt"}


def _put(text: str, name: str, value: str) -> str:
    """`text` with the parameter `name` set to `value`, its comment kept."""
    import re

    new, count = re.subn(rf"^{name} = \S+", f"{name} = {value}", text, flags=re.M)
    assert count == 1, name
    return new


def _mirrored(texts: dict[str, str]) -> dict[str, str]:
    """A three-layer input turned around: the layers reversed, the work functions and the
    surface recombination velocities exchanged, the left electrode the other one."""
    import re

    t = dict(texts)
    setup = t["simulation_setup.txt"]

    def get(name):
        return re.search(rf"^{name} = (\S+)", setup, flags=re.M).group(1)

    files = [get(f"l{k}") for k in (1, 2, 3)]
    pairs = [("W_L", "W_R"), ("S_n_L", "S_n_R"), ("S_p_L", "S_p_R")]
    values = {name: get(name) for pair in pairs for name in pair}
    for k, f in zip((1, 2, 3), reversed(files), strict=True):
        setup = _put(setup, f"l{k}", f)
    for a, b in pairs:
        setup = _put(_put(setup, a, values[b]), b, values[a])
    t["simulation_setup.txt"] = _put(setup, "leftElec", str(-int(get("leftElec"))))
    return t


def test_surface_recombination_at_the_other_two_ends_is_the_mirror_image() -> None:
    """The two ends where SimSS 5.36 has the sign of `S` wrong (the electrons at the right
    electrode, the holes at the left; docs/differences.md): the uniformly lit 1 nm stack without
    band offsets, 100 m/s there, against the same device turned around (`leftElec = 1`), where
    the same carriers meet `S_n_L` and `S_p_R`, the ends SimSS has right. The same curve to
    1e-11, and the minority carriers' recombination at the contacts costs most of the current
    at 1 V (an infinite `S` 44 A/m² of 877)."""
    from jaxdd.simss_input import setup_from_texts

    t = _texts("none_light")
    for f in ("L1_parameters.txt", "L3_parameters.txt"):
        t[f] = _put(_put(t[f], "E_c", "4"), "E_v", "5.6")
    finite = dict(t)
    setup = _put(_put(t["simulation_setup.txt"], "S_n_R", "1e2"), "S_p_L", "1e2")
    finite["simulation_setup.txt"] = setup
    V = np.round(np.arange(-0.2, 1.2 + 1e-9, 0.05), 6)
    curves = {}
    for name, texts in (("infinite", t), ("finite", finite), ("mirrored", _mirrored(finite))):
        res = solve_jv(build_device(setup_from_texts(texts)), V, bernoulli_name="simss")
        assert res.converged.all(), name
        curves[name] = res.J
    scale = np.max(np.abs(curves["finite"]))
    assert np.max(np.abs(curves["finite"] - curves["mirrored"])) < 1e-11 * scale
    at_1V = np.isclose(V, 1.0)
    assert curves["finite"][at_1V][0] < 10 * curves["infinite"][at_1V][0] < 0


@pytest.mark.parametrize("field", [False, True])
@pytest.mark.parametrize("surface", ["extraction", "robin"])
def test_the_contacts_jacobian_is_autodiff(surface: str, field: bool) -> None:
    """`_assemble`'s blocks with finite surface recombination (each contact's density a function
    of its neighbour's unknowns, the chain rule added to the first and last block) are
    `jax.jacfwd` of its residual, at a perturbed state; also with a field-dependent mobility in
    the contacts' layers (the contact's interval in its own field)."""
    from jaxdd import solver as S
    from jaxdd.simss_input import setup_from_texts
    from jaxdd.solver import _assemble, _ion_setup, _stencils, bernoulli

    texts = _texts("surface_light")
    if field:
        for f, g in (("L1_parameters.txt", "gamma_n"), ("L3_parameters.txt", "gamma_p")):
            texts[f] = _put(_put(texts[f], f"mob{g[-1]}Dep", "1"), g, "3e-4")
    dev = build_device(setup_from_texts(texts), surface=surface)
    s = jax.tree_util.tree_map(jnp.asarray, _stencils(dev))
    y0 = S._initial(dev, 0.6)
    NP = len(dev.x) - 2
    noise = np.zeros_like(y0)
    noise[1:-1] = np.random.default_rng(3).normal(0, 1, (NP, 3)) * np.array([dev.Vt, 1.0, 1.0])
    full = jnp.asarray(y0 + noise)
    ion = jax.tree_util.tree_map(jnp.asarray, _ion_setup(dev))
    la, lc = S._point_logs(S._ion_logs(full[:, 0], ion), ion)

    def residual(y):
        return _assemble(full.at[1:-1].set(y), s, dev.Vt, dev.G_frac, la, lc, bernoulli)[0]

    dense = np.asarray(jax.jacfwd(residual)(full[1:-1]))
    _, A, B, C, _ = (np.asarray(a) for a in _assemble(full, s, dev.Vt, dev.G_frac, la, lc,
                                                      bernoulli))  # fmt: skip
    i = np.arange(NP)
    rows = np.max(np.abs(dense), axis=(2, 3))[:, :, None]
    for block, shift in ((B, 0), (A[1:], -1), (C[:-1], 1)):
        k = i[max(0, -shift) : NP - max(0, shift)]
        assert np.max(np.abs(block - dense[k, :, k + shift, :]) / rows[k]) < 1e-11
    assert np.abs(B[0] - dense[0, :, 0, :]).max() < 1e-11 * rows[0].max()


def test_sfb_work_functions_and_the_gap_check() -> None:
    """'sfb' gives each work function the adjacent layer's equilibrium Fermi level (mid-gap for
    an undoped one, as SIMsalabim's `ObtainWorkFunction`); a work function outside its layer's
    gap is refused."""
    from jaxdd.simss_input import setup_from_texts

    t = _texts("thin_l1_light")
    t["L3_parameters.txt"] = _put(t["L3_parameters.txt"], "N_A", "1e24")
    sfb = dict(t, **{"simulation_setup.txt": _put(_put(t["simulation_setup.txt"], "W_L", "sfb"),
                                                  "W_R", "sfb")})  # fmt: skip
    dev = build_device(setup_from_texts(sfb))
    assert abs(dev.W_L - 0.5 * (4 + 6.6)) < 1e-12
    assert abs(dev.W_R - (5.6 - dev.Vt * np.log(2.2237571e24 / 1e24))) < 1e-12
    bad = dict(t, **{"simulation_setup.txt": _put(t["simulation_setup.txt"], "W_L", "3.9")})
    with pytest.raises(ValueError, match="W_L"):
        build_device(setup_from_texts(bad))


def test_the_gradient_in_the_surface_recombination_is_finite_differences() -> None:
    """The curve's derivative in both finite surface recombination velocities, through
    `device_tree` and the converged solution, against central finite differences (a step of
    1e-3: close to the open-circuit voltage the curve's round-off is 1e-7 A/m²)."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "surface_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 1.0 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        p = dict(params, S_n_L=params["S_n_L"] * jnp.exp(a), S_p_R=params["S_p_R"] * jnp.exp(-a))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        J, ok, _ = solve_batch(batch, V, strict=True)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-3
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-5 * abs(fd)


@pytest.mark.parametrize("case", ["langevin_light", "langevin_field_light"])
def test_langevin_recombination_reproduces_simss(case: str) -> None:
    """Stage B, Langevin recombination in all three layers (`preLangevin` 1e-2 in the absorber,
    1e-3 in the transport layers; SimSS at tolerances of 1e-9), which moves the curve by 2 % of
    its scale; with constant mobilities, and with `fieldmob_light`'s field-dependent ones (the
    rate in the field of its interval). With the mobilities SimSS takes (`langevin="simss"`)
    within 1e-9 of its scale; with each point's own layer's (the default; docs/differences.md
    §8) 3.7e-5 and 6.2e-5 off, the last point of each layer but the last."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    scale = np.max(np.abs(ref["Jext"]))
    gap = {}
    for langevin in ("simss", "layer"):
        res = solve_jv(build_device(setup, langevin=langevin), ref["Vint"], bernoulli_name="simss")
        assert res.converged.all()
        gap[langevin] = np.max(np.abs(res.J - ref["Jext"])) / scale
    assert gap["simss"] < 1e-9
    assert 1e-5 < gap["layer"] < 1e-4


@pytest.mark.parametrize("case", ["langevin_light", "langevin_field_light"])
def test_the_gradient_in_the_langevin_prefactor_is_finite_differences(case: str) -> None:
    """The curve's derivative in the absorber's `preLangevin`, through `device_tree` and the
    converged solution, against central finite differences; with constant and with
    field-dependent mobilities."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / case / "simulation_setup.txt"))
    V = np.round(np.arange(-0.1, 1.0 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        pre = jnp.asarray(params["preLangevin"]).at[1].multiply(jnp.exp(a))
        batch = jax.tree_util.tree_map(
            lambda v: v[None], device_tree(structure, dict(params, preLangevin=pre))
        )
        J, ok, _ = solve_batch(batch, V, strict=True)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-3
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-5 * abs(fd)


def test_the_field_dependent_mobility_reproduces_simss() -> None:
    """Stage B: SIMsalabim's field-dependent mobility (`mobnDep`, `mobpDep` = 1), `μ·exp(γ·√|F|)`
    on each interval in its own field, the interfaces' transfer mobility left alone: γ from 0
    to 3e-4 (m/V)^½ in the three layers, the electrons' constant in the last (the curve moves
    by 32 %); SimSS at a tolerance of 1e-9, in as many Newton steps as without."""
    setup = read_setup(DATA / "fieldmob_light" / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / "fieldmob_light" / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all() and res.iterations.max() <= 16
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-9 * np.max(np.abs(ref["Jext"]))


def test_the_gradient_in_the_field_dependence_is_finite_differences() -> None:
    """The curve's derivative in the absorber's `gamma_n` and `gamma_p`, through `device_tree`
    and the converged solution, against central finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "fieldmob_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 1.0 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        g = {k: jnp.asarray(params[k]).at[1].multiply(1 + a * c)
             for k, c in (("gamma_n", 1.0), ("gamma_p", -0.5))}  # fmt: skip
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, params | g))
        J, ok, _ = solve_batch(batch, V, strict=True)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-4
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_a_mobility_model_other_than_0_or_1_is_refused() -> None:
    """`mobnDep` other than 0 or 1 is refused, as SIMsalabim does."""
    from jaxdd.simss_input import setup_from_texts

    texts = _texts("fieldmob_light")
    bad = dict(texts, **{"L2_parameters.txt": _put(texts["L2_parameters.txt"], "mobnDep", "2")})
    with pytest.raises(ValueError, match="mobnDep"):
        build_device(setup_from_texts(bad))


def test_the_field_dependent_generation_reproduces_simss() -> None:
    """Stage B: Braun's field-dependent generation (`fieldDepG`) in all three layers, a
    Gaussian distance distribution in the absorber (`P0` 0.1), an exponential in the first
    layer and a delta function in the last (k_f 1e9, a 0.5–0.6 nm), which halves J_sc. With
    SimSS's Romberg integration and Bessel series (`braun="simss"`) within 1e-9 of SimSS's
    scale at its tolerances of 1e-9; exact (the default) within 1e-7, its integration that
    close on this device (docs/differences.md §9). Every third voltage, for time."""
    setup = read_setup(DATA / "braun_light" / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / "braun_light" / "JV.dat", names=True)[::3]
    scale = np.max(np.abs(ref["Jext"]))
    for rule, tol in (("simss", 1e-9), ("exact", 1e-7)):
        res = solve_jv(build_device(setup, braun=rule), ref["Vint"], bernoulli_name="simss")
        assert res.converged.all()
        assert np.max(np.abs(res.J - ref["Jext"])) < tol * scale, rule


def _braun_stencils(eps_r, a, k_f, kind, romberg=False):
    """`_dissociation`'s stencils for one point of each kind given, at `T` 295 K."""
    from jaxdd.constants import EPS_0, K_B, Q

    Vt, eps, n = K_B * 295 / Q, eps_r * EPS_0, len(kind)
    s = {"br_on": np.ones(n), "br_kind": np.asarray(kind, float),
         "br_lpre": np.full(n, np.log(3 / (4 * np.pi * a**3))),
         "br_cE": np.full(n, Q / (4 * np.pi * eps * a * Vt)),
         "br_bq": np.full(n, Q / (8 * np.pi * eps * Vt**2)), "br_cF": np.ones(n),
         "br_lkf": np.full(n, np.log(k_f))}  # fmt: skip
    if romberg:
        s["br_romberg"] = np.zeros(n)
    return {k: jnp.asarray(v) for k, v in s.items()}, Vt, eps


@pytest.mark.parametrize("F", [0.0, 1e7, 1e8])
@pytest.mark.parametrize(("eps_r", "a", "k_f"), [(24.1, 1e-9, 1e6), (3.5, 1.5e-9, 1e5),
                                                 (3.5, 3e-10, 1e4)])  # fmt: skip
def test_the_dissociation_probability_is_braun_integrated(eps_r, a, k_f, F) -> None:
    """Invariant: `_dissociation` (exact) is Braun's probability averaged over each distance
    distribution, against an adaptive quadrature to 1e-12, with the exact Bessel function; and
    SimSS's rule (`br_romberg`) is a direct port of its `RombergIntegrationValues` and series
    `Bessel` to round-off."""
    integrate = pytest.importorskip("scipy.integrate")
    from scipy.special import i1

    from jaxdd.constants import Q
    from jaxdd.solver import BRAUN_LIMITS, _dissociation

    k = 1e-17
    dens = {2: lambda u: 4 / np.sqrt(np.pi) * u**2 * np.exp(-(u**2)), 3: lambda u: np.exp(-u),
            4: lambda u: u**2 * np.exp(-u) / 2,
            5: lambda u: 8 * u**4 * np.exp(-(u**2)) / (3 * np.sqrt(np.pi))}  # fmt: skip

    def p_delta(u, Vt, eps, bessel):
        b = Q * F / (8 * np.pi * eps * Vt**2)
        kd = 3 * k / (4 * np.pi * (a * u) ** 3) * np.exp(-Q / (4 * np.pi * eps * a * u * Vt))
        kd = kd * bessel(b)
        return kd / (kd + k_f)

    def exact_bessel(b):
        z = np.sqrt(8 * b)
        return 2 * i1(z) / z if b > 0 else 1.0

    def simss_bessel(x):
        r = 1 + x / 66
        for c in (55, 45, 36, 28, 21, 15, 10, 6, 3):
            r = 1 + x / c * r
        return 1 + x * r

    def romberg(f, lo, hi, tol=0.01, max_it=15):  # SIMsalabim's, line by line
        h = hi - lo
        prev, i, d_new, d_old = [(f(lo) + f(hi)) / 2 * h], 1, 1.0, 1.0
        while (d_new > tol or d_old > tol) and i < max_it:
            d_old, i = d_new, i + 1
            s = sum(f(lo + (m - 0.5) * h) for m in range(1, 2 ** (i - 2) + 1))
            row = [0.5 * (prev[0] + h * s)]
            for j in range(2, i + 1):
                row.append(row[j - 2] + (row[j - 2] - prev[j - 2]) / (4 ** (j - 1) - 1))
            h, d_new, prev = h / 2, abs((row[i - 1] - prev[i - 2]) / row[i - 1]), row
        return prev[i - 1]

    kinds = [1, 2, 3, 4, 5]
    for romberg_rule in (False, True):
        s, Vt, eps = _braun_stencils(eps_r, a, k_f, kinds, romberg_rule)
        got = np.asarray(_dissociation(jnp.full(5, F), jnp.full(5, k), s))
        bessel = simss_bessel if romberg_rule else exact_bessel
        for idx, kind in enumerate(kinds):

            def g(u, kind=kind, Vt=Vt, eps=eps, bessel=bessel):
                return dens[kind](u) * p_delta(u, Vt, eps, bessel)

            if kind == 1:
                want = p_delta(1.0, Vt, eps, bessel)
            elif romberg_rule:
                want = romberg(g, *BRAUN_LIMITS)
            else:
                want = integrate.quad(g, *BRAUN_LIMITS, epsabs=0, epsrel=1e-13, limit=2000,
                                      points=list(np.geomspace(0.02, 10, 40)))[0]  # fmt: skip
            assert abs(got[idx] - want) < 1e-12 * max(want, 1e-3), (romberg_rule, kind)


def test_the_gradient_in_the_braun_parameters_is_finite_differences() -> None:
    """The curve's derivative in the absorber's `a`, `k_f` and `P0`, through `device_tree` and
    the converged solution, against central finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "braun_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 1.0 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(x):
        moved = {k: jnp.asarray(params[k]).at[1].multiply(1 + x * c)
                 for k, c in (("a", 0.5), ("k_f", -1.0), ("P0", 2.0))}  # fmt: skip
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, params | moved))
        J, ok, _ = solve_batch(batch, V, strict=True)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-4
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_field_dependent_generation_inputs_are_checked() -> None:
    """As SIMsalabim: `0 ≤ P0 < 1`, 0 without `fieldDepG`, `thermLengDist` 1–5, `k_f > 0`; a
    batch takes one Braun rule."""
    from jaxdd.batch import prepare
    from jaxdd.simss_input import setup_from_texts

    texts = _texts("braun_light")
    for name, value in (("P0", "1"), ("thermLengDist", "6"), ("k_f", "0")):
        bad = dict(texts, **{"L2_parameters.txt": _put(texts["L2_parameters.txt"], name, value)})
        with pytest.raises(ValueError, match=name):
            build_device(setup_from_texts(bad))
    off = dict(texts, **{"L2_parameters.txt": _put(texts["L2_parameters.txt"], "fieldDepG", "0")})
    with pytest.raises(ValueError, match="P0"):
        build_device(setup_from_texts(off))
    setup = setup_from_texts(texts)
    with pytest.raises(ValueError, match="Braun rule"):
        prepare([build_device(setup), build_device(setup, braun="simss")])


def test_series_and_shunt_resistance_are_simss_external_circuit() -> None:
    """Stage B: SimSS's external circuit, `J_ext = J + V_int/R_shunt`, `V_ext = V_int +
    J_ext·R_series`, with 2e-4 and 0.3 Ω m² (the series resistance moves the voltage by 1.8 V at
    the top of the scan): both solvers' `J` and `V_ext` against SimSS's."""
    from jaxdd.batch import external_voltages, prepare, solve_batch

    setup = read_setup(DATA / "resist_light" / "simulation_setup.txt")
    check_supported(setup)
    dev = build_device(setup)
    ref = np.genfromtxt(DATA / "resist_light" / "JV.dat", names=True)
    scale = np.max(np.abs(ref["Jext"]))
    single = solve_jv(dev, ref["Vint"], bernoulli_name="simss")
    batch = prepare([dev])
    J, ok, _ = solve_batch(batch, ref["Vint"], bernoulli_name="simss")
    V_ext = np.asarray(external_voltages(batch, ref["Vint"], J))[0]
    assert single.converged.all() and np.asarray(ok).all()
    for got_J, got_V in ((single.J, single.V_ext), (np.asarray(J)[0], V_ext)):
        assert np.max(np.abs(got_J - ref["Jext"])) < 1e-5 * scale
        assert np.max(np.abs(got_V - ref["Vext"])) < 1e-4


@pytest.mark.parametrize("case", ["calc_light", "pin_calc_light"])
def test_the_transfer_matrix_profile_is_simss(case: str) -> None:
    """Stage B: the generation profile by the transfer-matrix method (`genProfile = calc`) on
    SimSS's grid is SimSS's to round-off (its variable file, `G_simss.dat`): glass, ITO, the
    layers, gold or silver, AM1.5G from 350 to 800 nm; the 1 nm stack, and 20/500/10 nm."""
    dev = build_device(read_setup(DATA / case / "simulation_setup.txt"))
    ref = np.loadtxt(DATA / case / "G_simss.dat", skiprows=1)
    assert np.max(np.abs(dev.x - ref[:, 0])) < 1e-15
    assert np.max(np.abs(dev.G[1:-1] - ref[1:-1, 1])) < 1e-10 * np.max(ref[:, 1])


@pytest.mark.parametrize("case", ["calc_light", "pin_calc_light", "none_light"])
def test_jv_with_computed_or_uniform_generation_reproduces_simss(case: str) -> None:
    """The curves with the transfer-matrix profile, and with a uniform one (`genProfile = none`,
    each layer's `G_ehp`), against SimSS's."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-5 * np.max(np.abs(ref["Jext"]))


def test_the_left_electrode_as_the_anode_reproduces_simss() -> None:
    """Stage B, `leftElec = 1`: a p-i-n cell in its own orientation, lit through glass, ITO and
    its 10 nm HTL, with interface traps, free ions, the transfer-matrix profile and a shunt; both
    solvers against SimSS at tolerances of 1e-9 (the voltages at the contacts and the currents'
    signs SIMsalabim's)."""
    from jaxdd.batch import prepare, solve_batch

    setup = read_setup(DATA / "pin_anode_left_light" / "simulation_setup.txt")
    check_supported(setup)
    dev = build_device(setup)
    ref = np.genfromtxt(DATA / "pin_anode_left_light" / "JV.dat", names=True)
    scale = np.max(np.abs(ref["Jext"]))
    single = solve_jv(dev, ref["Vint"], bernoulli_name="simss")
    J, ok, _ = solve_batch(prepare([dev]), ref["Vint"], bernoulli_name="simss")
    assert single.converged.all() and np.asarray(ok).all()
    assert np.max(np.abs(single.J - ref["Jext"])) < 2e-5 * scale
    assert np.max(np.abs(np.asarray(J)[0] - ref["Jext"])) < 2e-5 * scale


def test_the_gradient_through_the_optics_is_finite_differences() -> None:
    """The absorber's thickness moves the grid and the optics (interference in the 20/500/10 nm
    stack): the derivative of the short-circuit current, through `device_tree`, against central
    finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "pin_calc_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = jnp.array([0.0, 0.5])

    def jsc(scale):
        p = dict(params, L=params["L"] * jnp.array([1.0, scale, 1.0]))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        return solve_batch(batch, V)[0][0, 0]

    _, by_jvp = jax.jvp(jsc, (1.0,), (1.0,))
    h = 1e-5
    fd = (float(jsc(1 + h)) - float(jsc(1 - h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-5 * abs(fd)


@pytest.mark.parametrize(
    "with_",
    [["iface_charged_light", "iface_ions_light"], ["bulk_acceptor_light", "trapfiles_light"],
     ["surface_light", "fieldmob_light"], ["langevin_field_light", "braun_light"]],
    ids=["interface", "bulk", "contacts-mobility", "langevin-braun"],
)  # fmt: skip
def test_a_batch_mixes_devices_with_and_without_modules(with_: list[str]) -> None:
    """`prepare` gives a device without a module what does nothing in it — no interface traps
    on any edge, no charged bulk traps, infinite surface recombination, no field dependence of
    the mobility, Langevin's rate or the generation — and every device the most trap levels of
    any (the others of density 0): each curve of a batch of `thin_l1_light` (none of them) and
    devices with them is the one-device solver's. One batch per pair of modules: the cost of a
    batch is its tracing, which grows with the modules it has."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", *with_]
    devices = [build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    J, ok, _ = solve_batch(prepare(devices), V, bernoulli_name="simss")
    assert np.asarray(ok).all()
    for k, dev in enumerate(devices):
        single = solve_jv(dev, V, bernoulli_name="simss")
        assert np.max(np.abs(np.asarray(J[k]) - single.J)) < 1e-6 * np.max(np.abs(single.J))


@pytest.mark.parametrize(
    "case", ["iface_charged_light", "iface_ions_light", "pin_iface_light", "trapfiles_light"]
)
def test_the_interface_traps_jacobian_is_autodiff(case: str) -> None:
    """`_assemble`'s blocks with interface traps (their capture in both continuity equations,
    their charge in Poisson's, both coupling the two points of an interface) are `jax.jacfwd`
    of its residual in the whole state, which is block tridiagonal: at a perturbed state."""
    from jaxdd import solver as S
    from jaxdd.solver import _assemble, _ion_setup, _stencils, bernoulli

    dev = build_device(read_setup(DATA / case / "simulation_setup.txt"))
    s = jax.tree_util.tree_map(jnp.asarray, _stencils(dev))
    y0 = S._initial(dev, 0.6)
    NP = len(dev.x) - 2
    noise = np.zeros_like(y0)
    noise[1:-1] = np.random.default_rng(2).normal(0, 1, (NP, 3)) * np.array([dev.Vt, 1.0, 1.0])
    full = jnp.asarray(y0 + noise)
    ion = jax.tree_util.tree_map(jnp.asarray, _ion_setup(dev))
    la, lc = S._point_logs(S._ion_logs(full[:, 0], ion), ion)

    def residual(y):
        return _assemble(full.at[1:-1].set(y), s, dev.Vt, dev.G_frac, la, lc, bernoulli)[0]

    dense = np.asarray(jax.jacfwd(residual)(full[1:-1]))  # NP × 3 × NP × 3
    _, A, B, C, _ = (np.asarray(a) for a in _assemble(full, s, dev.Vt, dev.G_frac, la, lc,
                                                      bernoulli))  # fmt: skip
    i = np.arange(NP)
    rows = np.max(np.abs(dense), axis=(2, 3))[:, :, None]
    for block, shift in ((B, 0), (A[1:], -1), (C[:-1], 1)):
        k = i[max(0, -shift) : NP - max(0, shift)]
        ref = dense[k, :, k + shift, :]
        assert np.max(np.abs(block - ref) / rows[k]) < 1e-11
    band = np.abs(i[:, None] - i[None, :]) <= 1
    assert np.all(dense.transpose(0, 2, 1, 3)[~band] == 0.0)
    assert dev.interface is not None and np.count_nonzero(dev.interface["Nt"]) >= 1


def test_free_ions_keep_their_number() -> None:
    """Invariant: the steady state moves the ions, never creates them — the trapezoidal number
    in their layer is N·L at any bias."""
    import jax.numpy as jnp

    from jaxdd import solver as S

    dev = build_device(read_setup(DATA / "steady_1e17_thin_light" / "simulation_setup.txt"))
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in S._ion_setup(dev).items()}  # fmt: skip
    rng = np.random.default_rng(0)
    V = jnp.asarray(rng.normal(0.0, 0.3, len(dev.x)))
    logs = S._ion_logs(V, ion)
    g = S._ion_constraints(V, logs, ion)
    assert np.allclose(np.asarray(g), 0.0, atol=1e-12)


def test_batch_matches_the_one_device_solver() -> None:
    """The batched solver (`vmap` over devices, `while_loop` per voltage) gives each device's
    curve as the one-device solver does, with and without ions, in one batch."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    devices = [build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    J, ok, _ = solve_batch(prepare(devices), V, bernoulli_name="simss")
    assert np.asarray(ok).all()
    for k, dev in enumerate(devices):
        single = solve_jv(dev, V, bernoulli_name="simss")
        scale = np.max(np.abs(single.J))
        assert np.max(np.abs(np.asarray(J[k]) - single.J)) < 1e-6 * scale


def test_two_passes_give_the_one_pass_result() -> None:
    """`solve_batch` re-solves the devices that missed `first_max_iter` with `max_iter`: the
    result is the one pass's, flags and Newton steps exactly, curves to round-off (a smaller
    batch compiles to other roundings); the stragglers are padded to a power of two by
    repeating the last."""
    from jaxdd.batch import _solve_batch, _stragglers, prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    devices = [build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    batch = prepare(devices + devices[:2])
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    one = [np.asarray(a) for a in solve_batch(batch, V, first_max_iter=None)]
    first = np.asarray(_solve_batch(batch, V, 1e-10, 6, 60, "exact", False)[1])
    assert 0 < (~first.all(axis=1)).sum() < len(first)  # thin_l1_light needs 7 at a voltage
    two = [np.asarray(a) for a in solve_batch(batch, V, first_max_iter=6)]
    assert one[1].all()
    assert np.array_equal(one[1], two[1]) and np.array_equal(one[2], two[2])
    assert np.max(np.abs(one[0] - two[0])) < 1e-12 * np.max(np.abs(one[0]))

    ok = np.ones((6, 2), dtype=bool)
    ok[[1, 4, 5], 0] = False
    idx, padded = _stragglers(ok)
    assert idx.tolist() == [1, 4, 5] and padded.tolist() == [1, 4, 5, 5]
    assert _stragglers(np.ones((3, 2), dtype=bool))[1].size == 0


def test_the_batch_returns_the_voltages_in_the_order_given() -> None:
    """`solve_batch` scans outwards from the pre-bias whatever the order of `voltages` and
    returns each result at its voltage's place: reversed voltages, reversed results."""
    from jaxdd.batch import prepare, solve_batch

    case = DATA / "frozen_1e17_x1_thin_light"
    batch = prepare([build_device(read_setup(case / "simulation_setup.txt"))])
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    up = [np.asarray(a) for a in solve_batch(batch, V)]
    down = [np.asarray(a) for a in solve_batch(batch, V[::-1])]
    for a, b in zip(up, down, strict=True):
        assert np.array_equal(a, b[:, ::-1])


def test_the_batch_runs_inside_jit_and_jvp() -> None:
    """`solve_batch` inside `jax.jit` and `jax.jvp` (one pass there: the second pass's shape is
    known only once the first has run): the curves of a call outside, and forward-mode
    derivatives in a parameter (the generation's share) that finite differences confirm."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light"]
    batch = prepare([build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases])
    V = np.round(np.arange(1.3, -0.2 - 1e-9, -0.1), 6)

    def curves(g):
        return solve_batch(dict(batch, g_frac=batch["g_frac"] * g), V)[0]

    outside = np.asarray(solve_batch(batch, V)[0])
    inside = np.asarray(jax.jit(curves)(1.0))
    assert np.max(np.abs(inside - outside)) < 1e-9 * np.max(np.abs(outside))
    _, dJ = jax.jvp(curves, (1.0,), (1.0,))
    fd = (np.asarray(curves(1.0 + 1e-6)) - np.asarray(curves(1.0 - 1e-6))) / 2e-6
    assert np.max(np.abs(np.asarray(dJ) - fd)) < 1e-5 * np.max(np.abs(fd))


def test_a_dark_scan_with_a_density_far_below_the_floor_converges() -> None:
    """Regression: in the dark, a thick doped transport layer holds a minority density far
    below `n_floor`, whose logarithm is undetermined between neighbouring voltages. The batched
    scan's extrapolated start once carried that into NaN at every voltage from short circuit on.
    Every voltage converges, and the dark current is what a dark current must be: monotone in
    `V` and zero at 0 V, to `CURRENT_TOL` of the generation's current; and SimSS's curve to 2e-5
    of its scale (SimSS's own early stops at −0.2 and −0.175 V lie well within that)."""
    from jaxdd.batch import CURRENT_TOL, prepare, solve_batch

    case = DATA / "frozen_thick_htl_dark"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    batch = prepare([build_device(setup)])
    J, ok, _ = solve_batch(batch, ref["Vint"], bernoulli_name="simss")
    J, scale = np.asarray(J[0]), float(batch["J_scale"][0])
    assert np.asarray(ok).all()
    assert np.all(np.diff(J) > -CURRENT_TOL * scale)
    assert abs(J[np.argmin(np.abs(ref["Vint"]))]) < CURRENT_TOL * scale
    assert np.max(np.abs(J - ref["Jext"])) < 2e-5 * np.max(np.abs(ref["Jext"]))


@pytest.mark.parametrize("bern", ["exact", "simss"])
@pytest.mark.parametrize(
    "case",
    ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light", "bulk_acceptor_light",
     "fieldmob_light", "langevin_field_light", "braun_light", "zimt_photo_light",
     "zimt_acceptor_photo_light"],
)  # fmt: skip
def test_the_jacobian_by_edges_is_autodiff_of_the_rows(case: str, bern: str) -> None:
    """`_assemble` (each interval's fluxes once, assembled by hand) gives `_kernels`' residual,
    Jacobian blocks and ion columns (`jacfwd` of each row) to round-off, at a perturbed state
    whose neighbouring potentials differ from nothing to many V_T, free ions included; for a
    ZimT case with the time terms of a step of 1 ns from the unperturbed state."""
    from jaxdd import solver as S
    from jaxdd.solver import BERNOULLI, _assemble, _ion_setup, _kernels, _stencils

    dev = build_device(read_setup(DATA / case / "simulation_setup.txt"))
    s = {k: jnp.asarray(v) for k, v in _stencils(dev).items()}
    rng = np.random.default_rng(1)
    y0 = S._initial(dev, 0.3)
    NP = len(dev.x) - 2
    noise = np.zeros_like(y0)
    noise[1:-1] = rng.normal(0, 1, (NP, 3)) * np.array([5 * dev.Vt, 2.0, 2.0])
    noise[1:-1:7, 0] = 0.0  # some neighbours at exactly equal potential: B's series
    full = jnp.asarray(y0 + noise)
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in _ion_setup(dev).items()}  # fmt: skip
    la, lc = S._point_logs(S._ion_logs(full[:, 0], ion), ion)
    if case.startswith("zimt"):  # a step of 1 ns, the traps' occupations before it at random
        f_bulk = jnp.asarray(rng.uniform(0, 1, np.shape(s["Nt"])))
        s = S._time_terms(s, jnp.asarray(y0), 1e9, f_bulk)
    B = BERNOULLI[bern]
    row_val, row_jac, row_ions = _kernels(B)
    args = (full[:-2], full[1:-1], full[2:], s, dev.Vt, dev.G_frac, la, lc)
    ref = (row_val(*args), *row_jac(*args), jnp.stack(row_ions(*args), -1))
    got = _assemble(full, s, dev.Vt, dev.G_frac, la, lc, B)
    for name, r, g in zip(("F", "A", "B", "C", "U"), ref, got, strict=True):
        r, g = np.asarray(r), np.asarray(g)
        assert r.shape == g.shape, name
        scale = np.max(np.abs(r), axis=tuple(range(1, r.ndim)), keepdims=True)
        assert np.max(np.abs(r - g) / np.maximum(scale, 1e-300)) < 1e-11, name


@pytest.mark.parametrize("b", [3, 5])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("n", [5, 70])
def test_the_sweep_kernel_is_the_scan(n: int, transposed: bool, b: int) -> None:
    """`sweep.pallas_sweep` (the whole batch's sweep in one kernel) gives `_block_thomas`'s
    solution to round-off, 3×3 blocks with three right-hand sides and the 5×5 of mobile ions in
    time with four, a batch that is not a whole number of tiles, in either layout of the blocks;
    on a CPU in Pallas's interpreter."""
    from jaxdd import solver as S
    from jaxdd.sweep import pallas_sweep

    rng = np.random.default_rng(3)
    NP, k = 12, b - 1
    A = rng.normal(size=(n, NP, b, b))
    C = rng.normal(size=(n, NP, b, b))
    B = rng.normal(size=(n, NP, b, b)) + 8 * np.eye(b)
    d = rng.normal(size=(n, NP, b, k))
    ref = np.asarray(jax.vmap(S._block_thomas)(A, B, C, d))
    gpu = jax.default_backend() == "gpu"
    got = np.asarray(pallas_sweep(*(jnp.asarray(x) for x in (A, B, C, d)), tile=32,
                                  interpret=not gpu, transposed=transposed))  # fmt: skip
    assert np.max(np.abs(got - ref)) < 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("b", [3, 5])
@pytest.mark.parametrize("transposed", [False, True])
def test_the_mosaic_kernel_is_the_scan(transposed: bool, b: int) -> None:
    """`sweep.mosaic_sweep` (the kernel in Pallas's Mosaic GPU backend) gives `_block_thomas`'s
    solution to round-off, 3×3 and 5×5 blocks, a batch that is not a whole number of tiles.
    Only on an NVIDIA GPU (Mosaic GPU has no interpreter that runs this kernel on a CPU)."""
    from jaxdd import solver as S
    from jaxdd.sweep import kernel_available, mosaic_sweep

    if not kernel_available("mosaic"):
        pytest.skip("Mosaic GPU runs on an NVIDIA GPU, Ampere or newer")
    rng = np.random.default_rng(4)
    n, NP = 200, 12
    A = rng.normal(size=(n, NP, b, b))
    C = rng.normal(size=(n, NP, b, b))
    B = rng.normal(size=(n, NP, b, b)) + 8 * np.eye(b)
    d = rng.normal(size=(n, NP, b, b))
    ref = np.asarray(jax.vmap(S._block_thomas)(A, B, C, d))
    got = np.asarray(mosaic_sweep(*(jnp.asarray(x) for x in (A, B, C, d)), transposed=transposed))
    assert np.max(np.abs(got - ref)) < 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("kind", ["pallas", "mosaic"])
def test_the_batch_is_the_same_with_either_sweep(monkeypatch, kind: str) -> None:
    """`solve_batch` with a sweep kernel (Triton or Mosaic GPU) and with the scan
    (`JAXDD_SWEEP`): the same curves to round-off, the same flags and Newton steps. Only where
    the kernels run (an NVIDIA GPU)."""
    from jaxdd.batch import prepare, solve_batch
    from jaxdd.sweep import kernel_available

    if not kernel_available(kind):
        pytest.skip(f"the {kind} kernel does not run here")
    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    batch = prepare([build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases])
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    monkeypatch.setenv("JAXDD_SWEEP", kind)
    jax.clear_caches()
    kernel = [np.asarray(a) for a in solve_batch(batch, V)]
    monkeypatch.setenv("JAXDD_SWEEP", "scan")
    jax.clear_caches()  # the sweep is chosen when `solve_batch` is traced
    scan = [np.asarray(a) for a in solve_batch(batch, V)]
    jax.clear_caches()
    assert np.array_equal(kernel[1], scan[1]) and np.array_equal(kernel[2], scan[2])
    assert np.max(np.abs(kernel[0] - scan[0])) < 1e-9 * np.max(np.abs(scan[0]))


def test_the_five_by_five_inverse_is_lapacks() -> None:
    """`solver._inverse5` (by the carriers' 3×3 block and the ions' 2×2 Schur complement,
    elementwise) against LAPACK's inverse, with an ion's row held (an identity row) too."""
    from jaxdd.solver import _inverse5

    rng = np.random.default_rng(5)
    m = rng.normal(size=(500, 5, 5)) + 4 * np.eye(5)
    held = m.copy()
    held[:, 3, :] = np.eye(5)[3]
    for x in (m, held):
        assert np.max(np.abs(np.asarray(_inverse5(jnp.asarray(x))) - np.linalg.inv(x))) < 1e-12


@pytest.mark.parametrize("kind", ["pallas", "mosaic"])
def test_transients_with_ions_are_the_same_with_either_sweep(monkeypatch, kind: str) -> None:
    """Mobile ions in time (5×5 blocks) through the sweep kernel and through the scan
    (`JAXDD_SWEEP`): the same currents to round-off, the same flags and Newton steps. Only where
    the kernels run."""
    from jaxdd.batch import prepare
    from jaxdd.sweep import kernel_available
    from jaxdd.transient import solve_transient

    if not kernel_available(kind):
        pytest.skip(f"the {kind} kernel does not run here")
    setup, tvg, _ = _zimt_case("zimt_ions_fast_light")
    tvg = dataclasses.replace(tvg, **{k: getattr(tvg, k)[:40] for k in
                                      ("t", "Vext", "oc", "G_frac", "track", "dti")})  # fmt: skip
    batch = prepare([build_device(setup), build_device(_zimt_case("zimt_photo_light")[0])])
    runs = []
    for sweep in (kind, "scan"):
        monkeypatch.setenv("JAXDD_SWEEP", sweep)
        jax.clear_caches()  # the sweep is chosen when the solver is traced
        runs.append(solve_transient(batch, tvg, bernoulli_name="simss"))
    jax.clear_caches()
    kernel, scan = runs
    assert np.array_equal(kernel.converged, scan.converged)
    assert np.array_equal(kernel.iterations, scan.iterations)
    assert np.max(np.abs(kernel.J_ext - scan.J_ext)) < 1e-9 * np.max(np.abs(scan.J_ext))


def test_the_kernels_imports_do_not_compile_the_solver_again() -> None:
    """Importing Pallas registers states that enter every `jax.jit`'s cache key. Imported
    during the solver's first trace, it made that trace's key stale: the second call of
    `solve_batch` compiled again. In a fresh interpreter: after `jaxdd.batch`, what the kernel
    imports leaves a function jitted before with its one entry."""
    code = (
        "import jax, jaxdd.batch\n"
        "f = jax.jit(lambda x: x + 1.0)\n"
        "f(1.0)\n"
        "from jax.experimental import pallas\n"
        "try:\n"
        "    from jax.experimental.pallas import triton\n"
        "    from jax.experimental.pallas import mosaic_gpu\n"
        "except ImportError:\n"
        "    pass\n"
        "f(1.0)\n"
        "assert f._cache_size() == 1, f._cache_size()\n"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_block_sweep_solves_the_block_tridiagonal_system() -> None:
    """Invariant: the sweep gives the dense solve's solution to round-off, with several
    right-hand sides."""
    from jaxdd.solver import _block_thomas

    rng = np.random.default_rng(1)
    n = 37
    A, C = rng.normal(size=(n, 3, 3)), rng.normal(size=(n, 3, 3))
    B = rng.normal(size=(n, 3, 3)) + 8 * np.eye(3)
    A[0], C[-1] = 0.0, 0.0
    d = rng.normal(size=(n, 3, 2))
    full = np.zeros((3 * n, 3 * n))
    for i in range(n):
        full[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = B[i]
        if i > 0:
            full[3 * i : 3 * i + 3, 3 * i - 3 : 3 * i] = A[i]
        if i < n - 1:
            full[3 * i : 3 * i + 3, 3 * i + 3 : 3 * i + 6] = C[i]
    ref = np.linalg.solve(full, d.reshape(3 * n, 2))
    x = np.asarray(_block_thomas(*map(jnp.asarray, (A, B, C, d))))
    assert np.max(np.abs(x.reshape(3 * n, 2) - ref)) < 1e-12


def _scaled(batch, a):
    """The batch with five of its arrays moved by `a` (1 leaves it): a mobility, the generation,
    the anions' total, the doping and the pre-bias."""
    s = dict(batch["s"], mun_m=batch["s"]["mun_m"] * a[0], G=batch["s"]["G"] * a[1],
             dop=batch["s"]["dop"] + 1e21 * (a[3] - 1))  # fmt: skip
    ion = dict(batch["ion"], a_totals=batch["ion"]["a_totals"] * a[2])
    return dict(batch, s=s, ion=ion, G=batch["G"] * a[1], V_pre=batch["V_pre"] + 0.1 * (a[4] - 1))


def test_gradients_of_the_curves_are_finite_differences() -> None:
    """M7: reverse mode (`jax.grad`) through `solve_batch`, by implicit differentiation of each
    converged voltage, in five arrays of the batch and in the voltages; on devices without ions,
    with free ions and with ions frozen after a pre-bias (the gradient through the frozen
    profile). The gradient is forward mode's (`jax.jvp`, the same tangents, transposed) to
    round-off, and central finite differences confirm both."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    batch = prepare([build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases])
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=(len(cases), len(V))))

    def loss(a, Vs):
        J, ok, _ = solve_batch(_scaled(batch, a), Vs)
        return jnp.sum(w * jnp.where(ok, J, 0.0))

    a0, V0 = jnp.ones(5), jnp.asarray(V)
    ga, gV = jax.grad(loss, argnums=(0, 1))(a0, V0)
    rng = np.random.default_rng(1)
    da, dV = jnp.asarray(rng.normal(size=5)), jnp.asarray(rng.normal(size=len(V)))
    _, jv = jax.jvp(loss, (a0, V0), (da, dV))
    by_grad = float(ga @ da + gV @ dV)
    assert abs(by_grad - float(jv)) < 1e-9 * abs(float(jv))
    h = 1e-6
    fd = (float(loss(a0 + h * da, V0 + h * dV)) - float(loss(a0 - h * da, V0 - h * dV))) / (2 * h)
    assert abs(by_grad - fd) < 1e-6 * abs(fd)


def test_a_voltage_not_converged_has_zero_gradient() -> None:
    """A voltage that misses `max_iter` contributes nothing to a gradient, and nothing that is
    not finite: its tangents are zero (`thin_l1_light` needs 7 Newton steps at a voltage)."""
    from jaxdd.batch import _solve_batch, prepare

    batch = prepare([build_device(read_setup(DATA / "thin_l1_light" / "simulation_setup.txt"))])
    V = jnp.asarray(np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6))

    def curves(a):
        return _solve_batch(_scaled(batch, a), V, 1e-10, 6, 60, "exact", False)

    ok = np.asarray(curves(jnp.ones(5))[1])
    assert ok.any() and not ok.all()
    jac = np.asarray(jax.jacrev(lambda a: curves(a)[0])(jnp.ones(5)))
    assert np.isfinite(jac).all()
    assert np.all(jac[~ok] == 0.0) and np.any(jac[ok] != 0.0)


@pytest.mark.parametrize("K", [2, 4])
def test_the_bordered_system_and_its_transpose_are_solved(K: int) -> None:
    """Invariant: `_bordered_solve` solves the block-tridiagonal system bordered by the ions' K
    rows and columns (one per slot, `solver._slots`), and on `_transposed` blocks its
    transpose; `_apply` is its product. Against the dense matrix."""
    from jaxdd.batch import _apply, _bordered_solve, _transposed

    rng = np.random.default_rng(2)
    n = 11
    A, C = rng.normal(size=(n, 3, 3)), rng.normal(size=(n, 3, 3))
    B = rng.normal(size=(n, 3, 3)) + 8 * np.eye(3)
    U, W = rng.normal(size=(n, 3, K)), rng.normal(size=(n, 3, K))
    D = rng.normal(size=(K, K)) + 30 * np.eye(K)
    full = np.zeros((3 * n + K, 3 * n + K))
    for i in range(n):
        full[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = B[i]
        if i > 0:
            full[3 * i : 3 * i + 3, 3 * i - 3 : 3 * i] = A[i]
        if i < n - 1:
            full[3 * i : 3 * i + 3, 3 * i + 3 : 3 * i + 6] = C[i]
    full[: 3 * n, 3 * n :] = U.reshape(3 * n, K)
    full[3 * n :, : 3 * n] = W.reshape(3 * n, K).T
    full[3 * n :, 3 * n :] = D
    blocks = tuple(map(jnp.asarray, (A, B, C, U, W, D)))
    r, rg = rng.normal(size=(n, 3)), rng.normal(size=K)
    rhs = np.concatenate([r.ravel(), rg])
    for matrix, b in ((full, blocks), (full.T, _transposed(blocks))):
        x, z = _bordered_solve(b, jnp.asarray(r), jnp.asarray(rg))
        sol = np.concatenate([np.asarray(x).ravel(), np.asarray(z)])
        assert np.max(np.abs(sol - np.linalg.solve(matrix, rhs))) < 1e-12
        out, lo = _apply(b, jnp.asarray(r), jnp.asarray(rg))
        prod = np.concatenate([np.asarray(out).ravel(), np.asarray(lo)])
        assert np.max(np.abs(prod - matrix @ rhs)) < 1e-12


@pytest.mark.parametrize(
    "case",
    ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light",
     "frozen_thick_htl_dark", "iface_ions_light", "pin_iface_light", "resist_light",
     "calc_light", "none_light", "pin_anode_left_light", "bulk_acceptor_light",
     "trapfiles_light", "surface_light", "langevin_light", "fieldmob_light",
     "langevin_field_light", "braun_light", "diff_iface_light", "held_thin_light",
     "region_etl_light", "region_htl_frozen_light", "two_regions_light"],
)  # fmt: skip
def test_the_device_in_jax_is_numpys(case: str) -> None:
    """`batch.device_tree` (the device in `jax.numpy` from its structure and parameters) gives
    `prepare`'s tree of `build_device`'s device, every array to round-off, booleans exactly; and
    `vmap` over the parameters gives the stacked trees."""
    import functools

    from jaxdd.batch import device_tree, prepare

    setup = read_setup(DATA / case / "simulation_setup.txt")
    structure, params = device_parameters(setup)
    ref = prepare([build_device(setup)])
    got = device_tree(structure, params)
    for (path, r), g in zip(jax.tree_util.tree_flatten_with_path(ref)[0],
                            jax.tree_util.tree_leaves(got), strict=True):  # fmt: skip
        r, g = np.asarray(r)[0], np.asarray(g)
        assert r.dtype == g.dtype and r.shape == g.shape, jax.tree_util.keystr(path)
        assert np.allclose(g, r, rtol=1e-13, atol=0), jax.tree_util.keystr(path)

    twice = jax.tree_util.tree_map(lambda v: jnp.stack([jnp.asarray(v), 1.01 * jnp.asarray(v)]),
                                   params)  # fmt: skip
    batched = jax.vmap(functools.partial(device_tree, structure))(twice)
    second = device_tree(structure, jax.tree_util.tree_map(lambda v: v[1], twice))
    for b, s in zip(jax.tree_util.tree_leaves(batched), jax.tree_util.tree_leaves(second),
                    strict=True):  # fmt: skip
        assert np.allclose(np.asarray(b)[1], np.asarray(s), rtol=1e-13, atol=0)


def test_gradients_in_physical_parameters_are_finite_differences() -> None:
    """M7: the gradient of the curves in a layer's parameters, through `device_tree` and
    `solve_batch`: the absorber's electron mobility and thickness (the grid moves, its points per
    layer stay), a transport layer's band edge, the ion density, the temperature and a work
    function; against central finite differences along a random direction."""
    from jaxdd.batch import device_tree, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    parts = [device_parameters(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=(len(cases), len(V))))
    absorber = jnp.array([0.0, 1.0, 0.0])

    def loss(a):
        trees = []
        for structure, p in parts:
            q = dict(p, mu_n=p["mu_n"] * (1 + absorber * (a[0] - 1)),
                     L=p["L"] * (1 + absorber * (a[1] - 1)),
                     E_c=p["E_c"] + jnp.array([0.05, 0.0, 0.0]) * (a[2] - 1),
                     N_anion=p["N_anion"] * a[3], T=p["T"] * a[4],
                     W_L=p["W_L"] + 0.05 * (a[5] - 1))  # fmt: skip
            trees.append(device_tree(structure, q))
        batch = jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *trees)
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok, J, 0.0))

    a0 = jnp.ones(6)
    direction = jnp.asarray(np.random.default_rng(1).normal(size=6))
    by_grad = float(jax.grad(loss)(a0) @ direction)
    h = 1e-6
    fd = (float(loss(a0 + h * direction)) - float(loss(a0 - h * direction))) / (2 * h)
    assert abs(by_grad - fd) < 1e-6 * abs(fd)


def test_a_fit_recovers_the_parameters_of_a_simss_curve() -> None:
    """M7: from a start 1.4–1.7× off, `validation/fit.py` (least squares, its Jacobian by
    `jax.jacfwd` through the converged solution) recovers the absorber's trap density and
    mobilities and the left work function from SimSS's curve; the direct recombination, which
    this curve barely determines (its singular value is 2.5e-6 of the largest), to 10 %."""
    pytest.importorskip("scipy")
    sys.path.insert(0, str(Path(__file__).parents[1] / "validation"))
    from fit import fit

    f = fit(DATA / "thin_l1_light", start=0.5)
    err = dict(zip(f.names, np.abs(f.errors()), strict=True))
    assert f.out.success and not f.out.active_mask.any()
    assert err["N_t_bulk[1]"] < 1e-3 and err["mu_n[1]"] < 1e-3 and err["mu_p[1]"] < 1e-3
    assert err["W_L"] < 1e-4  # eV
    assert err["k_direct[1]"] < 0.1


def test_the_gradient_in_the_interface_traps_is_finite_differences() -> None:
    """The curve's derivative in both interfaces' trap densities, through `device_tree` and the
    converged solution (forward mode), against central finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "pin_iface_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 0.95 + 1e-9, 0.05), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))
    direction = jnp.array([1.0, -0.6, 0.0])

    def loss(a):
        p = dict(params, N_t_int=params["N_t_int"] * jnp.exp(a * direction))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-5
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_the_gradient_of_the_differential_current_is_finite_differences() -> None:
    """With `currDiffInt = 1`, the curve's derivative in the interface traps and the absorber's
    mobilities, through `device_tree` and the converged solution, against central finite
    differences."""
    from jaxdd.batch import device_tree, solve_batch

    setup = read_setup(DATA / "diff_iface_light" / "simulation_setup.txt")
    structure, params = device_parameters(setup)
    assert structure.curr_diff
    V = np.round(np.arange(-0.1, 1.2 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        p = dict(params, N_t_int=params["N_t_int"] * jnp.exp(a * jnp.array([1.0, -0.6, 0.0])))
        for k in ("mu_n", "mu_p"):
            p[k] = jnp.asarray(params[k]).at[1].multiply(jnp.exp(0.5 * a))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-5
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_the_gradient_in_trap_file_levels_is_finite_differences() -> None:
    """The curve's derivative in the charged levels of a bulk and an interface trap file
    (their densities and one energy), through `device_tree` and the converged solution, against
    central finite differences."""
    from jaxdd.batch import device_tree, solve_batch

    structure, params = device_parameters(read_setup(DATA / "trapfiles_light" /
                                                     "simulation_setup.txt"))  # fmt: skip
    V = np.round(np.arange(-0.1, 1.2 + 1e-9, 0.1), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))
    bulk = jnp.array([1.0, -0.5, 0.7])
    iface = jnp.array([0.8, -1.0])

    def loss(a):
        p = dict(params)
        p["bulk_trap_N"] = jnp.asarray(params["bulk_trap_N"]).at[1].multiply(jnp.exp(a * bulk))
        p["bulk_trap_E"] = jnp.asarray(params["bulk_trap_E"]).at[1, 2].add(0.05 * a)
        p["int_trap_N"] = jnp.asarray(params["int_trap_N"]).at[0].multiply(jnp.exp(a * iface))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, p))
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    _, by_jvp = jax.jvp(loss, (0.0,), (1.0,))
    h = 1e-5
    fd = (float(loss(h)) - float(loss(-h))) / (2 * h)
    assert abs(float(by_jvp) - fd) < 1e-6 * abs(fd)


def test_a_trap_level_outside_the_gap_is_refused(tmp_path) -> None:
    """As SIMsalabim's `Init_Trapping`: every level of a trap file within its layer's gap."""

    for f in (DATA / "trapfiles_light").iterdir():
        (tmp_path / f.name).write_text(f.read_text())
    (tmp_path / "bulk_traps.txt").write_text("E Ntrap\n4.5 1e21\n5.7 1e21\n")
    with pytest.raises(ValueError, match="outside the gap"):
        device_parameters(read_setup(tmp_path / "simulation_setup.txt"))


def test_a_batch_split_across_devices_is_the_batch_on_one() -> None:
    """`solve_batch` across two devices (two XLA host devices here, the GPUs of a process on a
    GPU host): a batch of five, padded to six, gives the curves, flags and Newton steps of the
    batch on one device; the padding is dropped; with a second pass for the stragglers too,
    its results back on one device (no operation across the devices)."""
    code = """
import sys, jax, numpy as np
sys.path.insert(0, "validation")
from bench import V, variants
from jaxdd.batch import prepare, solve_batch
assert len(jax.devices()) == 2
batch = prepare(variants(5))
J1, ok1, it1 = solve_batch(batch, V, devices=jax.devices()[:1])
J2, ok2, it2 = solve_batch(batch, V)
assert J2.shape == J1.shape == (5, len(V))
assert np.array_equal(np.asarray(ok1), np.asarray(ok2)) and np.asarray(ok1).all()
assert np.array_equal(np.asarray(it1), np.asarray(it2))
scale = np.asarray(batch["J_scale"])[:, None]
assert np.max(np.abs(np.asarray(J1) - np.asarray(J2)) / scale) < 1e-10
# The second pass (a first pass of 2 Newton steps leaves stragglers): its results written back
# into the first's, which lie on one device.
J3, ok3, it3 = solve_batch(batch, V, first_max_iter=2, devices=jax.devices()[:1])
J4, ok4, it4 = solve_batch(batch, V, first_max_iter=2)
assert len(J4.sharding.device_set) == 1
assert np.array_equal(np.asarray(ok3), np.asarray(ok4)) and np.asarray(ok3).all()
assert np.max(np.abs(np.asarray(J3) - np.asarray(J4)) / scale) < 1e-10
print("ok")
"""
    env = {**os.environ, "JAX_PLATFORMS": "cpu",
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}  # fmt: skip
    root = Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                          cwd=root)  # fmt: skip
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]


@pytest.mark.parametrize("dti", [0.0, 1e2])
def test_the_ions_blocks_in_time_are_autodiff(dti: float) -> None:
    """`_assemble_ions`' blocks and slot columns (the ions' rows by hand, the carriers' from
    `_assemble`) are `jax.jacfwd` of its residual in the whole state (block tridiagonal) and in
    the slots, at a perturbed state of a region of two layers: over a step and in the steady
    state."""
    from jaxdd import solver as S

    dev = build_device(read_setup(DATA / "region_etl_light" / "simulation_setup.txt"))
    s = jax.tree_util.tree_map(jnp.asarray, S._stencils(dev))
    ion = jax.tree_util.tree_map(jnp.asarray, S._ion_setup(dev))
    rng = np.random.default_rng(4)
    y0 = S._initial(dev, 0.4)
    NP = len(dev.x) - 2
    ma, mc = (np.asarray(m) for m in S._ion_membership(ion))
    ions = np.stack([np.where(ma > 0, 48.0, 0.0), np.where(mc > 0, 48.0, 0.0)], -1)
    full = np.concatenate([y0, ions], -1)
    full[1:-1] += rng.normal(0, 1, (NP, 5)) * np.array([dev.Vt, 0.5, 0.5, 0.3, 0.3])
    full = jnp.asarray(full)
    s = S._time_terms(s, full, dti)
    logs = jnp.asarray(rng.normal(0, 0.1, ion["a_slots"].shape[0] + ion["c_slots"].shape[0]))
    _, A, B, C, U = S._assemble_ions(full, s, dev.Vt, dev.G_frac, logs, ion, S.bernoulli)

    def residual(y, slot_logs):
        return S._assemble_ions(full.at[1:-1].set(y), s, dev.Vt, dev.G_frac, slot_logs, ion,
                                S.bernoulli)[0]  # fmt: skip

    J = np.asarray(jax.jacfwd(residual)(full[1:-1], logs))  # NP × 5 × NP × 5
    i = np.arange(NP)
    for name, got, ref in (("B", B, J[i, :, i, :]), ("A", A[1:], J[i[1:], :, i[:-1], :]),
                           ("C", C[:-1], J[i[:-1], :, i[1:], :])):  # fmt: skip
        ref, got = np.asarray(ref), np.asarray(got)
        scale = np.max(np.abs(ref), axis=(1, 2), keepdims=True)
        assert np.max(np.abs(got - ref) / np.maximum(scale, 1e-300)) < 1e-10, name
    Ul = np.asarray(jax.jacfwd(residual, argnums=1)(full[1:-1], logs))
    assert np.max(np.abs(np.asarray(U) - Ul)) < 1e-10 * max(np.max(np.abs(Ul)), 1e-300)
