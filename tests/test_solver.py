"""jaxdd against SimSS, and the properties of its pieces."""

import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.device import build_device, make_grid
from jaxdd.simss_input import check_supported, read_setup
from jaxdd.solver import bernoulli, bernoulli_simss, solve_jv

DATA = Path(__file__).parent / "data"


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
    assert "Vdist = 1 " in text
    setup_file.write_text(text.replace("Vdist = 1 ", "Vdist = 2 ", 1))
    with pytest.raises(NotImplementedError, match="Vdist"):
        check_supported(read_setup(setup_file))


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
    la, lc = S._ion_logs(full[:, 0], ion)

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
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

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


def test_a_batch_mixes_devices_with_and_without_interface_traps() -> None:
    """`prepare` gives the devices without interface traps zero traps on every edge, those
    without charged bulk traps none, those with a constant mobility no field dependence, and
    every device the most trap levels of any (the others
    of density 0): each curve of a mixed batch is the one-device solver's."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", "iface_charged_light", "iface_ions_light", "bulk_acceptor_light",
             "trapfiles_light", "surface_light", "fieldmob_light",
             "langevin_field_light"]  # fmt: skip
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
    la, lc = S._ion_logs(full[:, 0], ion)

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
    g = S._ion_constraints(V, logs[0], logs[1], ion)
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
     "fieldmob_light", "langevin_field_light"],
)  # fmt: skip
def test_the_jacobian_by_edges_is_autodiff_of_the_rows(case: str, bern: str) -> None:
    """`_assemble` (each interval's fluxes once, assembled by hand) gives `_kernels`' residual,
    Jacobian blocks and ion columns (`jacfwd` of each row) to round-off, at a perturbed state
    whose neighbouring potentials differ from nothing to many V_T, free ions included."""
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
    la, lc = S._ion_logs(full[:, 0], {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
                                      for k, v in _ion_setup(dev).items()})  # fmt: skip
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


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("n", [5, 70])
def test_the_sweep_kernel_is_the_scan(n: int, transposed: bool) -> None:
    """`sweep.pallas_sweep` (the whole batch's sweep in one kernel) gives `_block_thomas`'s
    solution to round-off, three right-hand sides, a batch that is not a whole number of tiles,
    in either layout of the blocks; on a CPU in Pallas's interpreter."""
    from jaxdd import solver as S
    from jaxdd.sweep import pallas_sweep

    rng = np.random.default_rng(3)
    NP = 12
    A = rng.normal(size=(n, NP, 3, 3))
    C = rng.normal(size=(n, NP, 3, 3))
    B = rng.normal(size=(n, NP, 3, 3)) + 8 * np.eye(3)
    d = rng.normal(size=(n, NP, 3, 3))
    ref = np.asarray(jax.vmap(S._block_thomas)(A, B, C, d))
    gpu = jax.default_backend() == "gpu"
    got = np.asarray(pallas_sweep(*(jnp.asarray(x) for x in (A, B, C, d)), tile=32,
                                  interpret=not gpu, transposed=transposed))  # fmt: skip
    assert np.max(np.abs(got - ref)) < 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("transposed", [False, True])
def test_the_mosaic_kernel_is_the_scan(transposed: bool) -> None:
    """`sweep.mosaic_sweep` (the kernel in Pallas's Mosaic GPU backend) gives `_block_thomas`'s
    solution to round-off, a batch that is not a whole number of tiles. Only on an NVIDIA GPU
    (Mosaic GPU has no interpreter that runs this kernel on a CPU)."""
    from jaxdd import solver as S
    from jaxdd.sweep import kernel_available, mosaic_sweep

    if not kernel_available("mosaic"):
        pytest.skip("Mosaic GPU runs on an NVIDIA GPU, Ampere or newer")
    rng = np.random.default_rng(4)
    n, NP = 200, 12
    A = rng.normal(size=(n, NP, 3, 3))
    C = rng.normal(size=(n, NP, 3, 3))
    B = rng.normal(size=(n, NP, 3, 3)) + 8 * np.eye(3)
    d = rng.normal(size=(n, NP, 3, 3))
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
    ion = dict(batch["ion"], a_total=batch["ion"]["a_total"] * a[2])
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


def test_the_bordered_system_and_its_transpose_are_solved() -> None:
    """Invariant: `_bordered_solve` solves the block-tridiagonal system bordered by the ions' two
    rows and columns, and on `_transposed` blocks its transpose; `_apply` is its product. Against
    the dense matrix."""
    from jaxdd.batch import _apply, _bordered_solve, _transposed

    rng = np.random.default_rng(2)
    n = 11
    A, C = rng.normal(size=(n, 3, 3)), rng.normal(size=(n, 3, 3))
    B = rng.normal(size=(n, 3, 3)) + 8 * np.eye(3)
    U, W = rng.normal(size=(n, 3, 2)), rng.normal(size=(n, 3, 2))
    D = rng.normal(size=(2, 2)) + 30 * np.eye(2)
    full = np.zeros((3 * n + 2, 3 * n + 2))
    for i in range(n):
        full[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = B[i]
        if i > 0:
            full[3 * i : 3 * i + 3, 3 * i - 3 : 3 * i] = A[i]
        if i < n - 1:
            full[3 * i : 3 * i + 3, 3 * i + 3 : 3 * i + 6] = C[i]
    full[: 3 * n, 3 * n :] = U.reshape(3 * n, 2)
    full[3 * n :, : 3 * n] = W.reshape(3 * n, 2).T
    full[3 * n :, 3 * n :] = D
    blocks = tuple(map(jnp.asarray, (A, B, C, U, W, D)))
    r, rg = rng.normal(size=(n, 3)), rng.normal(size=2)
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
     "langevin_field_light"],
)  # fmt: skip
def test_the_device_in_jax_is_numpys(case: str) -> None:
    """`batch.device_tree` (the device in `jax.numpy` from its structure and parameters) gives
    `prepare`'s tree of `build_device`'s device, every array to round-off, booleans exactly; and
    `vmap` over the parameters gives the stacked trees."""
    import functools

    from jaxdd.batch import device_tree, prepare
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

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


def test_the_gradient_in_trap_file_levels_is_finite_differences() -> None:
    """The curve's derivative in the charged levels of a bulk and an interface trap file
    (their densities and one energy), through `device_tree` and the converged solution, against
    central finite differences."""
    from jaxdd.batch import device_tree, solve_batch
    from jaxdd.device import device_parameters

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
    from jaxdd.device import device_parameters

    for f in (DATA / "trapfiles_light").iterdir():
        (tmp_path / f.name).write_text(f.read_text())
    (tmp_path / "bulk_traps.txt").write_text("E Ntrap\n4.5 1e21\n5.7 1e21\n")
    with pytest.raises(ValueError, match="outside the gap"):
        device_parameters(read_setup(tmp_path / "simulation_setup.txt"))
