"""The temperature laws (`jaxdd.temperature`): at the reference temperature every device to the
bit, each law its closed form, the gradients in their coefficients away from it, the gap check."""

import dataclasses
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.batch import _device_tree, device_tree, solve_batch
from jaxdd.constants import K_B, Q
from jaxdd.device import device_arrays
from jaxdd.device import device_parameters as _device_parameters
from jaxdd.simss_input import read_setup, read_tvg
from jaxdd.temperature import Gap, Law, at_temperature, check_gaps, textbook_laws

DATA = Path(__file__).parent / "data"

#: The parameters the item names for a law (`docs/model.md`, "Temperature laws").
SCALED = ("N_c", "mu_n", "mu_p", "gamma_n", "gamma_p", "k_direct", "C_n_bulk", "C_p_bulk",
          "C_n_int", "C_p_int", "C_n_Auger", "C_p_Auger", "mu_anion", "mu_cation", "nu_int_n",
          "nu_int_p")  # fmt: skip
SURFACE = ("S_n_L", "S_p_L", "S_n_R", "S_p_R")


def parameters(case: str):
    """`device_parameters` on SimSS's grid, as the solver's tests take it."""
    return _device_parameters(read_setup(DATA / case / "simulation_setup.txt"), grid="simss")


def every_law(n: int, seed: int = 0) -> dict:
    """A law on every parameter the item names, each layer its own power and activation energy;
    the surfaces' each one; and one law the caller's own."""
    rng = np.random.default_rng(seed)
    laws = {k: Law(m=rng.uniform(-2, 2, n), E=rng.uniform(0, 0.5, n)) for k in SCALED}
    laws.update({k: Law(m=rng.uniform(-1, 1), E=rng.uniform(0, 0.3)) for k in SURFACE})

    def g(T):  # any function of T: its ratio is 1 at T0 to the bit
        return 1.0 + 2e-4 * (T - 250.0) ** 2

    laws["N_t_bulk"] = lambda v, T, T0: v * (g(T) / g(T0))
    return laws


def every_gap(n: int) -> Gap:
    return Gap(slope=np.linspace(-4e-4, 5e-4, n), share=np.linspace(0.2, 0.9, n))


def same(a, b) -> list[str]:
    """The paths of the leaves where two trees differ, to the bit."""
    return [jax.tree_util.keystr(k)
            for (k, x), y in zip(jax.tree_util.tree_flatten_with_path(a)[0],
                                 jax.tree_util.tree_leaves(b), strict=True)
            if not (np.shape(x) == np.shape(y) and np.array_equal(x, y))]  # fmt: skip


CASES = ["frozen_1e17_x1_thin_light", "steady_1e17_thin_light", "pin_iface_light",
         "trapfiles_light", "surface_light", "langevin_light", "two_regions_light"]  # fmt: skip


@pytest.mark.parametrize("case", CASES)
def test_at_the_reference_temperature_every_law_gives_the_device_to_the_bit(case: str) -> None:
    """Every law on (each parameter the item names, the gap's, every level and work function
    following the bands, the edge), mapped to the dict's own `T`: the dict, the device in NumPy
    and in JAX (eager, and compiled with the dict as its argument) are the unmapped ones', every
    leaf `array_equal`. The edge's `dE_nk` too, where the dict has it."""
    structure, p = parameters(case)
    p = dict(p, dE_nk=np.zeros(len(p["L"])))
    n = len(p["L"])
    kw = dict(laws=every_law(n), gap=every_gap(n), traps=["midgap", "E_c", "E_v"][:n],
              contacts=("E_c", "E_v"), edge=True)  # fmt: skip
    q = at_temperature(p, p["T"], **kw)
    assert set(q) == set(p) and not same(p, q)
    q = at_temperature(p, p["T"], **(kw | dict(traps="E_v", structure=structure)))
    assert not same(p, q)

    numpy = [_device_tree(device_arrays(structure, d, np), np) for d in (p, q)]
    assert not same(*numpy)
    assert not same(device_tree(structure, p), device_tree(structure, q))

    pj = jax.tree_util.tree_map(jnp.asarray, p)
    ref = jax.jit(functools.partial(device_tree, structure))(pj)
    mapped = jax.jit(lambda d, T: device_tree(structure, at_temperature(d, T, **kw)))(pj, pj["T"])
    assert not same(ref, mapped)


def test_each_law_is_its_closed_form() -> None:
    """At 340 K from 298.15 K: `Law`'s factor, the textbook laws (`N_c ∝ T^{3/2}`, `ν_int ∝
    T^{1/2}`, the ions' Arrhenius `D = μ·V_T`, a constant `D` by default), a law reaching the
    levels' own capture coefficients (a caller's per layer, levels as many as layers, each
    layer's own), the infinite surfaces kept; the gap's change split as asked, linear or the
    caller's (Varshni's), and every level and work function following what it is told."""
    structure, p = parameters("pin_iface_light")
    p = dict(p, bulk_trap_Cn=np.array([[1e-14, 2e-14]] * 3), dE_nk=np.array([0.0, 0.01, 0.0]))
    T0, T = p["T"], 340.0
    beta = 1.0 / (K_B * T / Q) - 1.0 / (K_B * T0 / Q)  # 1/kT − 1/kT0, per eV

    m, E = np.array([-1.5, 0.7, 2.0]), np.array([0.0, 0.12, 0.3])
    laws = textbook_laws(E_anion=0.4, E_cation=np.array([0.2, 0.6, 0.0])) | {
        "mu_n": Law(m=m, E=E), "C_n_bulk": Law(m=0.5, E=E),
        "S_n_L": Law(m=1.0, E=0.1), "k_direct": Law(m=-1.0)}  # fmt: skip
    q = at_temperature(p, T, laws)
    r = T / T0
    np.testing.assert_allclose(q["mu_n"], p["mu_n"] * r**m * np.exp(-E * beta), rtol=1e-14)
    np.testing.assert_allclose(q["N_c"], p["N_c"] * r**1.5, rtol=1e-14)
    np.testing.assert_allclose(q["nu_int_p"], p["nu_int_p"] * np.sqrt(r), rtol=1e-14)
    np.testing.assert_allclose(q["C_n_bulk"], p["C_n_bulk"] * np.sqrt(r) * np.exp(-E * beta),
                               rtol=1e-14)  # fmt: skip
    np.testing.assert_allclose(q["bulk_trap_Cn"], p["bulk_trap_Cn"] * (np.sqrt(r) *
                               np.exp(-E * beta))[:, None], rtol=1e-14)  # fmt: skip
    for species, Ea in (("anion", 0.4), ("cation", np.array([0.2, 0.6, 0.0]))):
        D, D0 = q[f"mu_{species}"] * K_B * T / Q, p[f"mu_{species}"] * K_B * T0 / Q
        np.testing.assert_allclose(D, D0 * np.exp(-Ea * beta), rtol=1e-14)
    ions = dict(p, mu_anion=np.array([1e-12, 2e-12, 3e-12]), mu_cation=np.full(3, 1e-13))
    q0 = at_temperature(ions, T, textbook_laws())
    for species in ("anion", "cation"):
        mu = ions[f"mu_{species}"]
        np.testing.assert_allclose(q0[f"mu_{species}"] * T, mu * T0, rtol=1e-14)

    powers = np.array([1.0, 2.0, 3.0])

    def g(T):  # a caller's factor, one per layer
        return (T / T0) ** powers

    square = dict(p, bulk_trap_Cn=np.array([[1e-14, 2e-14, 3e-14]] * 3))
    c = at_temperature(square, T, {"C_n_bulk": lambda v, T, T0: v * (g(T) / g(T0))})
    np.testing.assert_allclose(c["C_n_bulk"], p["C_n_bulk"] * r**powers, rtol=1e-14)
    np.testing.assert_allclose(c["bulk_trap_Cn"], square["bulk_trap_Cn"] * (r**powers)[:, None],
                               rtol=1e-14)  # fmt: skip
    assert q["S_n_L"] == -1.0  # infinite: stays
    s = at_temperature(dict(p, S_n_L=20.0), T, laws)["S_n_L"]
    assert s == pytest.approx(20.0 * r * np.exp(-0.1 * beta), rel=1e-14)
    assert q["T"] == T and np.array_equal(q["mu_p"], p["mu_p"]) and q["E_c"] is p["E_c"]
    langevin = dataclasses.replace(structure, langevin=np.array([True, False, True]))
    kd = at_temperature(p, T, laws, structure=langevin)["k_direct"]
    assert np.array_equal(kd[[0, 2]], p["k_direct"][[0, 2]]) and kd[1] == pytest.approx(
        p["k_direct"][1] / r, rel=1e-14)

    slope, share = np.array([2e-4, -3e-4, 1e-4]), np.array([0.0, 0.3, 1.0])
    dEg = slope * (T - T0)
    for traps, follow in (("fixed", 0.0), ("E_c", -(1 - share) * dEg), ("E_v", share * dEg),
                          ("midgap", (share - 0.5) * dEg)):  # fmt: skip
        q = at_temperature(p, T, gap=Gap(slope, share), traps=traps, contacts=("E_v", "midgap"),
                           edge=True)  # fmt: skip
        np.testing.assert_allclose(q["E_v"] - q["E_c"], p["E_v"] - p["E_c"] + dEg, rtol=1e-14)
        np.testing.assert_allclose(q["E_v"], p["E_v"] + share * dEg, rtol=1e-14)
        np.testing.assert_allclose(q["E_t_bulk"], p["E_t_bulk"] + follow, rtol=1e-14)
        np.testing.assert_allclose(q["E_t_int"], p["E_t_int"] + follow, rtol=1e-14)
        assert q["W_L"] == pytest.approx(p["W_L"] + share[0] * dEg[0], rel=1e-14)
        assert q["W_R"] == pytest.approx(p["W_R"] + (share[2] - 0.5) * dEg[2], rel=1e-14)
        np.testing.assert_allclose(q["dE_nk"], p["dE_nk"] + dEg, rtol=1e-14, atol=1e-16)
    assert "dE_nk" not in at_temperature(dict(parameters("thin_l1_light")[1]), T,
                                         gap=Gap(slope), edge=True)  # fmt: skip

    def varshni(T):  # Varshni's gap, eV, each layer its own
        return 1.6 - np.array([5e-4, 3e-4, 4e-4]) * T**2 / (T + np.array([200.0, 100.0, 300.0]))

    q = at_temperature(p, T, gap=Gap(share=0.5, Eg=varshni))
    np.testing.assert_allclose(q["E_v"] - q["E_c"] - (p["E_v"] - p["E_c"]),
                               varshni(T) - varshni(T0), rtol=1e-12)  # fmt: skip


def test_the_mapping_refuses_what_it_cannot_do() -> None:
    """A law on an energy or `T`, on a name the dict lacks, one per layer on a single value, or
    an option not in `FOLLOW`."""
    _, p = parameters("thin_l1_light")
    for name in ("E_c", "W_L", "E_t_bulk", "T", "mu_N"):
        with pytest.raises(ValueError, match=name):
            at_temperature(p, 320.0, {name: Law(m=1.0)})
    with pytest.raises(ValueError, match="S_n_L"):
        at_temperature(p, 320.0, {"S_n_L": Law(E=np.full(len(p["L"]), 0.1))})
    with pytest.raises(ValueError, match="traps"):
        at_temperature(p, 320.0, traps="E_g")
    with pytest.raises(ValueError, match="traps"):
        at_temperature(p, 320.0, traps=["E_c", "E_v"])
    with pytest.raises(ValueError, match="contacts"):
        at_temperature(p, 320.0, contacts=("E_c", "fixed", "E_v"))
    assert at_temperature(p, 320.0, {"bulk_trap_Cn": Law(m=1.0)})["T"] == 320.0  # none: nothing


def test_a_law_on_what_the_input_lacks_changes_nothing() -> None:
    """At 330 K, laws on a device's absent mechanisms (infinite surfaces, no Auger, no interface
    traps, no ions, unbounded sites), and on `k_direct` where Langevin's rate replaces it: the
    device is the one without them, to the bit, and an infinite `S` stays negative."""
    for case, laws in (
        ("thin_l1_light", {k: Law(m=1.3, E=0.2) for k in (
            *SURFACE, "C_n_Auger", "C_p_Auger", "N_t_int", "C_n_int", "C_p_int", "mu_anion",
            "mu_cation", "N_anion", "N_cation", "N_max_anion", "N_max_cation")}),
        ("langevin_light", {"k_direct": Law(m=-2.0, E=0.3)}),
    ):  # fmt: skip
        structure, p = parameters(case)
        base = at_temperature(p, 330.0, textbook_laws())
        for s in (None, structure):
            q = at_temperature(p, 330.0, textbook_laws() | laws, structure=s)
            assert not same(device_tree(structure, base), device_tree(structure, q)), case
            assert all(q[k] == -1.0 for k in SURFACE)
        assert np.array_equal(q["k_direct"], p["k_direct"])  # with the structure: kept


def test_the_gap_check_refuses_a_level_or_work_function_moved_out() -> None:
    """`check_gaps` passes every input at its own temperature and refuses what a gap moved out:
    a bulk level and a work function the valence band passed, an interface level the conduction
    band of its left layer passed; a level or work function that follows the band stays in. On
    a stack of temperatures too (`vmap`), one outside enough."""
    for case in [*CASES, "iface_charged_light", "held_thin_light", "pin_anode_left_light",
                 "chargetransport_ions"]:  # fmt: skip
        structure, p = parameters(case)
        check_gaps(structure, p)
        check_gaps(structure, at_temperature(p, 340.0, gap=every_gap(len(p["L"]))))

    # 'sfb' at both ends: no work function to check, and none moved where the structure is given
    structure, p = parameters("chargetransport_ions")
    far = Gap(slope=1e-2)
    check_gaps(structure, at_temperature(p, 400.0, gap=far, contacts="E_c"))
    q = at_temperature(p, 400.0, gap=far, contacts="E_c", structure=structure)
    assert q["W_L"] == p["W_L"] == 0.0 and q["W_R"] == p["W_R"] == 0.0

    # The absorber's levels 4.5–5.1 eV in 4.0–5.6 eV, W_R 5.325 eV in the last layer's 3.0–5.6;
    # at 100 K more their E_v at 5.05 and 5.3 eV.
    structure, p = parameters("trapfiles_light")
    down = Gap(slope=np.array([0.0, -5.5e-3, -3e-3]), share=1.0)
    T = p["T"] + 100.0
    with pytest.raises(ValueError, match=r"W_R .* outside the gap") as e:
        check_gaps(structure, at_temperature(p, T, gap=down))
    assert "bulk trap level outside the gap ([5.1] eV)" in str(e.value)
    assert "interface" not in str(e.value)
    with pytest.raises(ValueError, match="bulk trap") as e:
        check_gaps(structure, at_temperature(p, T, gap=down, contacts="E_v"))
    assert "W_R" not in str(e.value)
    with pytest.raises(ValueError, match=r"bulk trap level outside the gap \(\[3.95\] eV\)"):
        check_gaps(structure, at_temperature(p, T, gap=down, traps="E_v", contacts="E_v"))
    check_gaps(structure, at_temperature(p, T, gap=down, traps="midgap", contacts="E_v"))

    structure, p = parameters("pin_iface_light")  # E_t_int 4.7 eV; layer 1's E_c 4.0 eV
    up = Gap(slope=np.array([-5e-3, 0.0, 0.0]), share=0.0)  # its E_c to 4.75 eV at 448.15 K
    with pytest.raises(ValueError, match="interface trap between layers 1 and 2") as e:
        check_gaps(structure, at_temperature(p, 448.15, gap=up, contacts="E_c"))
    assert "bulk" not in str(e.value) and "W_L" not in str(e.value)
    check_gaps(structure, at_temperature(p, 448.15, gap=up, contacts="E_c", traps="E_c"))

    pj = jax.tree_util.tree_map(jnp.asarray, p)
    stack = jax.vmap(lambda T: at_temperature(pj, T, gap=up, contacts="E_c"))
    check_gaps(structure, stack(jnp.array([280.0, 298.15, 320.0])))
    with pytest.raises(ValueError, match="interface trap"):
        check_gaps(structure, stack(jnp.array([280.0, 298.15, 448.15])))


def test_vmap_over_temperatures_is_each_temperature() -> None:
    """`device_tree` of the mapped dict by `vmap` over the temperature: each device the one
    mapped alone, to round-off."""
    structure, p = parameters("pin_iface_light")
    p = jax.tree_util.tree_map(jnp.asarray, p)
    n = len(p["L"])

    def tree(T):
        return device_tree(structure, at_temperature(p, T, every_law(n), every_gap(n),
                                                     traps="midgap", contacts="E_c"))  # fmt: skip

    T = jnp.array([270.0, 298.15, 330.0])
    batched = jax.vmap(tree)(T)
    for k in range(len(T)):
        for b, s in zip(jax.tree_util.tree_leaves(batched), jax.tree_util.tree_leaves(tree(T[k])),
                        strict=True):  # fmt: skip
            assert np.allclose(np.asarray(b)[k], np.asarray(s), rtol=1e-13, atol=0)


def test_gradients_in_the_laws_away_from_the_reference_are_finite_differences() -> None:
    """At 30 K above the input's temperature, the gradient of the curve of a device with free
    ions, bulk and interface traps (`pin_iface_light`) in the laws' powers and activation
    energies (`N_c`, the mobilities, the capture coefficients of the bulk and the interfaces,
    `ν_int`) and the gap's slope and share, the levels and work functions following mid-gap:
    through `device_tree` and `solve_batch`, against central finite differences along a random
    direction."""
    structure, p = parameters("pin_iface_light")
    V = np.round(np.arange(-0.1, 0.95 + 1e-9, 0.05), 6)
    w = jnp.asarray(np.random.default_rng(0).normal(size=len(V)))

    def loss(a):
        laws = {"N_c": Law(m=1.5 * a[0]), "mu_n": Law(m=-1.5 * a[1], E=0.05 * a[2]),
                "mu_p": Law(m=-1.0 * a[1], E=0.03 * a[2]),
                "C_n_bulk": Law(m=0.5, E=0.1 * a[3]), "C_p_bulk": Law(E=0.08 * a[3]),
                "C_n_int": Law(m=0.5 * a[6], E=0.1 * a[3]), "nu_int_n": Law(m=0.5 * a[6]),
                "nu_int_p": Law(m=0.5 * a[6])}  # fmt: skip
        gap = Gap(slope=3e-4 * a[4], share=0.5 * a[5])
        q = at_temperature(p, p["T"] + 30.0, laws, gap, traps="midgap", contacts="midgap")
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, q))
        J, ok, _ = solve_batch(batch, V)
        return jnp.sum(w * jnp.where(ok[0], J[0], 0.0))

    a0 = jnp.ones(7)
    direction = jnp.asarray(np.random.default_rng(1).normal(size=7))
    by_grad = float(jax.grad(loss)(a0) @ direction)
    h = 1e-6
    fd = (float(loss(a0 + h * direction)) - float(loss(a0 - h * direction))) / (2 * h)
    assert abs(by_grad - fd) < 1e-6 * abs(fd)


def test_the_ions_activation_energy_through_a_scan_is_finite_differences() -> None:
    """A scan's current (the first 20 steps of `zimt_ions_fast_light`, mobile ions, 0 to 0.14 V
    in 8 ms) at 30 K above the input's temperature: its tangent in the ions' activation energy
    (`textbook_laws`), through `solve_transient`, against central finite differences."""
    from jaxdd.transient import solve_transient

    setup = read_setup(DATA / "zimt_ions_fast_light" / "simulation_setup.txt")
    tvg = read_tvg(setup)
    fields = ("t", "Vext", "oc", "G_frac", "track", "dti")
    tvg = dataclasses.replace(tvg, **{k: getattr(tvg, k)[:20] for k in fields})
    structure, p = _device_parameters(setup, grid="simss")
    p = jax.tree_util.tree_map(jnp.asarray, p)

    def curve(E):
        q = at_temperature(p, p["T"] + 30.0, textbook_laws(E_anion=E, E_cation=0.5))
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(structure, q))
        return solve_transient(batch, tvg, bernoulli_name="simss", ions=True,
                               vint=False).J_ext[0]

    E0 = jnp.asarray(0.4)
    _, tangent = jax.jvp(curve, (E0,), (jnp.ones_like(E0),))
    h = 1e-4
    fd = (curve(E0 + h) - curve(E0 - h)) / (2 * h)
    assert np.max(np.abs(tangent - fd)) < 1e-5 * np.max(np.abs(fd))
    assert np.max(np.abs(fd)) > 1e-5 * np.max(np.abs(curve(E0)))  # the ions move it
