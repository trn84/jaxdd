"""Free ends (`W_L`, `W_R` = free): stacks with one electrode or none (docs/model.md, "Free
surfaces"). SimSS has no such end, so the references are analytic and the device's own
invariances."""

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxdd.batch import device_tree, prepare, solve_batch, solve_states
from jaxdd.device import build_device, device_parameters
from jaxdd.simss_input import TVG, read_setup, setup_from_texts, simss_voltages
from jaxdd.solver import solve_jv, solve_state
from jaxdd.transient import solve_transient

CASE = Path(__file__).parent / "data" / "none_light"
SETUP = (CASE / "simulation_setup.txt").read_text()
ABSORBER = (CASE / "L2_parameters.txt").read_text()
ETL = {"L": "30e-9", "E_c": 3.9, "E_v": 6.2, "N_D": "1e23", "mu_n": "1e-4", "mu_p": "1e-4",
       "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0, "k_direct": "1e-18", "eps_r": 4}  # fmt: skip
HTL = {"L": "20e-9", "E_c": 2.5, "E_v": 5.4, "N_A": "1e23", "mu_n": "1e-4", "mu_p": "1e-4",
       "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0, "k_direct": "1e-18", "eps_r": 3}  # fmt: skip


def _integral(f, x):
    """The trapezoidal rule (NumPy 1.26, the oldest supported, has no `np.trapezoid`)."""
    return np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x))


def _set(text, key, value):
    new, k = re.subn(rf"^{key}\s*=\s*\S+", f"{key} = {value}", text, flags=re.M)
    assert k == 1, key
    return new


def stack(layers, W_L="free", W_R="free", S=(0, 0, 0, 0), NP=400, **setup):
    """A device of `layers` (each the absorber of `none_light` with the given changes)."""
    return build_device(stack_setup(layers, W_L, W_R, S, NP, **setup))


def stack_setup(layers, W_L="free", W_R="free", S=(0, 0, 0, 0), NP=400, **setup):
    """`stack`'s input."""
    s = SETUP
    for k in ("l1", "l2", "l3"):
        s = re.sub(rf"^{k} = .*\n", "", s, flags=re.M)
    lines = "".join(f"l{i + 1} = L{i + 1}.txt * layer\n" for i in range(len(layers)))
    s = s.replace("**Contacts", lines + "**Contacts", 1)
    values = {"W_L": W_L, "W_R": W_R, "NP": NP, **setup}
    values.update(zip(("S_n_L", "S_p_L", "S_n_R", "S_p_R"), S, strict=True))
    for k, v in values.items():
        s = _set(s, k, v)
    files = {"simulation_setup.txt": s}
    for i, changes in enumerate(layers):
        text = ABSORBER
        for k, v in changes.items():
            text = _set(text, k, v)
        files[f"L{i + 1}.txt"] = text
    return setup_from_texts(files)


def test_a_film_without_contacts_holds_what_it_generates() -> None:
    """One undoped layer, both ends free, no surface recombination: uniform, `G = k·n² + SRH`
    with `n = p` (the trap at mid-gap: `N_t·C·(n² − n_i²)/(2·(n + n_i))`), and the PLQY the
    direct share."""
    d = stack([{}])
    r = solve_state(d)
    G, k, Nt, C = 1.5e28, 4.7281398e-17, 3.9473671e21, 5.2070311e-14
    ni = d.ni[1]
    # k·n² + Nt·C·(n − n_i)/2 = G + k·n_i²
    a, b, c = k, Nt * C / 2, -(G + k * ni**2 + Nt * C * ni / 2)
    n = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert r.converged
    np.testing.assert_allclose(r.n[1:-1], n, rtol=1e-10)
    np.testing.assert_allclose(r.p[1:-1], n, rtol=1e-10)
    assert r.implied_voc == pytest.approx(d.Vt * np.log(n**2 / ni**2), rel=1e-10)
    assert r.plqy == pytest.approx(k * (n**2 - ni**2) / G, rel=1e-9)
    assert r.J == 0.0


def test_ambipolar_diffusion_to_a_recombining_surface() -> None:
    """Equal mobilities, SRH at high injection (`τ = 2/(N_t·C)`), surface SRH `S/2·n` at the
    left, none at the right: `n = p = G·τ + A·cosh((x − L)/L_D)`."""
    Nt, C, S, G, mu, L = 1e21, 1e-14, 10.0, 1.5e28, 1e-4, 5e-7
    d = stack([{"N_t_bulk": Nt, "C_n_bulk": C, "C_p_bulk": C, "k_direct": 0, "mu_n": mu,
                "mu_p": mu}], S=(S, S, 0, 0))  # fmt: skip
    r = solve_state(d)
    D, tau = mu * d.Vt, 2 / (Nt * C)
    Ld = np.sqrt(D * tau)
    A = -(S / 2) * G * tau / (D * np.sinh(L / Ld) / Ld + (S / 2) * np.cosh(L / Ld))
    exact = G * tau + A * np.cosh((r.x - L) / Ld)
    assert r.converged
    np.testing.assert_allclose(r.n[1:-1], exact[1:-1], rtol=1e-5)
    np.testing.assert_allclose(r.p, r.n, rtol=1e-12)


def test_one_free_end_does_not_depend_on_the_voltage_or_the_contacts_role() -> None:
    """With one contact the voltage only moves the potential's gauge: the same densities, the
    potential shifted as a whole; no current."""
    states = [solve_state(stack([HTL, {}], W_L=5.0, S=(-1, -1, 100, 100), leftElec=le), Vint=V)
              for V, le in ((0.0, -1), (0.4, -1), (0.0, 1))]  # fmt: skip
    for r in states:
        assert r.converged and r.J == 0.0
        np.testing.assert_allclose(r.n, states[0].n, rtol=1e-8)
        np.testing.assert_allclose(r.p, states[0].p, rtol=1e-8)
        assert np.ptp(r.V - states[0].V) < 1e-9
    jv = solve_jv(stack([HTL, {}], W_L=5.0, S=(-1, -1, 100, 100)), np.array([0.0, 0.5, 1.0]))
    assert jv.converged.all() and (jv.J == 0).all()


def test_a_free_stack_is_neutral_without_field_beyond_its_ends() -> None:
    """HTL/absorber/ETL with both ends free: charged layers, but neutral as a whole (to the
    Poisson stencil's non-conservation at the jumps in ε), the field at either end small
    against the field inside, and the implied V_oc converged in the grid."""
    voc = []
    for NP in (400, 1600):
        r = solve_state(stack([HTL, {}, ETL], S=(10, 10, 10, 10), NP=NP))
        assert r.converged
        voc.append(r.implied_voc)
        d_x = np.gradient(r.x)
        rho = r.p - r.n + np.where(r.x < 20e-9, -1e23, 0.0) + np.where(r.x > 520e-9, 1e23, 0.0)
        assert abs(np.sum(rho * d_x)) < 1e-4 * np.sum(np.abs(rho) * d_x)
        E = np.diff(r.V) / np.diff(r.x)
        assert max(abs(E[1]), abs(E[-2])) < 1e-4 * np.max(np.abs(E))
    assert voc[0] == pytest.approx(voc[1], abs=1e-5)


def test_the_free_surface_is_not_a_blocking_contact() -> None:
    """SimSS's nearest device, a blocking contact (S = 0) in place of the free end, gives an
    implied V_oc that moves with the contact's work function and the voltage; the free end's
    is one number."""
    free = solve_state(stack([HTL, {}], W_L=5.0, S=(-1, -1, 0, 0))).implied_voc
    blocking = [solve_state(stack([HTL, {}], W_L=5.0, W_R=W, S=(-1, -1, 0, 0)), Vint=V).implied_voc
                for W in (4.0, 5.4) for V in (-0.5, 0.5)]  # fmt: skip
    assert np.ptp(blocking) > 0.05
    assert min(abs(np.array(blocking) - free)) > 1e-3


#: A perovskite's ions: mobile anions over immobile cations, 1e18 cm⁻³ each.
IONS = {"N_anion": "1e24", "N_cation": "1e24", "mu_anion": "1e-12", "mu_cation": 0,
        "ionsMayEnter": 1}  # fmt: skip


def test_mobile_ions_at_a_free_end() -> None:
    """HTL/absorber with mobile ions/ETL, both ends free: the ions all in their layer, the film
    neutral with them, the field at either end small; the same with the ions settled at a
    pre-bias and frozen (no voltage moves anything here), in a batch, and converged in the
    grid. With one contact, independent of the voltage as without ions."""
    d = stack([HTL, IONS, ETL], S=(10, 10, 10, 10))
    r = solve_state(d)
    assert r.converged
    absorber = (r.x > 20e-9) & (r.x < 520e-9)
    count = _integral(np.where(absorber, r.anion, 0.0), r.x)
    assert count == pytest.approx(1e24 * 500e-9, rel=1e-4)
    assert np.ptp(r.anion[absorber]) > 0.1 * 1e24  # the ions moved
    N_D = np.where(r.x > 520e-9, 1e23, 0.0)
    N_A = np.where(r.x < 20e-9, 1e23, 0.0)
    rho = r.p - r.n + N_D - N_A + r.cation - r.anion
    assert abs(_integral(rho, r.x)) < 1e-4 * _integral(np.abs(rho), r.x)
    E = np.diff(r.V) / np.diff(r.x)
    assert max(abs(E[1]), abs(E[-2])) < 1e-4 * np.max(np.abs(E))
    frozen = solve_state(stack([HTL, IONS, ETL], S=(10, 10, 10, 10), preCond=1, fixIons=1))
    assert frozen.implied_voc == pytest.approx(r.implied_voc, abs=1e-9)
    batch = solve_states(prepare([d]))
    assert float(batch["implied_voc"][0, 0]) == pytest.approx(r.implied_voc, abs=1e-8)
    np.testing.assert_allclose(np.asarray(batch["anion"][0, 0]), r.anion, rtol=1e-7)
    fine = solve_state(stack([HTL, IONS, ETL], S=(10, 10, 10, 10), NP=1600))
    assert fine.implied_voc == pytest.approx(r.implied_voc, abs=1e-5)
    one = [solve_state(stack([HTL, IONS], W_L=5.0, S=(-1, -1, 10, 10)), Vint=V) for V in (0, 0.5)]
    np.testing.assert_allclose(one[1].n, one[0].n, rtol=1e-8)
    np.testing.assert_allclose(one[1].anion, one[0].anion, rtol=1e-8)


@pytest.mark.parametrize(
    ("layer", "S", "what"),
    [
        ({}, (-1, -1, 0, 0), "S_n_L < 0 at a free end"),
        ({"fieldDepG": 1}, (0, 0, 0, 0), "field-dependent generation"),
    ],
)  # fmt: skip
def test_what_a_free_end_does_not_yet_solve_is_refused(layer, S, what) -> None:
    with pytest.raises(NotImplementedError, match=what):
        stack([layer], S=S)


def test_a_batch_with_free_ends_is_each_devices_own_solve() -> None:
    """A batch of a film without contacts, a charged free stack, a film on one contact and a
    cell between two (`none_light`): each device's state, implied V_oc and PLQY those of
    `solve_state` (the densities where Newton weighs them, above its floor: docs/solver.md), the
    cell's J-V curve that of a batch without the others."""
    cell = build_device(read_setup(CASE / "simulation_setup.txt"))
    devices = [stack([{}]), stack([HTL, {}, ETL], S=(10, 10, 10, 10)),
               stack([HTL, {}], W_L=5.0, S=(-1, -1, 100, 100)), cell]  # fmt: skip
    r = solve_states(prepare(devices))
    assert np.asarray(r["converged"]).all()
    for i, d in enumerate(devices):
        ref = solve_state(d)
        assert float(r["implied_voc"][i, 0]) == pytest.approx(ref.implied_voc, abs=1e-8)
        assert float(r["plqy"][i, 0]) == pytest.approx(ref.plqy, rel=1e-7)
        floor = 1e-12 * np.max(d.N_c)
        for got, want in ((r["n"][i, 0], ref.n), (r["p"][i, 0], ref.p)):
            weighed = want > floor
            np.testing.assert_allclose(np.asarray(got)[weighed], want[weighed], rtol=1e-7)
    assert (np.asarray(r["J"][:3]) == 0).all()
    V = simss_voltages(read_setup(CASE / "simulation_setup.txt"))[::6]
    J, ok, _ = solve_batch(prepare(devices), V)
    J_alone, _, _ = solve_batch(prepare([cell]), V)
    assert np.asarray(ok).all() and (np.asarray(J[:3]) == 0).all()
    np.testing.assert_allclose(np.asarray(J[3]), np.asarray(J_alone[0]), rtol=1e-9, atol=1e-9)


def test_the_implied_voc_is_differentiable_in_the_surface() -> None:
    """`device_tree`'s absorber/ETL film: the implied V_oc's derivative in the absorber's
    surface recombination velocity (the implicit function, `batch._converged_jvp`) is its
    central difference."""
    structure, params = device_parameters(stack_setup([{}, ETL], S=(10, 10, 10, 10)))

    def voc(log_S):
        p = dict(params, S_n_L=10.0**log_S, S_p_L=10.0**log_S)
        batch = jax.tree_util.tree_map(lambda a: a[None], device_tree(structure, p))
        return solve_states(batch, first_max_iter=None)["implied_voc"][0, 0]

    grad = float(jax.jvp(voc, (1.0,), (1.0,))[1])  # forward mode: one parameter
    h = 1e-3
    fd = (float(voc(1.0 + h)) - float(voc(1.0 - h))) / (2 * h)
    assert grad < 0
    assert grad == pytest.approx(fd, rel=1e-5)


def test_a_free_end_in_time_is_refused() -> None:
    tvg = TVG(t=np.array([0.0]), Vext=np.array([0.0]), oc=np.array([False]),
              G_frac=np.array([1.0]), track=np.array([0]), dti=np.array([0.0]))  # fmt: skip
    with pytest.raises(NotImplementedError, match="free end"):
        solve_transient(prepare([stack([{}])]), tvg)
