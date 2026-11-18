"""Free ends (`W_L`, `W_R` = free): stacks with one electrode or none (docs/model.md, "Free
surfaces"). SimSS has no such end, so the references are analytic and the device's own
invariances."""

import re
from pathlib import Path

import numpy as np
import pytest

from jaxdd.batch import prepare
from jaxdd.device import build_device
from jaxdd.simss_input import setup_from_texts
from jaxdd.solver import solve_jv, solve_state

CASE = Path(__file__).parent / "data" / "none_light"
SETUP = (CASE / "simulation_setup.txt").read_text()
ABSORBER = (CASE / "L2_parameters.txt").read_text()
ETL = {"L": "30e-9", "E_c": 3.9, "E_v": 6.2, "N_D": "1e23", "mu_n": "1e-4", "mu_p": "1e-4",
       "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0, "k_direct": "1e-18", "eps_r": 4}  # fmt: skip
HTL = {"L": "20e-9", "E_c": 2.5, "E_v": 5.4, "N_A": "1e23", "mu_n": "1e-4", "mu_p": "1e-4",
       "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0, "k_direct": "1e-18", "eps_r": 3}  # fmt: skip


def _set(text, key, value):
    new, k = re.subn(rf"^{key}\s*=\s*\S+", f"{key} = {value}", text, flags=re.M)
    assert k == 1, key
    return new


def stack(layers, W_L="free", W_R="free", S=(0, 0, 0, 0), NP=400, **setup):
    """A device of `layers` (each the absorber of `none_light` with the given changes)."""
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
    return build_device(setup_from_texts(files))


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


@pytest.mark.parametrize(
    ("layer", "S", "what"),
    [
        ({}, (-1, -1, 0, 0), "S_n_L < 0 at a free end"),
        ({"N_anion": "1e22", "N_cation": "1e22", "mu_anion": "1e-12", "mu_cation": "1e-12",
          "ionsMayEnter": 1},
         (0, 0, 0, 0), "mobile ions"),
    ],
)  # fmt: skip
def test_what_a_free_end_does_not_yet_solve_is_refused(layer, S, what) -> None:
    with pytest.raises(NotImplementedError, match=what):
        stack([layer], S=S)


def test_a_batch_refuses_a_free_end() -> None:
    with pytest.raises(NotImplementedError, match="free end"):
        prepare([stack([{}])])
