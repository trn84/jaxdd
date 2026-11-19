"""Free ends (`W_L`, `W_R` = free): stacks with one electrode or none (docs/model.md, "Free
surfaces"). SimSS has no such end, so the references are analytic and the device's own
invariances."""

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxdd.batch import device_tree, prepare, solve_batch, solve_states
from jaxdd.constants import EPS_0, Q
from jaxdd.device import build_device, device_parameters
from jaxdd.simss_input import TVG, read_setup, setup_from_texts, simss_voltages
from jaxdd.solver import solve_jv, solve_state
from jaxdd.transient import solve_transient, tj_columns

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
    is one number. Each state converged: where the voltage drives the blocked carrier into
    the contact (an accumulation layer, e^(V/V_T) above equilibrium) Newton does not reach it
    from SIMsalabim's start (PLAN.md), so the work functions and voltages are those it reaches."""
    free = solve_state(stack([HTL, {}], W_L=5.0, S=(-1, -1, 0, 0)))
    assert free.converged
    blocking = [solve_state(stack([HTL, {}], W_L=5.0, W_R=W, S=(-1, -1, 0, 0)), Vint=V)
                for W, V in ((5.3, 0.0), (5.4, 0.0), (5.4, 0.5), (5.6, 0.5))]  # fmt: skip
    assert all(r.converged for r in blocking)
    voc = np.array([r.implied_voc for r in blocking])
    assert np.ptp(voc) > 0.03
    assert min(abs(voc - free.implied_voc)) > 1e-3


def test_a_blocking_contact_in_equilibrium_converges() -> None:
    """In the dark at 0 V a contact's two branches (`_contact_ends`) give the same density, to
    round-off: the branch held stays (`solver.BRANCH_TIE`). Decided by the sign of that
    round-off it flipped at every check, each flip starting Newton's convergence anew, and with
    `S` = 0 the electrons, blocked behind the HTL's barrier, floated off: no state converged."""
    for S in ((-1, -1, 0, 0), (-1, -1, -1, 0), (-1, -1, 1e-3, 1e-3)):
        d = stack([HTL, {}], W_L=5.0, W_R=4.0, S=S)
        r = solve_state(d, g_frac=0.0)
        assert r.converged, S
        np.testing.assert_allclose(r.n * r.p, d.ni**2, rtol=1e-6)


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


def test_a_free_surface_needs_its_recombination_velocities() -> None:
    with pytest.raises(ValueError, match="S_n_L < 0 at a free end"):
        stack([{}], S=(-1, -1, 0, 0))


def test_field_dependent_rates_in_a_film_without_field() -> None:
    """A uniform film with both ends free has no field anywhere, so the rates in the field take
    their zero-field values. Braun's generation (one distance `a`, `P0 = 0`): `P·G =
    (1 − P)·k·(n² − n_i²)` with `P = k_d/(k_d + k_f)`, `k_d = 3k/(4π·a³)·exp(−E_B/V_T)`,
    `E_B = q/(4π·ε·a)`. Langevin's rate in a field-dependent mobility: `G = k_L·(n² − n_i²)`,
    `k_L = q·(μ_n + μ_p)/ε`."""
    G, k, a, k_f, eps = 1.5e28, 4.7281398e-17, 1e-9, 1e9, 24.1 * EPS_0
    d = stack([{"N_t_bulk": 0, "fieldDepG": 1, "thermLengDist": 1, "a": a, "k_f": k_f}])
    r = solve_state(d)
    k_d = 3 * k / (4 * np.pi * a**3) * np.exp(-Q / (4 * np.pi * eps * a) / d.Vt)
    P = k_d / (k_d + k_f)
    n = np.sqrt(P * G / ((1 - P) * k) + d.ni[1] ** 2)
    assert r.converged and 0.1 < P < 0.9
    np.testing.assert_allclose(r.n[1:-1], n, rtol=1e-8)
    assert r.plqy == pytest.approx(P, rel=1e-6)  # the dissociated share recombines directly
    mu = 1e-4
    d = stack([{"N_t_bulk": 0, "useLangevin": 1, "mobnDep": 1, "mobpDep": 1, "gamma_n": 1e-4,
                "gamma_p": 1e-4, "mu_n": mu, "mu_p": mu}])  # fmt: skip
    r = solve_state(d)
    n = np.sqrt(G * eps / (Q * 2 * mu) + d.ni[1] ** 2)
    assert r.converged
    np.testing.assert_allclose(r.n[1:-1], n, rtol=1e-8)


def test_field_dependent_generation_in_a_free_stack() -> None:
    """Braun's generation in the absorber of a charged HTL/absorber/ETL with both ends free,
    where the interfaces' fields split pairs: converged, the same in a batch, and its implied
    V_oc converged in the grid."""
    braun = {"fieldDepG": 1, "a": 1e-9, "k_f": 1e9}
    r = solve_state(stack([HTL, braun, ETL], S=(10, 10, 10, 10)))
    fine = solve_state(stack([HTL, braun, ETL], S=(10, 10, 10, 10), NP=1600))
    assert r.converged and fine.converged
    assert fine.implied_voc == pytest.approx(r.implied_voc, abs=1e-5)
    b = solve_states(prepare([stack([HTL, braun, ETL], S=(10, 10, 10, 10))]))
    assert float(b["implied_voc"][0, 0]) == pytest.approx(r.implied_voc, abs=1e-8)
    assert float(b["plqy"][0, 0]) == pytest.approx(r.plqy, rel=1e-7)


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


def _tvg(times, G, V=None, oc=None):
    """Time steps `(t, G_frac)` at `Vext` = `V` (0) or the open circuit (`oc`)."""
    t = np.asarray(times, dtype=float)
    return TVG(t=t, Vext=np.zeros_like(t) if V is None else np.asarray(V, dtype=float),
               oc=np.zeros(len(t), bool) if oc is None else np.asarray(oc),
               G_frac=np.asarray(G, dtype=float), track=np.zeros(len(t), int),
               dti=np.concatenate([[0.0], 1 / np.diff(t)]))  # fmt: skip


def _decay(steps, dt):
    """The light on at t = 0 (the steady state), then off for `steps` steps of `dt`."""
    return _tvg(np.arange(steps + 1) * dt, np.r_[1.0, np.zeros(steps)])


def test_a_free_film_decays_as_implicit_euler() -> None:
    """A uniform film with both ends free, direct recombination alone, the light switched off:
    each step of implicit Euler of `dn/dt = −k·(n² − n_i²)` in closed form, to 1e-9 (the
    density from the QFLS column); no current."""
    d = stack([{"N_t_bulk": 0}])
    dt, k, G = 1e-7, 4.7281398e-17, 1.5e28
    r = solve_transient(prepare([d]), _decay(30, dt))
    assert r.converged.all() and (r.J == 0).all()
    ni = d.ni[1]
    n = [np.sqrt(G / k + ni**2)]
    for _ in range(30):
        a = dt * k
        n.append((-1 + np.sqrt(1 + 4 * a * (n[-1] + a * ni**2))) / (2 * a))
    QFLS = tj_columns(r, 0)["QFLSL1"]
    np.testing.assert_allclose(ni * np.exp(QFLS / (2 * d.Vt)), n, rtol=1e-9)
    # Each step's profiles (`profiles=True`): the same steps, and the film uniform at `n`.
    with_profiles = solve_transient(prepare([d]), _decay(30, dt), profiles=True)
    np.testing.assert_array_equal(tj_columns(with_profiles, 0)["QFLSL1"], QFLS)
    prof = with_profiles.profiles
    assert prof["n"].shape == (1, 31, len(d.x)) and r.profiles is None
    uniform = np.broadcast_to(np.array(n)[:, None], prof["n"][0, :, 1:-1].shape)
    np.testing.assert_allclose(prof["n"][0, :, 1:-1], uniform, rtol=1e-9)
    np.testing.assert_allclose(prof["p"][0], prof["n"][0], rtol=1e-9)


def test_a_free_stack_in_time() -> None:
    """HTL/absorber with charged traps and mobile ions/ETL, both ends free, through a decay: every
    step converged, its first the steady state of `solve_state`, and the splitting at the end
    converging at first order in the time step (implicit Euler)."""
    layers = [HTL, {**IONS, "bulkTrapType": -1, "N_t_bulk": "1e22"}, ETL]
    d = stack(layers, S=(10, 10, 10, 10))
    ends = []
    for k in (10, 20, 40):  # 40 steps each (one program), the splitting at 2e-7 s, step k
        r = solve_transient(prepare([d]), _decay(40, 2e-7 / k))
        assert r.converged.all() and (r.J == 0).all()
        ends.append(tj_columns(r, 0)["QFLSL2"][k])
    steady = solve_state(d)
    absorber = (steady.x[1:] > 20e-9) & (steady.x[:-1] < 520e-9)
    first = tj_columns(r, 0)["QFLSL2"][0]
    w = np.diff(steady.x) * absorber
    assert first == pytest.approx(np.sum(w * 0.5 * (steady.qfls[1:] + steady.qfls[:-1])) / w.sum(),
                                  abs=1e-3)  # fmt: skip
    ratio = (ends[0] - ends[1]) / (ends[1] - ends[2])
    assert 1.6 < ratio < 2.4  # first order: each halving of the step halves the change


def test_one_free_end_in_time_does_not_feel_the_voltage() -> None:
    """A film on one contact: a voltage step at the contact (or its open circuit) only moves the
    potential's gauge, so the splitting is that at 0 V, and no current flows; in a batch with a
    cell at its open circuit (the internal voltage an unknown) the cell is as alone."""
    d = stack([HTL, {}], W_L=5.0, S=(-1, -1, 100, 100))
    times = np.arange(11) * 1e-8
    G = np.r_[1.0, np.zeros(10)]
    at_zero = solve_transient(prepare([d]), _tvg(times, G))
    stepped = solve_transient(prepare([d]), _tvg(times, G, V=np.r_[0.0, np.full(10, 0.5)]))
    q0, q1 = (tj_columns(r, 0)["QFLSL2"] for r in (at_zero, stepped))
    assert at_zero.converged.all() and stepped.converged.all() and (stepped.J == 0).all()
    np.testing.assert_allclose(q1, q0, atol=1e-9)
    cell = build_device(read_setup(CASE / "simulation_setup.txt"))
    oc = np.ones(len(times), bool)
    mixed = solve_transient(prepare([d, cell]), [_tvg(times, G, oc=oc)] * 2)
    alone = solve_transient(prepare([cell]), _tvg(times, G, oc=oc))
    assert mixed.converged.all()
    np.testing.assert_allclose(tj_columns(mixed, 0)["QFLSL2"], q0, atol=1e-9)
    np.testing.assert_allclose(mixed.V_int[1], alone.V_int[0], atol=1e-9)
