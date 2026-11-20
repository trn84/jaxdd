"""Ions bounded by their sites against ChargeTransport.jl (`validation/chargetransport`): its
Ex103 cell (TiO2 | MAPI | spiro, mobile vacancies limited by the Fermi–Dirac integral of order
−1) with Boltzmann's electrons and holes, at 1e24 m⁻³ of vacancies on 1e25 sites and 4e24 on
2e25, where the sites cut the accumulation by up to a fifth. Equilibrium, dark and lit steady
states, and the transients after a voltage's step (docs/validation.md, "Ions bounded by their
sites, against ChargeTransport.jl"); and the carriers' statistics, Fermi–Dirac's and Blakemore's,
in degenerate transport layers."""

import re
from pathlib import Path

import numpy as np
import pytest

from jaxdd.batch import prepare, solve_states
from jaxdd.device import build_device
from jaxdd.simss_input import TVG, setup_from_texts
from jaxdd.transient import solve_transient

DATA = Path(__file__).parent / "data" / "chargetransport_ions"
#: Each layer's conduction band edge (eV): which of jaxdd's points are the layer's.
E_C = (4.0, 3.7, 3.4)
#: The voltages solved, from the pre-bias at 0 V outwards; those without a reference are steps on
#: the way.
CASES = {"dark": (-0.3, -0.2, -0.1, 0.0, 0.6, 1.0), "light": (0.0, 0.9)}


#: The perovskite's bulk traps (SIMsalabim's L2.txt): ChargeTransport.jl's SRH lifetimes,
#: `τ_n = 1/(N_t·C_n)` = 3 ns and `τ_p` = 300 ns.
TRAPS = (1e21, 1 / (1e21 * 3e-9), 1 / (1e21 * 3e-7))


def device(ions, sites, G_frac, mu=None, doping=None, statistics="boltzmann", trap_share=1.0,
           NP=None, extra=()):  # fmt: skip
    """The cell with `ions` on `sites` in the perovskite; `doping` the transport layers' (the
    ETL's donors and the HTL's acceptors, else 1e24 m⁻³), `statistics` the carriers'.
    `trap_share`: that share of the perovskite's traps at the same lifetimes (their capture
    coefficients over it), so that what they hold in time is that share. `NP` the grid's points
    (the reference's 2000), `extra` more `(file, key, value)` changes."""
    files = {name: (DATA / name).read_text() for name in ("L1.txt", "L2.txt", "L3.txt")}
    changes = [("L2.txt", "N_anion", ions), ("L2.txt", "N_cation", ions),
               ("L2.txt", "N_max_cation", sites)]  # fmt: skip
    if mu is not None:
        changes.append(("L2.txt", "mu_cation", mu))
    if trap_share != 1.0:
        Nt, Cn, Cp = TRAPS
        changes += [("L2.txt", "N_t_bulk", Nt * trap_share),
                    ("L2.txt", "C_n_bulk", Cn / trap_share),
                    ("L2.txt", "C_p_bulk", Cp / trap_share)]  # fmt: skip
    changes += list(extra)
    if doping is not None:
        changes += [("L1.txt", "N_D", doping), ("L3.txt", "N_A", doping)]
    for name, key, value in changes:
        files[name], k = re.subn(rf"^{key} = \S+", f"{key} = {value:g}", files[name], flags=re.M)
        assert k == 1, key
    setup = (DATA / "simulation_setup.txt").read_text()
    setup = re.sub(r"^G_frac = \S+", f"G_frac = {G_frac}", setup, flags=re.M)
    if NP is not None:
        setup = re.sub(r"^NP = \S+", f"NP = {NP}", setup, flags=re.M)
    files["simulation_setup.txt"] = setup
    return build_device(setup_from_texts(files), statistics=statistics)


def worst(d, st, k, ref):
    """The largest differences of a state (`solve_states`' `k`-th) from the reference's on its
    nodes, within each layer's own points: `ln n` and `ln p` (above `n_floor`) and the potential
    (the reference's in jaxdd's gauge, from the left contact), over all three layers."""
    x, Ec = np.asarray(d.x), np.asarray(d.E_c)
    floor = 1e-12 * float(np.max(d.N_c))
    Vs, n, p = st["V"][0, k], st["n"][0, k], st["p"][0, k]
    gauge = Vs[0] - ref["psi"][0]
    dens = dV = 0.0
    for layer in (1, 2, 3):
        ours = np.isclose(Ec, E_C[layer - 1])
        a = ref[ref["region"] == layer]
        a = a[(a["x"] >= x[ours][0]) & (a["x"] <= x[ours][-1])]
        for got, want in ((n, a["n"]), (p, a["p"])):
            above = want > floor
            if above.any():
                err = np.interp(a["x"][above], x[ours], np.log(got[ours])) - np.log(want[above])
                dens = max(dens, float(np.abs(err).max()))
        dV = max(dV, float(np.abs(np.interp(a["x"], x[ours], Vs[ours]) - gauge - a["psi"]).max()))
    return dens, dV


def reference(name, case):
    path = DATA / name / f"{case}.csv"
    J = float(re.search(r"J = (\S+)", path.read_text().splitlines()[0]).group(1))
    return J, np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


@pytest.mark.parametrize("ions, sites", [(1e24, 1e25), (4e24, 2e25), (1.6e25, 1e27)])
def test_steady_states_with_sites_are_chargetransport_jl(ions, sites) -> None:
    """On the reference's nodes, within each layer's own points: `ln n`, `ln p` to 0.04 (spiro's,
    the rest to 0.01; with both grids refined spiro's falls to 3e-3), the ions' `ln` to 4e-3, the
    potential (the reference's in jaxdd's gauge, from the left contact) to 1 mV, the current to
    1e-3 of itself where it is above 1e-3 A/m² (below, both are 0 to their accuracy). Densities
    below `n_floor` (1e-12 of the largest `N_c`: a doped transport layer's minority carriers in
    the dark, 1e-3 m⁻³ and less) do not count for convergence (docs/solver.md), nor here."""
    name = f"ions{ions:g}_sites{sites:g}".replace("+", "")
    for kind, voltages in CASES.items():
        d = device(ions, sites, 1.0 if kind == "light" else 0.0)
        floor = 1e-12 * float(np.max(d.N_c))  # `n_floor`
        st = solve_states(prepare([d]), list(voltages), max_iter=200)
        x, Ec = np.asarray(d.x), np.asarray(d.E_c)
        for k, V in enumerate(voltages):
            case = "eq" if kind == "dark" and V == 0.0 else f"{kind}_{V}"
            assert bool(st["converged"][0, k]), case
            if not (DATA / name / f"{case}.csv").exists():
                continue
            J_ref, ref = reference(name, case)
            Vs, n, p = st["V"][0, k], st["n"][0, k], st["p"][0, k]
            cation = st["cation"][0, k]
            gauge = Vs[0] - ref["psi"][0]
            for layer in (1, 2, 3):
                ours = np.isclose(Ec, E_C[layer - 1])
                a = ref[ref["region"] == layer]
                a = a[(a["x"] >= x[ours][0]) & (a["x"] <= x[ours][-1])]
                tol = 0.04 if layer == 3 else 0.012
                for got, want in ((n, a["n"]), (p, a["p"])):
                    above = want > floor
                    err = np.interp(a["x"][above], x[ours], np.log(got[ours])) - np.log(want[above])
                    assert not above.any() or np.abs(err).max() < tol, (case, layer)
                dV = np.interp(a["x"], x[ours], Vs[ours]) - (gauge + a["psi"])
                assert np.abs(dV).max() < 1e-3, (case, layer)
                if layer == 2:
                    err = np.abs(np.interp(a["x"], x[ours], np.log(cation[ours])) - np.log(a["na"]))
                    assert err.max() < 4e-3, case
            if abs(J_ref) > 1e-3:
                assert float(st["J"][0, k]) == pytest.approx(J_ref, rel=1e-3), case


#: The voltage steps: ions, sites, the step (V); the ions' mobility the reference's 1e-12 m²/Vs.
#: 1.6e25 on 1e27 sites is Ex103's own cell, which needs the start's stages in the ions' density
#: (`batch._start`).
STEPS = ((1e24, 1e25, 1.0), (4e24, 2e25, 0.6), (4e24, 2e25, 1.0), (1.6e25, 1e27, 1.0))


def test_transients_with_sites_are_chargetransport_jl() -> None:
    """A voltage's step from equilibrium in the dark, 100 s on ChargeTransport's own 122 times
    (implicit Euler both, so what is left is the grids'), the vacancies at 1e-12 m²/Vs: the
    flux of the electrochemical potential at a constant mobility (`solver._excess`). The
    current to 2e-2 of itself to 1 ms (the field's relaxation over the grids, and with Ex103's
    1.6e25 m⁻³ of ions their fastest motion, 1.5e-2 at most there whether jaxdd's grid is 2000
    points or 4000; the median 3e-3) and 3e-3 after, while the ions move and when they have
    settled (with Boltzmann's flux, sites 1e35, 1.6e-2 there); the states at 1e-6, 1e-3 and
    1 s, `ln n`, `ln p` to 0.1, the potential to 0.5 mV. The last step is the steady state at
    the voltage (`solve_states`)."""
    devices, tvgs, refs = [], [], []
    for ions, sites, V in STEPS:
        name = f"ions{ions:g}_sites{sites:g}".replace("+", "")
        ref = np.genfromtxt(DATA / name / f"tstep_dark_{V}_J.csv", delimiter=",", names=True)
        t = np.concatenate([[0.0], ref["t"]])
        n = len(t)
        tvgs.append(TVG(t=t, Vext=np.concatenate([[0.0], np.full(n - 1, V)]),
                        oc=np.zeros(n, bool), G_frac=np.zeros(n), track=np.zeros(n, int),
                        dti=np.concatenate([[0.0], 1.0 / np.diff(t)])))  # fmt: skip
        devices.append(device(ions, sites, 0.0, mu=1e-12))
        refs.append((name, V, ref, t))
    r = solve_transient(prepare(devices), tvgs, profiles=True)
    steady = solve_states(prepare(devices), [0.0, 0.6, 1.0], max_iter=200)
    assert np.asarray(r.converged).all()
    for b, (name, V, ref, t) in enumerate(refs):
        J = np.asarray(r.J)[b, 1:]
        rel = np.abs(J - ref["J"]) / np.abs(ref["J"])
        for hi, tol in ((1e-6, 2e-2), (1e-3, 2e-2), (np.inf, 3e-3)):
            lo = {1e-6: 0.0, 1e-3: 1e-6, np.inf: 1e-3}[hi]
            m = (ref["t"] > lo) & (ref["t"] <= hi)
            assert rel[m].max() < tol, (name, V, hi)
        assert J[-1] == pytest.approx(float(steady["J"][b, [0.0, 0.6, 1.0].index(V)]), rel=1e-6)
        d = devices[b]
        x, Ec = np.asarray(d.x), np.asarray(d.E_c)
        floor = 1e-12 * float(np.max(d.N_c))
        for j in (1, 2, 3):
            tj = float((DATA / name / f"tstep_dark_{V}_at{j}_t.txt").read_text())
            k = int(np.argmin(np.abs(t - tj)))
            a = np.genfromtxt(DATA / name / f"tstep_dark_{V}_at{j}.csv", delimiter=",",
                              names=True, skip_header=1)  # fmt: skip
            Vs = np.asarray(r.profiles["V"])[b, k]
            gauge = Vs[0] - a["psi"][0]
            for layer in (1, 2, 3):
                ours = np.isclose(Ec, E_C[layer - 1])
                c = a[a["region"] == layer]
                c = c[(c["x"] >= x[ours][0]) & (c["x"] <= x[ours][-1])]
                for key in ("n", "p"):
                    got = np.log(np.asarray(r.profiles[key])[b, k])
                    above = c[key] > floor
                    err = np.interp(c["x"][above], x[ours], got[ours]) - np.log(c[key][above])
                    assert not above.any() or np.abs(err).max() < 0.1, (name, V, j, layer, key)
                dV = np.interp(c["x"], x[ours], Vs[ours]) - (gauge + c["psi"])
                assert np.abs(dV).max() < 5e-4, (name, V, j, layer)


#: The voltage steps with the modified drift (`ion_flux="modified-drift"`), each against
#: ChargeTransport.jl's `DiffusionEnhancedModifiedDrift` (`FLUX=modified-drift`).
DRIFT_STEPS = ((1e24, 1e25, 1.0), (4e24, 2e25, 0.6), (4e24, 2e25, 1.0))


def test_transients_with_the_modified_drift_are_chargetransport_jl() -> None:
    """The steps of `STEPS` with the ions' drift modified by the free sites' share at a constant
    diffusivity, against ChargeTransport.jl's Bessemoulin-Chatard scheme with a modified drift:
    the current to 1e-2 of itself to 1 µs (8.7e-3 at most), 1.5e-2 to 1 ms (1.0e-2) and 3e-3
    after (2.2e-3); the states at 1e-6, 1e-3 and 1 s, `ln n`, `ln p` to 0.1 (9.5e-2 at 1 µs, as
    the electrochemical flux's), the potential to 0.5 mV. The two fluxes' transients differ by
    up to 0.1–1 of the current (the references' own difference): the electrochemical flux's
    against this reference is 0.13–0.5 off, here more than 0.1."""
    devices, tvgs, refs = [], [], []
    for ions, sites, V in DRIFT_STEPS:
        name = f"ions{ions:g}_sites{sites:g}".replace("+", "") + "/modified-drift"
        ref = np.genfromtxt(DATA / name / f"tstep_dark_{V}_J.csv", delimiter=",", names=True)
        t = np.concatenate([[0.0], ref["t"]])
        n = len(t)
        tvgs.append(TVG(t=t, Vext=np.concatenate([[0.0], np.full(n - 1, V)]),
                        oc=np.zeros(n, bool), G_frac=np.zeros(n), track=np.zeros(n, int),
                        dti=np.concatenate([[0.0], 1.0 / np.diff(t)])))  # fmt: skip
        devices.append(device(ions, sites, 0.0, mu=1e-12))
        refs.append((name, V, ref, t))
    r = solve_transient(prepare(devices), tvgs, profiles=True, ion_flux="modified-drift")
    assert np.asarray(r.converged).all()
    for b, (name, V, ref, t) in enumerate(refs):
        J = np.asarray(r.J)[b, 1:]
        rel = np.abs(J - ref["J"]) / np.abs(ref["J"])
        for lo, hi, tol in ((0.0, 1e-6, 1e-2), (1e-6, 1e-3, 1.5e-2), (1e-3, np.inf, 3e-3)):
            m = (ref["t"] > lo) & (ref["t"] <= hi)
            assert rel[m].max() < tol, (name, V, lo, rel[m].max())
        other = np.genfromtxt(DATA / name.split("/")[0] / f"tstep_dark_{V}_J.csv", delimiter=",",
                              names=True)["J"]  # fmt: skip
        assert np.max(np.abs(other - ref["J"]) / np.abs(ref["J"])) > 0.1, (name, V)
        for j in (1, 2, 3):
            tj = float((DATA / name / f"tstep_dark_{V}_at{j}_t.txt").read_text())
            k = int(np.argmin(np.abs(t - tj)))
            state = {key: np.asarray(r.profiles[key])[b, k][None, None] for key in ("V", "n", "p")}
            dens, dV = worst(devices[b], state, 0, reference(name, f"tstep_dark_{V}_at{j}")[1])
            assert dens < 0.1 and dV < 5e-4, (name, V, j, dens, dV)


@pytest.mark.parametrize("statistics", ["fermi-dirac", "blakemore"])
def test_degenerate_transport_layers_are_chargetransport_jl(statistics) -> None:
    """The cell with its transport layers doped 3e25 m⁻³ against `N_c` 5e25 (degenerate: the
    Fermi level 0.3 eV within the band; 1e24 vacancies on 1e25 sites), each statistics against
    ChargeTransport.jl's own (Fermi–Dirac's `F_{1/2}` by TeSCA's approximation, both codes'):
    `ln n`, `ln p` to 0.04, the potential to 1 mV, the current to 3e-3 of itself where it is above
    1e-3 A/m² (2.3e-2, 0.5 mV and 1.3e-3 at most). Boltzmann's against the same reference is 6e-2
    and 7 mV off at 1.0 V in the dark, its current 9e-3 (docs/validation.md)."""
    name = f"degenerate_{statistics}"
    # the lit curve through 0.3 and 0.6 V: from 0 V straight to 0.9 the batch's Newton does not
    # converge in this cell, with any statistics, round-off deciding
    for kind, voltages in (("dark", (0.0, 0.6, 1.0)), ("light", (0.0, 0.3, 0.6, 0.9))):
        d = device(1e24, 1e25, 1.0 if kind == "light" else 0.0, doping=3e25, statistics=statistics)
        st = solve_states(prepare([d]), list(voltages), max_iter=200)
        for k, V in enumerate(voltages):
            case = "eq" if kind == "dark" and V == 0.0 else f"{kind}_{V}"
            assert bool(st["converged"][0, k]), case
            if not (DATA / name / f"{case}.csv").exists():
                continue
            J_ref, ref = reference(name, case)
            dens, dV = worst(d, st, k, ref)
            assert dens < 0.04 and dV < 1e-3, (case, dens, dV)
            if abs(J_ref) > 1e-3:
                assert float(st["J"][0, k]) == pytest.approx(J_ref, rel=3e-3), case
    if statistics == "fermi-dirac":  # what the test sees: Boltzmann's is far off
        d = device(1e24, 1e25, 0.0, doping=3e25)
        st = solve_states(prepare([d]), [1.0], max_iter=200)
        dens, dV = worst(d, st, 0, reference(name, "dark_1.0")[1])
        assert dens > 0.05 and dV > 5e-3


@pytest.mark.parametrize("statistics", ["fermi-dirac", "blakemore"])
def test_transients_with_another_statistics_are_chargetransport_jl(statistics) -> None:
    """The degenerate cell's voltage step from equilibrium to 1.0 V in the dark (the vacancies
    at 1e-12 m²/Vs, 100 s on ChargeTransport's 122 times), each statistics against
    ChargeTransport.jl's own: the current to 3e-2 of itself to 1 µs (the field's relaxation over
    the grids, 2.2e-2 at most), 2e-3 to 1 ms and 6e-3 after (4.5e-3, as the ions move); the
    states at 1e-6, 1e-3 and 1 s, `ln n`, `ln p` to 0.03, the potential to 1 mV; the end the
    steady state (`solve_states`). With 1e-4 of the traps at the same lifetimes: the reference's
    SRH takes the traps' occupation as the densities' at every instant (lifetimes only), jaxdd's
    carries it in time as ZimT's, and 1e21 m⁻³ of them fill with electrons for a millisecond in
    the half of the perovskite where the vacancies pile up: its electrons there 4 orders below
    the reference's at 1 ms, Boltzmann's statistics alike (docs/validation.md)."""
    name = f"degenerate_{statistics}"
    ref = np.genfromtxt(DATA / name / "tstep_dark_1.0_J.csv", delimiter=",", names=True)
    t = np.concatenate([[0.0], ref["t"]])
    n = len(t)
    tvg = TVG(t=t, Vext=np.concatenate([[0.0], np.full(n - 1, 1.0)]), oc=np.zeros(n, bool),
              G_frac=np.zeros(n), track=np.zeros(n, int),
              dti=np.concatenate([[0.0], 1.0 / np.diff(t)]))  # fmt: skip
    d = device(1e24, 1e25, 0.0, mu=1e-12, doping=3e25, statistics=statistics, trap_share=1e-4)
    r = solve_transient(prepare([d]), [tvg], profiles=True)
    assert np.asarray(r.converged).all()
    J = np.asarray(r.J)[0, 1:]
    rel = np.abs(J - ref["J"]) / np.abs(ref["J"])
    for lo, hi, tol in ((0.0, 1e-6, 3e-2), (1e-6, 1e-3, 2e-3), (1e-3, np.inf, 6e-3)):
        m = (ref["t"] > lo) & (ref["t"] <= hi)
        assert rel[m].max() < tol, (lo, hi, rel[m].max())
    steady = solve_states(prepare([d]), [0.0, 0.6, 1.0], max_iter=200)
    assert J[-1] == pytest.approx(float(steady["J"][0, -1]), rel=1e-6)
    for j in (1, 2, 3):
        tj = float((DATA / name / f"tstep_dark_1.0_at{j}_t.txt").read_text())
        k = int(np.argmin(np.abs(t - tj)))
        state = {key: np.asarray(r.profiles[key])[:, k][:, None] for key in ("V", "n", "p")}
        dens, dV = worst(d, state, 0, reference(name, f"tstep_dark_1.0_at{j}")[1])
        assert dens < 0.03 and dV < 1e-3, (j, dens, dV)
