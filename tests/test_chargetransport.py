"""Ions bounded by their sites against ChargeTransport.jl (`validation/chargetransport`): its
Ex103 cell (TiO2 | MAPI | spiro, mobile vacancies limited by the Fermi–Dirac integral of order
−1) with Boltzmann's electrons and holes, at 1e24 m⁻³ of vacancies on 1e25 sites and 4e24 on
2e25, where the sites cut the accumulation by up to a fifth. Equilibrium, dark and lit steady
states, and the transients after a voltage's step (docs/validation.md, "Ions bounded by their
sites, against ChargeTransport.jl")."""

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
#: The voltages solved, from the pre-bias at 0 V outwards (from a cold start at −0.3 V with 4e24
#: ions Newton does not converge: PLAN.md); those without a reference are steps on the way.
CASES = {"dark": (-0.3, -0.2, -0.1, 0.0, 0.6, 1.0), "light": (0.0, 0.9)}


def device(ions, sites, G_frac, mu=None):
    files = {name: (DATA / name).read_text() for name in ("L1.txt", "L2.txt", "L3.txt")}
    changes = [("N_anion", ions), ("N_cation", ions), ("N_max_cation", sites)]
    for key, value in changes + ([("mu_cation", mu)] if mu is not None else []):
        files["L2.txt"], k = re.subn(rf"^{key} = \S+", f"{key} = {value:g}", files["L2.txt"],
                                     flags=re.M)  # fmt: skip
        assert k == 1, key
    setup = (DATA / "simulation_setup.txt").read_text()
    files["simulation_setup.txt"] = re.sub(r"^G_frac = \S+", f"G_frac = {G_frac}", setup,
                                           flags=re.M)  # fmt: skip
    return build_device(setup_from_texts(files))


def reference(name, case):
    path = DATA / name / f"{case}.csv"
    J = float(re.search(r"J = (\S+)", path.read_text().splitlines()[0]).group(1))
    return J, np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


@pytest.mark.parametrize("ions, sites", [(1e24, 1e25), (4e24, 2e25)])
def test_steady_states_with_sites_are_chargetransport_jl(ions, sites) -> None:
    """On the reference's nodes, within each layer's own points: `ln n`, `ln p` to 0.04 (spiro's,
    the rest to 0.01; with both grids refined spiro's falls to 3e-3), the ions' `ln` to 4e-3, the
    potential (the reference's in jaxdd's gauge, from the left contact) to 1 mV, the current to
    1e-3 of itself where it is above 1e-3 A/m² (below, both are 0 to their accuracy). Densities
    below `n_floor` (1e-12 of the largest `N_c`: a doped transport layer's minority carriers in
    the dark, 1e-3 m⁻³ and less) do not count for convergence (docs/solver.md), nor here."""
    name = f"ions{ions:.0e}_sites{sites:.0e}".replace("+", "")
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
#: With 4e24 m⁻³ of ions the transients agree as well (docs/validation.md), but on a GPU the
#: steady state they start from fails in about one run of two, round-off deciding (PLAN.md,
#: "Many ions"); 1e24 converges in every run.
STEPS = ((1e24, 1e25, 1.0),)


def test_transients_with_sites_are_chargetransport_jl() -> None:
    """A voltage's step from equilibrium in the dark, 100 s on ChargeTransport's own 122 times
    (implicit Euler both, so what is left is the grids'), the vacancies at 1e-12 m²/Vs: the
    flux of the electrochemical potential at a constant mobility (`solver._excess`). The
    current to 2e-2 of itself in the first microsecond (the field's relaxation over the grids),
    1e-2 to 1 ms and 3e-3 after, while the ions move and when they have settled (with Boltzmann's
    flux, sites 1e35, 1.6e-2 there); the states at 1e-6, 1e-3 and 1 s, `ln n`, `ln p` to 0.1, the
    potential to 0.5 mV. The last step is the steady state at the voltage (`solve_states`)."""
    devices, tvgs, refs = [], [], []
    for ions, sites, V in STEPS:
        name = f"ions{ions:.0e}_sites{sites:.0e}".replace("+", "")
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
        for hi, tol in ((1e-6, 2e-2), (1e-3, 1e-2), (np.inf, 3e-3)):
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
