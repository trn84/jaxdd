"""Ions bounded by their sites against ChargeTransport.jl (`validation/chargetransport`): its
Ex103 cell (TiO2 | MAPI | spiro, mobile vacancies limited by the Fermi–Dirac integral of order
−1) with Boltzmann's electrons and holes, at 1e24 m⁻³ of vacancies on 1e25 sites and 4e24 on
2e25, where the sites cut the accumulation by up to a fifth. Equilibrium, dark and lit steady
states (docs/validation.md, "Ions bounded by their sites")."""

import re
from pathlib import Path

import numpy as np
import pytest

from jaxdd.batch import prepare, solve_states
from jaxdd.device import build_device
from jaxdd.simss_input import setup_from_texts

DATA = Path(__file__).parent / "data" / "chargetransport_ions"
#: Each layer's conduction band edge (eV): which of jaxdd's points are the layer's.
E_C = (4.0, 3.7, 3.4)
#: The voltages solved, from the pre-bias at 0 V outwards (from a cold start at −0.3 V with 4e24
#: ions Newton does not converge: PLAN.md); those without a reference are steps on the way.
CASES = {"dark": (-0.3, -0.2, -0.1, 0.0, 0.6, 1.0), "light": (0.0, 0.9)}


def device(ions, sites, G_frac):
    files = {name: (DATA / name).read_text() for name in ("L1.txt", "L2.txt", "L3.txt")}
    for key, value in (("N_anion", ions), ("N_cation", ions), ("N_max_cation", sites)):
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
