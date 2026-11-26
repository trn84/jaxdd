"""Solves for `test_reproducible.py`, run in a process of their own: a J-V scan with ions frozen
after a pre-bias (`solve_batch`), the steady states of a free film (`solve_states`), and a
transient with mobile ions (`solve_transient`), every array they return written to the `.npz`
file named on the command line. Not a test module (no `test_` prefix)."""

import dataclasses
import sys
from pathlib import Path

import numpy as np

from jaxdd.batch import prepare, solve_batch, solve_states
from jaxdd.device import build_device
from jaxdd.simss_input import read_setup, read_tvg, simss_voltages
from jaxdd.transient import solve_transient

DATA = Path(__file__).parent / "data"


def setup(case):
    return read_setup(DATA / case / "simulation_setup.txt")


def main(out: str) -> None:
    arrays = {}
    jv = setup("frozen_1e17_x1_thin_light")
    J, ok, its = solve_batch(prepare([build_device(jv)]), np.sort(simss_voltages(jv)))
    arrays.update(jv_J=J, jv_ok=ok, jv_its=its)
    states = solve_states(prepare([build_device(setup("thin_l1_light"))]), [0.0, 0.5, 1.0])
    arrays.update({f"states_{k}": v for k, v in states.items()})
    ions = setup("zimt_ions_fast_light")
    tvg = read_tvg(ions)
    keys = ("t", "Vext", "oc", "G_frac", "track", "dti")
    tvg = dataclasses.replace(tvg, **{k: getattr(tvg, k)[:41] for k in keys})
    r = solve_transient(prepare([build_device(ions)]), tvg, profiles=True)
    arrays.update(tr_J=r.J_ext, tr_V=r.V_int, tr_its=r.iterations)
    arrays.update({f"tr_{k}": v for k, v in r.profiles.items()})
    np.savez(out, **{k: np.asarray(v) for k, v in arrays.items()})


if __name__ == "__main__":
    main(sys.argv[1])
