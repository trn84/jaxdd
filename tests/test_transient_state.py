"""A transient's state (`transient.TransientState`): any step's, from the profiles, and the last
one kept, from which `solve_transient(..., initial=)` goes on. What a caller that ages devices
step by step needs (PLAN.md §3.1): the state an ageing step leaves, the start of the next."""

import dataclasses
from pathlib import Path

import jax
import numpy as np
import pytest

from jaxdd.batch import prepare
from jaxdd.device import build_device
from jaxdd.simss_input import read_setup, read_tvg
from jaxdd.solver import _ion_number
from jaxdd.transient import TransientState, solve_transient

DATA = Path(__file__).parent / "data"

#: Mobile ions through a sweep and back; charged bulk traps and light on and off; charged
#: interface traps through a sweep; finite surface recombination at the contacts.
CASES = ["zimt_ions_fast_light", "zimt_acceptor_photo_light", "zimt_iface_sweep_light",
         "zimt_photo_surface_light"]  # fmt: skip


def _steps(tvg, part):
    """The steps `part` (a slice) of `tvg`."""
    keys = ("t", "Vext", "oc", "G_frac", "track", "dti")
    return dataclasses.replace(tvg, **{k: getattr(tvg, k)[part] for k in keys})


@pytest.fixture(scope="module", params=CASES)
def unsplit(request):
    """A case's device, its steps and its transient through all of them, with the profiles."""
    setup = read_setup(DATA / request.param / "simulation_setup.txt")
    tvg = read_tvg(setup)
    batch = prepare([build_device(setup, grid="simss")])
    return request.param, batch, tvg, solve_transient(batch, tvg, profiles=True)


def test_a_transient_split_at_a_step_goes_on_as_the_unsplit_one(unsplit) -> None:
    """Split at its middle step and continued from the first half's last state (`final`), or
    from that step's state in the unsplit run's profiles (`state(k)`), a transient gives the
    unsplit one's currents to 1e-9 of their scale and its densities and ions to 1e-6 (seen:
    1e-14 to 6e-11 and 1e-13 to 2e-9): only the first step's first guess differs."""
    case, batch, tvg, full = unsplit
    assert full.converged.all() and full.accepted.all()
    k = len(tvg.t) // 2
    first = solve_transient(batch, _steps(tvg, slice(0, k + 1)))
    rest = _steps(tvg, slice(k + 1, None))
    scale = np.max(np.abs(full.J_ext))
    for start in (first.final, full.state(k)):
        r = solve_transient(batch, rest, initial=start, profiles=True)
        assert r.converged.all(), case
        np.testing.assert_allclose(r.J_ext[0], full.J_ext[0, k + 1 :], rtol=0, atol=1e-9 * scale)
        for name in ("n", "p", "anion", "cation"):
            np.testing.assert_allclose(r.profiles[name][0], full.profiles[name][0, k + 1 :],
                                       rtol=1e-6, err_msg=f"{case}: {name}")  # fmt: skip
        np.testing.assert_allclose(r.profiles["V"][0], full.profiles["V"][0, k + 1 :],
                                   rtol=0, atol=1e-9)  # fmt: skip
        np.testing.assert_array_equal(r.t[0], tvg.t[k + 1 :])
    # The last state kept is the last step's (all are kept here), computed apart.
    last = full.state(-1)
    for name in ("V", "n", "p", "anion", "cation", "f_bulk", "f_interface", "V_int", "dti", "t"):
        np.testing.assert_allclose(getattr(full.final, name), getattr(last, name), rtol=1e-12,
                                   atol=1e-15, err_msg=name)  # fmt: skip


def test_the_profiles_ions_hold_each_species_total(unsplit) -> None:
    """The ions in the profiles (`anion`, `cation`, 0 outside their regions) count, at every
    step, each region's total as the solver counts it (`solver._ion_number`, trapezoidal as
    SIMsalabim's `Rescale_Ion_Density`), to 1e-9."""
    case, batch, _, full = unsplit
    ion = jax.tree_util.tree_map(lambda a: np.asarray(a[0]), batch["ion"])
    for sp, name in (("a", "anion"), ("c", "cation")):
        masks, totals = ion[f"{sp}_slots"], ion[f"{sp}_totals"]
        dens = full.profiles[name][0]
        assert np.all(dens[:, np.sum(masks, 0) == 0] == 0), case
        if not np.any(totals > 0):
            assert not dens.any(), case
            continue
        for step in dens:
            number = np.asarray(_ion_number(np.broadcast_to(step, masks.shape), masks, ion))
            has = totals > 0
            np.testing.assert_allclose(number[has], totals[has], rtol=1e-9, err_msg=case)


def test_a_state_is_checked_against_the_batch_and_its_steps() -> None:
    """`initial` refused before any solve: a first time not after the state's, a state of
    another grid, densities that are not positive."""
    setup = read_setup(DATA / "zimt_photo_light" / "simulation_setup.txt")
    tvg = read_tvg(setup)
    batch = prepare([build_device(setup, grid="simss")])
    points = batch["x"].shape[-1]
    ones = np.ones((1, points))
    good = TransientState(t=np.array([0.0]), V=0 * ones, n=1e10 * ones, p=1e10 * ones,
                          anion=0 * ones, cation=0 * ones, V_int=np.zeros(1), J_ext=np.zeros(1),
                          dti=np.zeros(1), f_bulk=np.zeros(batch["s"]["Nt"].shape),
                          f_interface=np.zeros(1))  # fmt: skip
    later = _steps(tvg, slice(1, None))
    with pytest.raises(ValueError, match="starts after it"):
        solve_transient(batch, later, initial=dataclasses.replace(good, t=np.array([1.0])))
    with pytest.raises(ValueError, match=r"initial\.n"):
        solve_transient(batch, later, initial=dataclasses.replace(good, n=np.ones((1, 3))))
    with pytest.raises(ValueError, match="positive"):
        solve_transient(batch, later, initial=dataclasses.replace(good, p=0 * ones))
    with pytest.raises(ValueError, match="profiles=True"):
        solve_transient(batch, _steps(tvg, slice(0, 3))).state(1)
