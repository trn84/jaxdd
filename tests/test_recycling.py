"""Photon recycling (jaxdd's, `jaxdd.recycling`, `solve_transient(recycle=...)`): the slab's
optics conserve the photons, mirror with the slab, and the recycled generation enters each time
step as the explicit term it is."""

import numpy as np
import pytest

from jaxdd.batch import prepare
from jaxdd.recycling import device_kernel, fresnel, slab_kernel
from jaxdd.transient import solve_transient, tj_columns
from test_free_surface import _decay, stack


def _cells(d, n):
    e = np.linspace(0.0, d, n + 1)
    return 0.5 * (e[1:] + e[:-1]), np.stack([e[:-1], e[1:]], -1)


def test_every_photon_is_absorbed_or_leaves() -> None:
    """Each point's photons: absorbed in the slab's cells or leaving at either face, to 1e-12;
    the slab between equal media its own mirror image; most of a 600 nm perovskite's light
    reabsorbed (its escape cones are narrow)."""
    d = 600e-9
    z, edges = _cells(d, 60)
    P, top, bottom = slab_kernel(z, edges, d)
    np.testing.assert_allclose(P.sum(0) + top + bottom, 1.0, atol=1e-12)
    assert (P >= 0).all() and 0.9 < P.sum(0).mean() < 0.99
    assert bottom[0] > bottom[-1] and top[-1] > top[0]  # near a face more of it leaves there
    P, top, bottom = slab_kernel(z, edges, d, n_top=1.5, n_bottom=1.5)
    np.testing.assert_allclose(top, bottom[::-1], rtol=1e-12)
    np.testing.assert_allclose(P, P[::-1, ::-1], rtol=1e-10, atol=1e-16)


def test_light_beyond_the_escape_cone_is_reflected() -> None:
    """Fresnel's reflectance: at normal incidence ((n₁ − n₂)/(n₁ + n₂))², 1 beyond the critical
    angle."""
    assert float(fresnel(1.0, 2.6, 1.0)) == pytest.approx((1.6 / 3.6) ** 2)
    assert float(fresnel(np.cos(np.arcsin(1 / 2.6) + 1e-6), 2.6, 1.0)) == 1.0


def test_a_kernel_on_a_devices_grid() -> None:
    """`device_kernel` on a stack: the absorber's points only (its rows and columns), each
    column's photons with the faces' shares summing to 1."""
    d = stack([{"L": "30e-9", "k_direct": 0, "N_t_bulk": 0}, {"N_t_bulk": 0}], NP=120)
    K, left, right = device_kernel(d, n_mu=48)
    on = np.asarray(d.k_direct)[1:-1] > 0
    assert K.shape == (120, 120) and not K[~on].any() and not K[:, ~on].any()
    np.testing.assert_allclose(K[:, on].sum(0) + left + right, 1.0, atol=1e-12)


def test_a_free_film_decays_with_its_photons_recycled() -> None:
    """The free film of `test_free_surface`, direct recombination alone, each point reabsorbing
    a share `c` of its own photons: each step of implicit Euler with the recycled term from the
    step before, `n − n₀ = −Δt·k·(n² − n_i²) + c·Δt·k·(n₀² − n_i²)`, in closed form to 1e-9; a
    zero matrix the decay without it, to the bit."""
    d = stack([{"N_t_bulk": 0}])
    dt, k, G, c = 1e-7, 4.7281398e-17, 1.5e28, 0.8
    batch = prepare([d])
    NP = len(d.x) - 2
    r = solve_transient(batch, _decay(30, dt), recycle=c * np.eye(NP))
    assert r.converged.all()
    ni = d.ni[1]
    n = [np.sqrt(G / k + ni**2)]
    for _ in range(30):
        a = dt * k
        rhs = n[-1] + a * ni**2 + c * a * (n[-1] ** 2 - ni**2)
        n.append((-1 + np.sqrt(1 + 4 * a * rhs)) / (2 * a))
    QFLS = tj_columns(r, 0)["QFLSL1"]
    np.testing.assert_allclose(ni * np.exp(QFLS / (2 * d.Vt)), n, rtol=1e-9)
    plain = solve_transient(batch, _decay(30, dt))
    zero = solve_transient(batch, _decay(30, dt), recycle=np.zeros((NP, NP)))
    for key in ("QFLSL1", "JdirL1"):
        np.testing.assert_array_equal(tj_columns(zero, 0)[key], tj_columns(plain, 0)[key])
    with pytest.raises(ValueError, match="recycle"):
        solve_transient(batch, _decay(30, dt), recycle=np.zeros((NP + 1, NP + 1)))
