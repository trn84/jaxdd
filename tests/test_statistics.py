"""The carriers' statistics (`jaxdd.statistics`, docs/model.md "The carriers' statistics"): the
functions themselves, the Jacobian with them, and what they refuse. Against ChargeTransport.jl:
`tests/test_chargetransport.py`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

import jaxdd.solver as sv
from jaxdd import statistics as st
from jaxdd.device import build_device
from test_chargetransport import device
from test_free_surface import HTL, stack_setup


def _fermi_dirac_half(eta):
    """The Fermi–Dirac integral of order 1/2, normalised to exp(η) far below the band."""
    f = integrate.quad(lambda x: np.sqrt(x) / (1 + np.exp(x - eta)), 0, max(50.0, eta + 50.0),
                       limit=200)[0]  # fmt: skip
    return 2 / np.sqrt(np.pi) * f


def test_the_statistics_functions() -> None:
    """TeSCA's `F_{1/2}` within 1 % of the integral's from η = −40 to 30; its inverse back to
    1e-12; `w`'s derivative (its own rule) the finite difference's; Blakemore's in closed form
    and below its bound; Boltzmann's `w` 0."""
    eta = np.linspace(-40.0, 30.0, 71)
    exact = np.array([_fermi_dirac_half(e) for e in eta])
    tesca = np.asarray(st.density(jnp.asarray(eta), "fermi-dirac"))
    assert np.max(np.abs(tesca / exact - 1)) < 1e-2
    ln_u = jnp.log(jnp.asarray(tesca))
    assert np.max(np.abs(np.asarray(ln_u + st.excess(ln_u, "fermi-dirac")) - eta)) < 1e-12
    grad = jax.grad(lambda a: jnp.sum(st.excess(a, "fermi-dirac")))(ln_u)
    h = 1e-6
    fd = (st.excess(ln_u + h, "fermi-dirac") - st.excess(ln_u - h, "fermi-dirac")) / (2 * h)
    np.testing.assert_allclose(grad, fd, atol=1e-7)
    eta_b = np.linspace(-30.0, 20.0, 51)
    u = np.asarray(st.density(jnp.asarray(eta_b), "blakemore"))
    assert np.all(u < 1 / st.BLAKEMORE_GAMMA)
    back = np.log(u) + np.asarray(st.excess(jnp.log(jnp.asarray(u)), "blakemore"))
    np.testing.assert_allclose(back, eta_b, atol=1e-9)
    assert np.all(np.asarray(st.excess(ln_u, "boltzmann")) == 0)
    np.testing.assert_allclose(st.ln_inverse(np.log(u), "blakemore", np), eta_b, atol=1e-9)


#: A degenerate HTL (`N_A` 1.5e24 against `N_c` 2.2e24) with charged interface traps towards
#: the absorber.
DEGENERATE_HTL = dict(HTL, N_A="1.5e24", N_t_int="1e16")
#: The cases of `test_the_jacobian_with_a_statistics_is_its_residuals`: a blocking contact, the
#: degenerate HTL's interface traps, a free end with its surface's recombination.
JACOBIAN_CASES = {
    "contact": dict(layers=[HTL, {}], W_L=5.0, W_R=5.4, S=(-1, -1, 0, 0)),
    "interface traps": dict(layers=[DEGENERATE_HTL, {}], W_L="sfb", W_R=4.275),
    "free end": dict(layers=[{"N_A": "1.5e24"}], W_L="sfb", W_R="free", S=(-1, -1, 1e2, 1e2)),
}


@pytest.mark.parametrize("case", list(JACOBIAN_CASES))
@pytest.mark.parametrize("statistics", ["fermi-dirac", "blakemore"])
def test_the_jacobian_with_a_statistics_is_its_residuals(statistics, case) -> None:
    """`_assemble`'s hand-written blocks (the fluxes' `x` with the excess potentials, a blocking
    contact's density by Newton; interface traps and a free surface by autodiff of their
    degenerate rates) against autodiff of its residual, row by row to 1e-12 of each row's
    scale, at a perturbed state of a device with a degenerate layer."""
    c = JACOBIAN_CASES[case]
    setup = stack_setup(c["layers"], W_L=c["W_L"], W_R=c["W_R"], S=c.get("S", (-1, -1, -1, -1)),
                        NP=200)  # fmt: skip
    d = build_device(setup, statistics=statistics)
    s = jax.tree_util.tree_map(jnp.asarray, sv._stencils(d))
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in sv._ion_setup(d).items()}  # fmt: skip
    rng = np.random.default_rng(0)
    y0 = np.asarray(sv._initial(d, -0.2, 0.0))
    y = jnp.asarray(y0[1:-1] + rng.normal(scale=[0.01, 0.3, 0.3], size=y0[1:-1].shape))
    bounds = jnp.array(
        [
            [d.V0 - 0.5 * d.left_elec * 0.2, np.log(d.n_left), np.log(d.p_left)],
            [d.VL + 0.5 * d.left_elec * 0.2, np.log(d.n_right), np.log(d.p_right)],
        ]
    )
    logs = sv._ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion)
    la, lc = sv._point_logs(logs, ion)
    full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
    if "ct" in s:
        s = dict(s, ct=dict(s["ct"], branch=sv._contact_branch(full, s, d.Vt, sv.bernoulli)))

    def residual(yy):
        return sv._assemble(jnp.concatenate([bounds[0][None], yy, bounds[1][None]]), s, d.Vt,
                            0.0, la, lc)[0]  # fmt: skip

    _, A, B, C, _ = sv._assemble(full, s, d.Vt, 0.0, la, lc)
    jac = np.asarray(jax.jacfwd(residual)(y))
    i = np.arange(y.shape[0])
    for ours, auto in ((B, jac[i, :, i, :]), (np.asarray(A)[1:], jac[i[1:], :, i[:-1], :]),
                       (np.asarray(C)[:-1], jac[i[:-1], :, i[1:], :])):  # fmt: skip
        ours = np.asarray(ours)
        scale = np.max(np.abs(auto), axis=2, keepdims=True) + 1e-300
        assert np.max(np.abs(ours - auto) / scale) < 1e-12


def test_the_default_statistics_is_boltzmanns() -> None:
    """Boltzmann's, given or by default, is the same device and stencils (no `stat`)."""
    a = device(1e24, 1e25, 0.0)
    b = device(1e24, 1e25, 0.0, statistics="boltzmann")
    assert a.statistics == b.statistics == "boltzmann"
    assert "stat" not in sv._stencils(a)
    np.testing.assert_array_equal(a.n_left, b.n_left)


def test_equilibrium_is_flat_with_interface_traps_and_free_surfaces() -> None:
    """In the dark at 0 V every rate vanishes at one Fermi level, also where a layer is
    degenerate (Fermi–Dirac's `F_{1/2}`; Blakemore's takes the same code): the degenerate HTL's
    interface traps (each point's equilibrium at its own quasi-Fermi level,
    `solver._interface_degenerate`) and a degenerate film's free surfaces (`_surface_srh`).
    Each carrier's quasi-Fermi level is flat to 1e-9 V where it is above `n_floor` (below it
    its logarithm does not converge, nor count) and no current flows. (The interface traps'
    rates themselves: `test_interface_traps_balance_in_a_degenerate_equilibrium`.)"""
    from jaxdd.batch import prepare, solve_states

    def spread(dev, res):  # each carrier's quasi-Fermi level's spread, above `n_floor`
        floor = 1e-12 * float(np.max(dev.N_c))
        V, n, p = (np.asarray(res[k][0, 0]) for k in ("V", "n", "p"))
        ln_inv = np.log(1.0 / dev.N_c)
        wn = np.asarray(st.excess(jnp.log(n) + ln_inv, dev.statistics))
        wp = np.asarray(st.excess(jnp.log(p) + ln_inv, dev.statistics))
        Efn = -dev.E_c - V + dev.Vt * (np.log(n) + ln_inv + wn)
        Efp = -dev.E_v - V - dev.Vt * (np.log(p) + ln_inv + wp)
        out = 0.0
        for E, dens in ((Efn, n), (Efp, p)):
            above = dens > floor
            if above.any():
                out = max(out, float(np.ptp(E[above])))
        return out

    devices = {
        "interface": stack_setup(
            [DEGENERATE_HTL, {}], W_L="sfb", W_R=4.275, S=(-1, -1, -1, -1), G_frac=0
        ),
        "film": stack_setup([{"N_A": "1.5e24"}], S=(1e2, 1e2, 1e2, 1e2), G_frac=0),
    }
    for name, setup in devices.items():
        dev = build_device(setup, statistics="fermi-dirac")
        batch = prepare([dev])
        res = solve_states(batch, [0.0], max_iter=200)
        assert bool(res["converged"][0, 0]), name
        assert spread(dev, res) < 1e-9, name
        if name == "interface":
            assert abs(float(res["J"][0, 0])) < 1e-9


def test_interface_traps_balance_in_a_degenerate_equilibrium() -> None:
    """At one Fermi level 0.05 eV inside the degenerate HTL's valence band (`p` 2.8 `N_v` there,
    the densities `F_{1/2}`'s), each of the interface traps' four processes, each carrier's
    capture at either point, vanishes by itself, to 1e-15 of the holes' capture: each point's
    equilibrium at its own quasi-Fermi level (`solver._interface_degenerate`). With Boltzmann's
    `n_t0`, `p_t0` the holes cross the interface through the traps, 3.9e-4 of their capture."""
    d = build_device(stack_setup([DEGENERATE_HTL, {}], W_L="sfb", W_R=4.275,
                                 S=(-1, -1, -1, -1)), statistics="fermi-dirac")  # fmt: skip
    ti = jax.tree_util.tree_map(jnp.asarray, sv._stencils(d)["ti"])
    k = int(np.argmax(np.asarray(ti["Nt"])))
    W = 5.45
    Ec, Ev, Nc = np.asarray(d.E_c), np.asarray(d.E_v), np.asarray(d.N_c)
    n = jnp.asarray(Nc * np.asarray(st.density(jnp.asarray((Ec - W) / d.Vt), "fermi-dirac")))
    p = jnp.asarray(Nc * np.asarray(st.density(jnp.asarray((W - Ev) / d.Vt), "fermi-dirac")))
    assert float(p[k] / Nc[k]) > 2
    scale = float(ti["Nt"][k] * ti["Cp"][k] * (p[k] + p[k + 1]))
    rates = sv._interface_edges(n[:-1], n[1:], p[:-1], p[1:], ti)[:4]
    assert max(abs(float(r[k])) for r in rates) < 1e-15 * scale
    boltzmann = {key: v for key, v in ti.items() if key != "stat"}
    rates = sv._interface_edges(n[:-1], n[1:], p[:-1], p[1:], boltzmann)[:4]
    assert max(abs(float(r[k])) for r in rates) > 1e-4 * scale


def test_what_another_statistics_refuses() -> None:
    """An unknown statistics."""
    with pytest.raises(ValueError, match="statistics"):
        build_device(stack_setup([HTL, {}], W_L=5.0, W_R=5.4), statistics="gauss")
