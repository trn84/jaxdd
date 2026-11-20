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


@pytest.mark.parametrize("statistics", ["fermi-dirac", "blakemore"])
def test_the_jacobian_with_a_statistics_is_its_residuals(statistics) -> None:
    """`_assemble`'s hand-written blocks (the fluxes' `x` with the excess potentials, a blocking
    contact's density by Newton) against autodiff of its residual, row by row to 1e-12 of each
    row's scale, at a perturbed state of the cell with degenerate transport layers."""
    setup = stack_setup([HTL, {}], W_L=5.0, W_R=5.4, S=(-1, -1, 0, 0), NP=200)
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


def test_what_another_statistics_refuses() -> None:
    """At a free end: refused by name; an unknown statistics too."""
    with pytest.raises(NotImplementedError, match="statistics"):
        build_device(stack_setup([HTL, {}], W_L=5.0), statistics="blakemore")
    with pytest.raises(ValueError, match="statistics"):
        build_device(stack_setup([HTL, {}], W_L=5.0, W_R=5.4), statistics="gauss")
