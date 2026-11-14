"""The batched least squares (`jaxdd.fit`) on small problems, against scipy's."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.fit import least_squares_many, residuals_and_jacobian

jax.config.update("jax_enable_x64", True)

T = jnp.linspace(0.0, 4.0, 40)
TRUE = jnp.array([2.0, 1.3, 0.4, 0.7])  # a·exp(−b·t) + c·sin(d·t)
DATA = TRUE[0] * jnp.exp(-TRUE[1] * T) + TRUE[2] * jnp.sin(TRUE[3] * T)


def model(theta):
    """starts × 4 → starts × 40: a decay and an oscillation, several local minima in `d`."""
    a, b, c, d = (theta[:, i : i + 1] for i in range(4))
    return a * jnp.exp(-b * T) + c * jnp.sin(d * T) - DATA


def test_the_jacobian_is_each_starts_own() -> None:
    """`residuals_and_jacobian`: the residuals, and each start's Jacobian as `jax.jacfwd` of its
    own residuals."""
    theta = jnp.asarray(np.random.default_rng(0).uniform(0.1, 3.0, (5, 4)))
    r, J = residuals_and_jacobian(model)(theta)
    assert np.allclose(r, model(theta))
    for s in range(5):
        one = jax.jacfwd(lambda t, s=s: model(t[None])[0])(theta[s])
        assert np.allclose(J[s], one, rtol=1e-13, atol=1e-13)


def _alone(start, lo, hi):
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    return least_squares(lambda t: np.asarray(model(jnp.asarray(t)[None])[0]), start,
                         jac=lambda t: np.asarray(jax.jacfwd(lambda u: model(u[None])[0])(
                             jnp.asarray(t))), bounds=(lo, hi), **KW)  # fmt: skip


KW = dict(method="trf", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)


def test_every_start_is_scipys_alone() -> None:
    """From 16 starts in the box, each start's path is scipy's `least_squares` from it alone: the
    same end, the same number of evaluations; all of them took one batched evaluation per
    iteration of the slowest; the best is the true parameters (the problem has several minima
    in `d`)."""
    lo, hi = np.array([0.1, 0.1, 0.0, 0.1]), np.array([5.0, 5.0, 2.0, 3.0])
    starts = np.random.default_rng(1).uniform(lo, hi, (16, 4))
    res = least_squares_many(model, starts, lo, hi, **KW)
    for s in range(16):
        ref = _alone(starts[s], lo, hi)
        got = res.results[s]
        assert got.nfev == ref.nfev and got.njev == ref.njev, s
        assert np.allclose(got.x, ref.x, rtol=1e-9, atol=1e-12), s
    assert res.calls == max(r.nfev for r in res.results)
    assert np.allclose(res.theta[res.best], TRUE, atol=1e-7)


def test_a_minimum_beyond_a_bound_ends_on_it() -> None:
    """With the box's upper bound of `a` below the truth, every start ends on that bound."""
    lo, hi = np.array([0.1, 0.1, 0.0, 0.5]), np.array([1.5, 5.0, 2.0, 1.0])
    starts = np.random.default_rng(2).uniform(lo, hi, (6, 4))
    res = least_squares_many(model, starts, lo, hi, **KW)
    assert all(r.active_mask[0] == 1 for r in res.results)
    assert np.allclose(res.theta[:, 0], 1.5)
    assert np.all(np.abs(res.cost - _alone(starts[0], lo, hi).cost) <= 1e-8 * res.cost)
