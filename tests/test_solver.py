"""jaxdd against SimSS, and the properties of its pieces."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.device import build_device, make_grid
from jaxdd.simss_input import check_supported, read_setup
from jaxdd.solver import bernoulli, bernoulli_simss, solve_jv

DATA = Path(__file__).parent / "data"


def test_bernoulli_is_finite_with_a_finite_derivative_at_zero() -> None:
    """The double where: neither B nor dB/dx sees 0/0."""
    assert float(bernoulli(0.0)) == 1.0
    assert float(jax.grad(bernoulli)(0.0)) == pytest.approx(-0.5)


@pytest.mark.parametrize("x", [-30.0, -1.0, -1e-3, 1e-3, 0.3, 2.0, 30.0])
def test_bernoulli_identity(x: float) -> None:
    """Invariant: B(−x) = B(x) + x, the identity behind Scharfetter–Gummel's symmetry."""
    assert float(bernoulli(-x)) == pytest.approx(float(bernoulli(x)) + x, rel=1e-12, abs=1e-14)


def test_simss_bernoulli_matches_the_exact_one_to_its_own_error() -> None:
    x = jnp.linspace(-5.0, 5.0, 2001)
    assert float(jnp.max(jnp.abs(bernoulli_simss(x) / bernoulli(x) - 1))) < 1e-3


def test_grid_spacing_sums_to_the_device_but_for_the_interfaces() -> None:
    """Invariant: one point more than NP, monotone, and Σh within 1e-3 of 1 (SIMsalabim
    equalises the spacing across each interface, which moves the sum slightly)."""
    x, h, i0, i1, _ = make_grid([1e-9, 500e-9, 1e-9], 400, 4.0)
    assert len(x) == 402 and np.all(np.diff(x) > 0)
    assert abs(h[:-1].sum() - 1.0) < 1e-3
    assert i0 == [0, 6, 397] and i1 == [5, 396, 401]


def test_jv_reproduces_simss() -> None:
    """The discrete equations are SimSS's: with its Bernoulli approximation, its curve to its
    own convergence tolerance at every voltage."""
    case = DATA / "thin_l1_light"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < 1e-5 * scale


def test_short_circuit_current_never_exceeds_the_generation() -> None:
    """Invariant: no device collects more than it generates (the generation's current)."""
    case = DATA / "thin_l1_light"
    dev = build_device(read_setup(case / "simulation_setup.txt"))
    res = solve_jv(dev, np.array([0.0]))
    from jaxdd.constants import Q

    generated = Q * np.sum(dev.h[:-1] * dev.G[:-1]) / np.sum(dev.h[:-1]) * dev.L
    assert -res.J[0] <= generated * (1 + 1e-9)


def test_unsupported_input_is_refused_by_name(tmp_path) -> None:
    case = DATA / "thin_l1_light"
    for f in case.iterdir():
        (tmp_path / f.name).write_text(f.read_text())
    setup_file = tmp_path / "simulation_setup.txt"
    setup_file.write_text(setup_file.read_text().replace("R_series = 0", "R_series = 1E-4", 1))
    with pytest.raises(NotImplementedError, match="R_series"):
        check_supported(read_setup(setup_file))
