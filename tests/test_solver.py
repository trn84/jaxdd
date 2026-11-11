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


#: Stacks (m) whose share of the grid is a whole number in exact arithmetic, with the absorber's
#: points as SimSS 5.36 gave them (its `Var.dat`, column `lid`; 2026-11-11), and exactly.
SPLIT_EDGES = [
    (("7e-10", "3.5e-07", "7e-10"), 389, 390),
    (("1.4e-09", "7e-07", "1.4e-09"), 389, 390),
    (("1e-07", "5e-07", "2e-07"), 250, 250),
    (("1e-06", "7e-07", "5e-07"), 127, 127),
]


@pytest.mark.parametrize(("stack", "simss", "exact"), SPLIT_EDGES)
def test_grid_split_on_a_whole_share(stack: tuple[str, ...], simss: int, exact: int) -> None:
    """Where FLOOR meets a whole number, "simss" lands where SimSS's 10-byte reals did and
    "exact" on the whole number; both give every point."""
    for split, absorber in (("simss", simss), ("exact", exact)):
        _, _, i0, i1, lid = make_grid(stack, 400, 4.0, split)
        assert i1[1] - i0[1] == absorber
        assert i1[-1] == 401 and np.all(np.diff(lid) >= 0)


def test_grid_split_first_layer_on_a_whole_share() -> None:
    """SimSS gave the 100 nm layer of 100/500/200 nm 49 points beyond its first, not 50."""
    assert make_grid(("1e-07", "5e-07", "2e-07"), 400, 4.0, "simss")[3][0] == 49
    assert make_grid(("1e-07", "5e-07", "2e-07"), 400, 4.0, "exact")[3][0] == 50


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


@pytest.mark.parametrize("case", ["frozen_1e17_x1_thin_light", "steady_1e17_thin_light"])
def test_jv_with_ions_reproduces_simss(case: str) -> None:
    """Frozen after a pre-bias, and free at every voltage: SimSS's curve to 2e-5 of its scale."""
    setup = read_setup(DATA / case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(DATA / case / "JV.dat", names=True)
    res = solve_jv(build_device(setup), ref["Vint"], bernoulli_name="simss")
    assert res.converged.all()
    scale = np.max(np.abs(ref["Jext"]))
    assert np.max(np.abs(res.J - ref["Jext"])) < 2e-5 * scale


def test_free_ions_keep_their_number() -> None:
    """Invariant: the steady state moves the ions, never creates them — the trapezoidal number
    in their layer is N·L at any bias."""
    import jax.numpy as jnp

    from jaxdd import solver as S

    dev = build_device(read_setup(DATA / "steady_1e17_thin_light" / "simulation_setup.txt"))
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in S._ion_setup(dev).items()}  # fmt: skip
    rng = np.random.default_rng(0)
    V = jnp.asarray(rng.normal(0.0, 0.3, len(dev.x)))
    logs = S._ion_logs(V, ion)
    g = S._ion_constraints(V, logs[0], logs[1], ion)
    assert np.allclose(np.asarray(g), 0.0, atol=1e-12)


def test_batch_matches_the_one_device_solver() -> None:
    """The batched solver (`vmap` over devices, `while_loop` per voltage) gives each device's
    curve as the one-device solver does, with and without ions, in one batch."""
    from jaxdd.batch import prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    devices = [build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    J, ok, _ = solve_batch(prepare(devices), V, bernoulli_name="simss")
    assert np.asarray(ok).all()
    for k, dev in enumerate(devices):
        single = solve_jv(dev, V, bernoulli_name="simss")
        scale = np.max(np.abs(single.J))
        assert np.max(np.abs(np.asarray(J[k]) - single.J)) < 1e-6 * scale


def test_two_passes_give_the_one_pass_result() -> None:
    """`solve_batch` re-solves the devices that missed `first_max_iter` with `max_iter`: the
    result is the one pass's, flags and Newton steps exactly, curves to round-off (a smaller
    batch compiles to other roundings); the stragglers are padded to a power of two by
    repeating the last."""
    from jaxdd.batch import _stragglers, prepare, solve_batch

    cases = ["thin_l1_light", "frozen_1e17_x1_thin_light", "steady_1e17_thin_light"]
    devices = [build_device(read_setup(DATA / c / "simulation_setup.txt")) for c in cases]
    batch = prepare(devices + devices[:2])
    V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)
    one = [np.asarray(a) for a in solve_batch(batch, V, first_max_iter=None)]
    first = np.asarray(solve_batch(batch, V, max_iter=14, first_max_iter=None)[1])
    assert 0 < (~first.all(axis=1)).sum() < len(first)  # thin_l1_light's start needs 15
    two = [np.asarray(a) for a in solve_batch(batch, V, first_max_iter=14)]
    assert one[1].all()
    assert np.array_equal(one[1], two[1]) and np.array_equal(one[2], two[2])
    assert np.max(np.abs(one[0] - two[0])) < 1e-12 * np.max(np.abs(one[0]))

    ok = np.ones((6, 2), dtype=bool)
    ok[[1, 4, 5], 0] = False
    idx, padded = _stragglers(ok)
    assert idx.tolist() == [1, 4, 5] and padded.tolist() == [1, 4, 5, 5]
    assert _stragglers(np.ones((3, 2), dtype=bool))[1].size == 0


def test_a_dark_scan_with_a_density_far_below_the_floor_converges() -> None:
    """Regression: in the dark, a thick doped transport layer holds a minority density far
    below `n_floor`, whose logarithm is undetermined between neighbouring voltages. The batched
    scan's extrapolated start once carried that into NaN at every voltage from short circuit on.
    Every voltage converges, and the dark current is what a dark current must be: monotone in
    `V` and zero at 0 V, to `CURRENT_TOL` of the generation's current; and SimSS's curve to 2e-5
    of its scale (SimSS's own early stops at −0.2 and −0.175 V lie well within that)."""
    from jaxdd.batch import CURRENT_TOL, prepare, solve_batch

    case = DATA / "frozen_thick_htl_dark"
    setup = read_setup(case / "simulation_setup.txt")
    check_supported(setup)
    ref = np.genfromtxt(case / "JV.dat", names=True)
    batch = prepare([build_device(setup)])
    J, ok, _ = solve_batch(batch, ref["Vint"], bernoulli_name="simss")
    J, scale = np.asarray(J[0]), float(batch["J_scale"][0])
    assert np.asarray(ok).all()
    assert np.all(np.diff(J) > -CURRENT_TOL * scale)
    assert abs(J[np.argmin(np.abs(ref["Vint"]))]) < CURRENT_TOL * scale
    assert np.max(np.abs(J - ref["Jext"])) < 2e-5 * np.max(np.abs(ref["Jext"]))


def test_the_block_sweep_solves_the_block_tridiagonal_system() -> None:
    """Invariant: the sweep gives the dense solve's solution to round-off, with several
    right-hand sides."""
    from jaxdd.solver import _block_thomas

    rng = np.random.default_rng(1)
    n = 37
    A, C = rng.normal(size=(n, 3, 3)), rng.normal(size=(n, 3, 3))
    B = rng.normal(size=(n, 3, 3)) + 8 * np.eye(3)
    A[0], C[-1] = 0.0, 0.0
    d = rng.normal(size=(n, 3, 2))
    full = np.zeros((3 * n, 3 * n))
    for i in range(n):
        full[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = B[i]
        if i > 0:
            full[3 * i : 3 * i + 3, 3 * i - 3 : 3 * i] = A[i]
        if i < n - 1:
            full[3 * i : 3 * i + 3, 3 * i + 3 : 3 * i + 6] = C[i]
    ref = np.linalg.solve(full, d.reshape(3 * n, 2))
    x = np.asarray(_block_thomas(*map(jnp.asarray, (A, B, C, d))))
    assert np.max(np.abs(x.reshape(3 * n, 2) - ref)) < 1e-12
