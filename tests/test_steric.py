"""Mobile ions bounded by their sites (`N_max_anion`, `N_max_cation`; docs/model.md, "Ions")."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd import solver as S
from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device as _build_device
from jaxdd.simss_input import setup_from_texts
from jaxdd.solver import occupancy, solve_jv

DATA = Path(__file__).parent / "data"
FREE = "steady_1e17_thin_light"  # 1e23 m⁻³ of each species, free at every voltage
FROZEN = "frozen_1e17_x1_thin_light"  # the same frozen after a pre-bias at 0 V
V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.1), 6)


def build_device(case: str, n_max: str | None = None, cation: str | None = None):
    """The case on SimSS's grid, its absorber's ions bounded by `n_max` sites (the cations by
    `cation` if given)."""
    texts = {f.name: f.read_text() for f in (DATA / case).iterdir() if f.suffix == ".txt"}
    if n_max is not None:
        texts["L2_parameters.txt"] += (f"N_max_anion = {n_max}\n"
                                       f"N_max_cation = {cation or n_max}\n")  # fmt: skip
    return _build_device(setup_from_texts(texts), grid="simss")


def steady_state(dev, Vint: float):
    """The converged state at `Vint` with the ions free: `(y, logs, s, ion)`."""
    s = jax.tree_util.tree_map(jnp.asarray, S._stencils(dev))
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in S._ion_setup(dev).items()}  # fmt: skip
    bounds = jnp.array([
        [dev.V0 + 0.5 * dev.left_elec * Vint, np.log(dev.n_left), np.log(dev.p_left)],
        [dev.VL - 0.5 * dev.left_elec * Vint, np.log(dev.n_right), np.log(dev.p_right)],
    ])  # fmt: skip
    y = jnp.asarray(S._initial(dev, Vint)[1:-1])
    for _ in range(200):
        y, size = S._poisson_step(y, bounds, s, ion, dev.Vt)
        if float(size) < 1e-8:
            break
    logs = S._ion_logs(jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]]), ion)
    for _ in range(100):
        y, logs, size = S._newton_step(y, logs, bounds, s, ion, dev.Vt, dev.G_frac)
        if float(size) < 1e-11:
            return y, logs, s, ion, bounds
    raise AssertionError("no convergence")


def test_the_occupancy_is_boltzmann_without_sites_and_bounded_with_them() -> None:
    """`occupancy`: `exp(z)` to the bit where `inv` is 0; below `N_max`, rising, `exp(z)` far
    below it and `N_max` far above, its derivative `n·(1 − n/N_max)` (as `N_max/(4·cosh²(u/2))`,
    `u = z − ln N_max`, without the cancellation) and autodiff's alike, finite where `exp(z)`
    overflows."""
    z = jnp.linspace(-50.0, 800.0, 2001)
    n, d = occupancy(z, jnp.zeros_like(z))
    assert np.array_equal(np.asarray(n), np.asarray(jnp.exp(z)))
    assert np.array_equal(np.asarray(d), np.asarray(jnp.exp(z)))
    N = 1e26
    n, d = (np.asarray(a) for a in occupancy(z, jnp.full_like(z, 1 / N)))
    assert np.isfinite(n).all() and np.isfinite(d).all()
    assert np.all(n < N * (1 + 1e-15)) and np.all(np.diff(n) >= 0)
    low = np.asarray(z) < np.log(N) - 30
    assert np.allclose(n[low], np.exp(np.asarray(z)[low]), rtol=1e-12)
    assert n[-1] == pytest.approx(N, rel=1e-15)
    with np.errstate(over="ignore"):
        exact = N / (4 * np.cosh((np.asarray(z) - np.log(N)) / 2) ** 2)
    assert np.allclose(d, exact, rtol=1e-12, atol=1e-250)  # (d underflows below 1e-250)
    grad = jax.vmap(jax.grad(lambda x: occupancy(x, 1 / N)[0]))(z)
    assert np.allclose(np.asarray(grad), d, rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("case", [FREE, FROZEN])
def test_unbounded_sites_are_boltzmanns_curve(case: str) -> None:
    """Sites far above every density (1e35 m⁻³) leave the curve Boltzmann's to round-off, free
    and frozen; infinitely many, as by default, to the bit."""
    V_ = V[::3]
    ref = solve_jv(build_device(case), V_)
    inf = solve_jv(build_device(case, "inf"), V_)
    many = solve_jv(build_device(case, "1e35"), V_)
    assert ref.converged.all() and many.converged.all()
    assert np.array_equal(inf.J, ref.J)
    assert np.max(np.abs(many.J - ref.J)) < 1e-10 * np.max(np.abs(ref.J))


@pytest.mark.parametrize(("n_max", "Vint"), [("2e24", 0.0), ("1e25", 0.0), ("1e24", 0.4)])
def test_bounded_ions_fill_their_sites_and_keep_their_number(n_max: str, Vint: float) -> None:
    """The steady state with 10–100× the mean density in sites: no density above them, the
    ions' electrochemical potential `ln(n/(N_max − n)) ∓ V/V_T` the same at every point of
    their layer (where `N_max − n` keeps its digits), their number N·L; where Boltzmann's would
    pile them up beyond the sites."""
    dev = build_device(FREE, n_max)
    N = float(n_max)
    y, logs, s, ion, bounds = steady_state(dev, Vint)
    la, lc = S._point_logs(logs, ion)
    a, c = (np.asarray(x) for x in S._ion_densities(y[:, 0], la, lc, s, dev.Vt))
    on = np.asarray(s["a_mask"]) > 0
    V_ = np.asarray(y[:, 0])[on]
    assert a.max() < N and c.max() < N and a[on].max() > 0.5 * N
    for dens, sign in ((a[on], 1.0), (c[on], -1.0)):
        mu = np.log(dens / (N - dens)) - sign * V_ / dev.Vt
        assert np.ptp(mu[dens < 0.99 * N]) < 1e-9
    V_full = jnp.concatenate([bounds[0][:1], y[:, 0], bounds[1][:1]])
    assert np.allclose(np.asarray(S._ion_constraints(V_full, logs, ion)), 0.0, atol=1e-10)
    y0, logs0, s0, ion0, _ = steady_state(build_device(FREE), Vint)
    a0 = np.asarray(S._ion_densities(y0[:, 0], *S._point_logs(logs0, ion0), s0, dev.Vt)[0])
    assert a0.max() > N


def test_the_start_puts_the_bounded_ions_number_into_their_layer() -> None:
    """`_ion_logs`: at any potential the normalisations that hold N·L, bounded or not."""
    dev = build_device(FREE, "3e23", cation="1.5e24")
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in S._ion_setup(dev).items()}  # fmt: skip
    rng = np.random.default_rng(0)
    for scale in (0.05, 0.3, 1.0):
        V_full = jnp.asarray(rng.normal(0.0, scale, len(dev.x)))
        g = S._ion_constraints(V_full, S._ion_logs(V_full, ion), ion)
        assert np.allclose(np.asarray(g), 0.0, atol=1e-12)


def test_the_jacobian_with_bounded_ions_is_autodiff_of_the_rows() -> None:
    """`_assemble`'s blocks and ion columns with bounded ions are `_kernels`' (`jacfwd` of each
    row) to round-off, at a perturbed state with ions near and far into saturation."""
    from jaxdd.solver import BERNOULLI, _assemble, _ion_setup, _kernels, _stencils

    dev = build_device(FREE, "5e23", cation="2e24")
    s = {k: jnp.asarray(v) for k, v in _stencils(dev).items()}
    rng = np.random.default_rng(1)
    y0 = S._initial(dev, 0.3)
    NP = len(dev.x) - 2
    noise = np.zeros_like(y0)
    noise[1:-1] = rng.normal(0, 1, (NP, 3)) * np.array([5 * dev.Vt, 2.0, 2.0])
    full = jnp.asarray(y0 + noise)
    ion = {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
           for k, v in _ion_setup(dev).items()}  # fmt: skip
    la, lc = S._point_logs(S._ion_logs(full[:, 0], ion) + 3.0, ion)
    a = np.asarray(S._ion_densities(full[1:-1, 0], la, lc, s, dev.Vt)[0])
    assert (a > 0.99 * 5e23).any() and (a[a > 0] < 0.5 * 5e23).any()
    B = BERNOULLI["exact"]
    row_val, row_jac, row_ions = _kernels(B)
    args = (full[:-2], full[1:-1], full[2:], s, dev.Vt, dev.G_frac, la, lc)
    ref = (row_val(*args), *row_jac(*args), jnp.stack(row_ions(*args), -1))
    got = _assemble(full, s, dev.Vt, dev.G_frac, la, lc, B)
    for name, r, g in zip(("F", "A", "B", "C", "U"), ref, got, strict=True):
        r, g = np.asarray(r), np.asarray(g)
        scale = np.max(np.abs(r), axis=tuple(range(1, r.ndim)), keepdims=True)
        assert np.max(np.abs(r - g) / np.maximum(scale, 1e-300)) < 1e-11, name


def test_bounded_ions_in_a_batch_are_the_one_device_curves() -> None:
    """`solve_batch` with bounded and unbounded ions, free and frozen, in one batch: each
    device's curve as `solve_jv` gives it, every voltage converged; the bound moves them."""
    devices = [build_device(FREE), build_device(FREE, "2e24"), build_device(FROZEN, "2e24"),
               build_device(FROZEN, "1e24", cation="5e24")]  # fmt: skip
    J, ok, _ = solve_batch(prepare(devices), V)
    assert np.asarray(ok).all()
    curves = []
    for k, dev in enumerate(devices):
        single = solve_jv(dev, V)
        assert single.converged.all()
        scale = np.max(np.abs(single.J))
        assert np.max(np.abs(np.asarray(J[k]) - single.J)) < 1e-8 * scale
        curves.append(single.J)
    assert np.max(np.abs(curves[1] - curves[0])) > 1e-2 * np.max(np.abs(curves[0]))


def test_frozen_bounded_ions_are_the_free_ones_at_the_pre_bias() -> None:
    """Frozen after the pre-bias, bounded ions are where the free steady state puts them: the
    frozen curve at the pre-bias is the free curve's there."""
    free = solve_jv(build_device(FREE, "2e24"), np.array([0.0]))
    frozen = solve_jv(build_device(FROZEN, "2e24"), np.array([0.0]))
    assert free.converged.all() and frozen.converged.all()
    assert abs(frozen.J[0] - free.J[0]) < 1e-9 * abs(free.J[0])


def test_the_curve_is_differentiable_in_the_sites() -> None:
    """`d J / d ln N_max` through `device_tree` and the converged solutions is the central
    difference's."""
    from jaxdd.batch import device_tree
    from jaxdd.device import device_parameters

    texts = {f.name: f.read_text() for f in (DATA / FROZEN).iterdir() if f.suffix == ".txt"}
    texts["L2_parameters.txt"] += "N_max_anion = 2e24\nN_max_cation = 2e24\n"
    structure, base = device_parameters(setup_from_texts(texts), grid="simss")
    base = {k: jnp.asarray(v) for k, v in base.items()}
    Vs = jnp.asarray([0.0, 0.8, 1.1])

    def curve(log_n):
        p = dict(base, N_max_anion=base["N_max_anion"].at[1].set(jnp.exp(log_n)))
        tree = jax.tree_util.tree_map(lambda x: x[None], device_tree(structure, p))
        return solve_batch(tree, Vs)[0][0]

    x0 = jnp.log(2e24)
    grad = np.asarray(jax.jacfwd(curve)(x0))
    h = 1e-4
    fd = (np.asarray(curve(x0 + h)) - np.asarray(curve(x0 - h))) / (2 * h)
    assert np.max(np.abs(grad - fd)) < 1e-5 * np.max(np.abs(fd)) + 1e-9
    assert np.max(np.abs(fd)) > 1e-3


def test_bounded_mobile_ions_in_time_are_refused() -> None:
    """The steric limit is the steady state's: a transient with bounded mobile ions is refused
    by name."""
    from jaxdd.simss_input import read_tvg
    from jaxdd.transient import solve_transient

    case = "zimt_ions_fast_light"
    texts = {f.name: f.read_text() for f in (DATA / case).iterdir() if f.suffix == ".txt"}
    for name in [k for k in texts if k.startswith("L") and k.endswith("_parameters.txt")]:
        texts[name] += "N_max_anion = 1e30\nN_max_cation = 1e30\n"
    setup = setup_from_texts(texts)
    tvg = read_tvg(setup)
    with pytest.raises(NotImplementedError, match="N_max"):
        solve_transient(prepare([_build_device(setup, grid="simss")]), tvg)


def test_ions_must_fit_into_their_sites() -> None:
    """A mean density at or above its sites is refused, as a site density not positive."""
    with pytest.raises(ValueError, match="below its sites"):
        build_device(FREE, "1e23")
    with pytest.raises(ValueError, match="positive"):
        build_device(FREE, "0")


def test_weakly_bounded_ions_converge_from_boltzmanns_solution() -> None:
    """A device of `validation/hard.py` (no. 258: 1.7e24 m⁻³ of ions frozen at full
    accumulation in 500 nm, a doped 200 nm HTL) whose ions Boltzmann's piles to 5e24 m⁻³, a
    thousandth of 1.21e28 sites: from Poisson's start Newton passed states far beyond the sites
    and cycled there; from Boltzmann's solution it converges, and the curve is Boltzmann's but
    for the bound's thousandth."""
    import re

    texts = {f.name: f.read_text() for f in (DATA / FROZEN).iterdir() if f.suffix == ".txt"}

    def put(name, key, value):
        texts[name], n = re.subn(rf"^{key} = \S+", f"{key} = {value}", texts[name], flags=re.M)
        assert n == 1, key

    put("L2_parameters.txt", "L", "5e-07")
    put("L2_parameters.txt", "N_anion", "1.69845e+24")
    put("L2_parameters.txt", "N_cation", "1.69845e+24")
    put("L2_parameters.txt", "N_t_bulk", "6.10678e+21")
    put("L1_parameters.txt", "L", "2e-07")
    put("L3_parameters.txt", "L", "2e-07")
    put("L3_parameters.txt", "N_A", "8.90504e+23")
    free = _build_device(setup_from_texts(texts))
    texts["L2_parameters.txt"] += "N_max_anion = 1.21e28\nN_max_cation = 1.21e28\n"
    bounded = _build_device(setup_from_texts(texts))
    J, ok, _ = solve_batch(prepare([free, bounded]), V)
    J, ok = np.asarray(J), np.asarray(ok)
    assert ok.all()
    assert np.max(np.abs(J[1] - J[0])) < 1e-2 * np.max(np.abs(J[0]))
    single = solve_jv(bounded, V[::5])
    assert single.converged.all()
    assert np.max(np.abs(single.J - J[1, ::5])) < 1e-8 * np.max(np.abs(J[1]))
