"""The transfer-matrix profile's shift of the layers' n,k in photon energy (`dE_nk`, jaxdd's)."""

import dataclasses
import functools
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdd.batch import device_tree, solve_batch
from jaxdd.device import OPTICS_PARAMETERS, build_device, device_arrays, device_parameters
from jaxdd.optics import HC_Q, shifted_indices, shifted_wavelengths
from jaxdd.simss_input import read_setup

DATA = Path(__file__).parent / "data"
CALC = ("calc_light", "pin_calc_light", "pin_anode_left_light")

#: Each layer's shift (eV) in `pin_calc_light`: C60, the perovskite, TaTm.
SHIFTS = {"L1_parameters.txt": 0.03, "L2_parameters.txt": 0.05, "L3_parameters.txt": -0.04}


def _parts(case: str):
    return device_parameters(read_setup(DATA / case / "simulation_setup.txt"))


def _leaves(tree) -> list:
    return [np.asarray(v) for v in jax.tree_util.tree_leaves(tree)]


def test_the_shift_is_a_layer_parameter_of_zero_by_default() -> None:
    """Every input's dict carries `dE_nk`, 0 on every layer, with computed generation or not."""
    for case in (*CALC, "none_light", "thin_l1_light"):
        _, p = _parts(case)
        for name in OPTICS_PARAMETERS:
            assert np.array_equal(p[name], np.zeros(len(p["L"]))), (case, name)


def test_a_zero_shift_gives_the_wavelengths_and_the_stored_indices_to_the_bit() -> None:
    """`λ' = λ` exactly at zero shift, and the layers' tables interpolated there are the stored
    indices on the 1 nm grid, in NumPy, `jax.numpy` and under `jit`."""
    for case in CALC:
        st, p = _parts(case)
        o, zero = st.optics, np.zeros(len(p["L"]))
        assert np.array_equal(shifted_wavelengths(o["lambdas"], 0.0), o["lambdas"])
        assert np.array_equal(shifted_indices(o, zero), o["n"])
        assert np.array_equal(np.asarray(shifted_indices(o, jnp.asarray(zero), jnp)), o["n"])
        jitted = jax.jit(lambda d, o=o: shifted_indices(o, d, jnp))(jnp.asarray(zero))
        assert np.array_equal(np.asarray(jitted), o["n"])


@pytest.mark.parametrize("case", CALC)
def test_a_zero_shift_gives_the_unshifted_device_to_the_bit(case: str) -> None:
    """With `dE_nk` 0 on every layer the device is that of a dict without the key (the stored
    indices on the 1 nm grid, the profile computed as without a shift), every array to the bit:
    `device_arrays` in NumPy, and `device_tree` eagerly, under `jit` and under `vmap`."""
    st, p = _parts(case)
    without = {k: v for k, v in p.items() if k not in OPTICS_PARAMETERS}
    got, ref = device_arrays(st, p), device_arrays(st, without)
    for field in dataclasses.fields(got):
        if isinstance(getattr(ref, field.name), np.ndarray):
            assert np.array_equal(getattr(got, field.name), getattr(ref, field.name)), field.name
    for build in (functools.partial(device_tree, st), jax.jit(functools.partial(device_tree, st))):
        for g, r in zip(_leaves(build(p)), _leaves(build(without)), strict=True):
            assert np.array_equal(g, r)

    def stack(q):
        return jax.tree_util.tree_map(lambda v: jnp.stack([jnp.asarray(v)] * 2), q)

    batched = jax.jit(jax.vmap(functools.partial(device_tree, st)))
    for g, r in zip(_leaves(batched(stack(p))), _leaves(batched(stack(without))), strict=True):
        assert np.array_equal(g, r)


def _shifted_copies(tmp_path: Path) -> tuple[Path, Path]:
    """`pin_calc_light` twice: its layers' n,k files shifted beforehand (each data point moved
    to the wavelength of its photon energy plus the layer's shift), and the files as they are
    with `dE_nk` named in the layer files."""
    pre, named = tmp_path / "pre", tmp_path / "named"
    shutil.copytree(DATA / "pin_calc_light", pre)
    shutil.copytree(DATA / "pin_calc_light", named)
    for layer, dE in SHIFTS.items():
        text = (pre / layer).read_text()
        nk = next(line.split("=")[1].split("*")[0].strip() for line in text.splitlines()
                  if line.startswith("nkLayer"))  # fmt: skip
        data = np.loadtxt(pre / nk, skiprows=1)
        data[:, 0] = data[:, 0] / (1 + dE * data[:, 0] / HC_Q)
        np.savetxt(pre / nk, data, header="lambda n k", comments="", fmt="%.17e")
        (named / layer).write_text((named / layer).read_text() + f"dE_nk = {dE}\n")
    return pre, named


def test_a_shifted_profile_is_that_of_n_k_files_shifted_beforehand(tmp_path: Path) -> None:
    """The three layers shifted by +0.03, +0.05 and −0.04 eV (the profile moves by 6 % of its
    peak): the profile of the unshifted files with `dE_nk` named in the layer files is the one of
    files whose wavelengths were shifted beforehand. The two interpolate linearly, one in `λ'`
    between the data's points, the other in `λ` between the shifted points; they differ by the
    curvature of `λ'(λ)` over a 1 nm interval, measured 3e-8 of the peak; to 1e-6. In NumPy
    (`build_device`) and in JAX (`device_tree`)."""
    pre, named = _shifted_copies(tmp_path)
    ref = build_device(read_setup(pre / "simulation_setup.txt")).G
    setup = read_setup(named / "simulation_setup.txt")
    st, p = device_parameters(setup)
    assert np.array_equal(p["dE_nk"], list(SHIFTS.values()))
    unshifted = build_device(read_setup(DATA / "pin_calc_light" / "simulation_setup.txt")).G
    peak = np.max(ref)
    assert np.max(np.abs(unshifted - ref)) > 0.05 * peak
    for got in (build_device(setup).G, np.asarray(device_tree(st, p)["G"])):
        assert np.max(np.abs(got - ref)) < 1e-6 * peak


def test_shifted_data_are_extrapolated_along_their_end_intervals() -> None:
    """Beyond either end of a layer's table the shifted index follows the line through its
    first or last two points, as the unshifted ones on the wavelengths do (SIMsalabim's)."""
    o = _parts("pin_calc_light")[0].optics
    data = o["nk"][1]  # the perovskite, 300 to 900 nm
    lam = o["lambdas"]
    for dE, end in ((0.6, -1), (-2.0, 0)):  # λ' beyond 900 nm at the longest, below 300 nm
        shift = np.array([0.0, dE, 0.0])
        lp = shifted_wavelengths(lam, dE)
        outside = lp > data[-1, 0] if end == -1 else lp < data[0, 0]
        assert outside.any()
        i, j = (-2, -1) if end == -1 else (0, 1)
        line = data[j, 1:] + (data[j, 1:] - data[i, 1:]) / (data[j, 0] - data[i, 0]) * (
            lp[outside, None] - data[j, 0])  # fmt: skip
        got = shifted_indices(o, shift)[2][outside]  # the stack's TCO, then the layers
        np.testing.assert_allclose(got.real, line[:, 0], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(got.imag, line[:, 1], rtol=1e-12, atol=1e-12)


def test_the_gradient_in_the_shift_is_finite_differences() -> None:
    """The derivative in the perovskite's shift at 0.0137 eV, between data points: of the
    generated total (`device_tree`) and of the short-circuit current (through `solve_batch`),
    by `jax.jvp` against central finite differences with a step of 1e-6 eV. A wavelength's `λ'`
    crosses a data point every 0.002 to 0.01 eV, where the derivative jumps; the step stays
    between such points (a step of 1e-5 eV crosses some and differs by 3e-4)."""
    st, p = _parts("pin_calc_light")
    x0, h = 0.0137, 1e-6

    def params(e):
        return dict(p, dE_nk=jnp.array([0.0, e, 0.0]))

    def total(e):
        return jnp.sum(device_tree(st, params(e))["G"])

    def jsc(e):
        batch = jax.tree_util.tree_map(lambda v: v[None], device_tree(st, params(e)))
        return solve_batch(batch, jnp.array([0.0]))[0][0, 0]

    for f, tol in ((total, 1e-8), (jsc, 1e-6)):
        _, by_jvp = jax.jvp(f, (x0,), (1.0,))
        fd = (float(f(x0 + h)) - float(f(x0 - h))) / (2 * h)
        assert abs(fd) > 0
        assert abs(float(by_jvp) - fd) < tol * abs(fd), f.__name__


def test_devices_of_different_shifts_batch() -> None:
    """`vmap` over devices whose layers are shifted differently, one of them not at all: each is
    the device built alone to round-off, the unshifted one that of a batch without shifts to
    the bit; and the batch solves."""
    st, p = _parts("pin_calc_light")
    shifts = np.array([[0.0, 0.0, 0.0], [0.0, 0.02, 0.0], [0.01, -0.03, 0.02]])
    stacked = jax.tree_util.tree_map(lambda v: jnp.stack([jnp.asarray(v)] * len(shifts)), p)
    build = jax.jit(jax.vmap(functools.partial(device_tree, st)))
    unshifted = build(stacked)
    stacked["dE_nk"] = jnp.asarray(shifts)
    batch = build(stacked)
    for b, u in zip(_leaves(batch), _leaves(unshifted), strict=True):
        assert np.array_equal(b[0], u[0])
    for k, dE in enumerate(shifts):
        alone = device_tree(st, dict(p, dE_nk=dE))
        for b, a in zip(_leaves(batch), _leaves(alone), strict=True):
            np.testing.assert_allclose(b[k], a, rtol=1e-13, atol=0)
    G = np.asarray(batch["G"])
    assert np.max(np.abs(G[1] - G[0])) > 1e-3 * np.max(G[0])
    J, ok, _ = solve_batch(batch, jnp.array([0.0, 0.5]))
    assert np.asarray(ok).all()
    assert len(set(np.asarray(J)[:, 0].tolist())) == len(shifts)


def test_without_computed_generation_the_shift_does_nothing() -> None:
    """With a profile from a file or a uniform one `dE_nk` is in the dict and unused."""
    for case in ("none_light", "thin_l1_light"):
        st, p = _parts(case)
        moved = dict(p, dE_nk=np.full(len(p["L"]), 0.1))
        for g, r in zip(_leaves(device_tree(st, moved)), _leaves(device_tree(st, p)),
                        strict=True):  # fmt: skip
            assert np.array_equal(g, r)


def test_a_shift_beyond_the_photon_energy_of_lambda_max_is_refused(tmp_path: Path) -> None:
    """`λ'` is finite and positive only for a shift below the photon energy of `lambda_max`
    (800 nm: 1.55 eV)."""
    case = tmp_path / "case"
    shutil.copytree(DATA / "pin_calc_light", case)
    layer = case / "L2_parameters.txt"
    layer.write_text(layer.read_text() + "dE_nk = 1.6\n")
    with pytest.raises(ValueError, match="dE_nk"):
        device_parameters(read_setup(case / "simulation_setup.txt"))
