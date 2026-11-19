"""Trap levels with their own capture coefficients (`device.TRAP_FILE_CAPTURE`: a trap file's
optional columns `C_n C_p`, jaxdd's): the layer's where the file does not give them, to the bit;
a level's own where it does, in the steady state, the batch and in time."""

import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from jaxdd.batch import prepare, solve_batch
from jaxdd.device import build_device
from jaxdd.simss_input import read_tvg, setup_from_texts
from jaxdd.solver import solve_jv
from jaxdd.transient import solve_transient

DATA = Path(__file__).parent / "data"
V = np.array([0.0, 0.4, 0.8, 1.0])
C_BULK = "5.2070311e-14"  # the absorber's C_n_bulk, C_p_bulk in both cases
C_INT = "2E-14"


def texts(case, **files):
    """The case's files as texts, `files` replacing some (by name) or editing a layer file
    (`L2_parameters.txt={"key": value}`)."""
    out = {f.name: f.read_text() for f in (DATA / case).iterdir() if f.is_file()}
    for name, new in files.items():
        name = name.replace("__", ".")
        if isinstance(new, dict):
            for key, v in new.items():
                out[name], k = re.subn(rf"^{key}\s*=\s*\S+", f"{key} = {v}", out[name], flags=re.M)
                assert k == 1, key
        else:
            out[name] = new
    return out


def own(levels, C_n, C_p):
    """A trap file with each level's own capture coefficients."""
    return "E Ntrap C_n C_p\n" + "".join(f"{E} {N} {C_n} {C_p}\n" for E, N in levels)


BULK = ((4.5, "1e21"), (4.8, "2e21"), (5.1, "1e21"))
INT = ((4.6, "1.5e14"), (5.0, "1.5e14"))


def device(case="trapfiles_light", **files):
    return build_device(setup_from_texts(texts(case, **files)), grid="simss")


def test_the_layers_coefficients_written_per_level_are_the_same_device() -> None:
    """Both trap files with each level's coefficients the layer's: the same curve to round-off,
    the coefficients now an array over points and levels."""
    plain = device()
    per_level = device(
        bulk_traps__txt=own(BULK, C_BULK, C_BULK), int_traps__txt=own(INT, C_INT, C_INT)
    )
    assert np.ndim(per_level.srh_Cn) == 2 and np.ndim(plain.srh_Cn) == 1
    assert np.ndim(per_level.interface["Cn"]) == 2
    a, b = solve_jv(plain, V), solve_jv(per_level, V)
    assert a.converged.all() and b.converged.all()
    np.testing.assert_allclose(b.J, a.J, rtol=1e-11, atol=1e-9)


def test_a_level_without_hole_capture_does_not_recombine() -> None:
    """Neutral bulk traps: a second level with `C_p` = 0 beside the recombining one leaves the
    curve as the recombining level alone has it; the same level with the layer's `C_p` does not."""
    neutral = {"bulkTrapType": 0}
    deep = ((4.8, "2e21"),)
    alone = solve_jv(
        device(L2_parameters__txt=neutral, bulk_traps__txt=own(deep, C_BULK, C_BULK)), V
    )
    shallow = ((4.8, "2e21"), (4.5, "5e21"))
    silent = f"E Ntrap C_n C_p\n4.8 2e21 {C_BULK} {C_BULK}\n4.5 5e21 {C_BULK} 0\n"
    with_silent = solve_jv(device(L2_parameters__txt=neutral, bulk_traps__txt=silent), V)
    with_both = solve_jv(
        device(L2_parameters__txt=neutral, bulk_traps__txt=own(shallow, C_BULK, C_BULK)), V
    )
    assert alone.converged.all() and with_silent.converged.all() and with_both.converged.all()
    np.testing.assert_allclose(with_silent.J, alone.J, rtol=1e-10, atol=1e-8)
    assert np.max(np.abs(with_both.J - alone.J)) > 1e-3 * np.max(np.abs(alone.J))


def test_a_batch_takes_devices_with_and_without_their_levels_coefficients() -> None:
    """One batch: a device whose levels have their own coefficients, one with a trap file
    without, one without trap files; each device's curve as `solve_jv` has it."""
    mixed = "E Ntrap C_n C_p\n4.8 2e21 5e-14 1e-16\n4.5 1e21 5e-14 5e-14\n"
    none = build_device(setup_from_texts(texts("none_light")), grid="simss")
    devices = [device(bulk_traps__txt=mixed), device(), none]
    J, ok, _ = solve_batch(prepare(devices), V)
    assert np.asarray(ok).all()
    for k, d in enumerate(devices):
        ref = solve_jv(d, V)
        scale = np.max(np.abs(ref.J))
        np.testing.assert_allclose(np.asarray(J[k]), ref.J, rtol=0, atol=1e-7 * scale)


def test_in_time_the_layers_coefficients_per_level_are_the_same() -> None:
    """ZimT's trap-file transient with each level's coefficients the layer's: the same steps."""
    case = "zimt_trapfiles_photo_light"
    tvg = read_tvg(setup_from_texts(texts(case)))
    tvg = dataclasses.replace(
        tvg, **{k: getattr(tvg, k)[:40] for k in ("t", "Vext", "oc", "G_frac", "track", "dti")}
    )
    plain = device(case)
    per_level = device(
        case, bulk_traps__txt=own(BULK, C_BULK, C_BULK), int_traps__txt=own(INT, C_INT, C_INT)
    )
    a, b = (solve_transient(prepare([d]), tvg) for d in (plain, per_level))
    assert a.converged.all() and b.converged.all()
    scale = np.max(np.abs(a.J))
    np.testing.assert_allclose(b.J, a.J, rtol=0, atol=1e-9 * scale)


@pytest.mark.parametrize(("text", "what"), [
    ("E Ntrap C_n C_p\n4.8 2e21\n", "do not give them"),
    ("E Ntrap C_n C_p\n4.8 2e21 1e-14 -1\n", "negative"),
])  # fmt: skip
def test_a_trap_files_coefficients_are_checked(text, what) -> None:
    with pytest.raises(ValueError, match=what):
        device(bulk_traps__txt=text)


def test_a_nearly_full_level_keeps_its_empty_share():
    """Over a time step a level's empty share is its own, not `1 − f̄`: a level nearly full (the
    electrons 1e25 m⁻³ beside n_t0 1e5) keeps every digit of it, so that its capture
    `n·(1 − f̄) − n_t0·f̄` is the steady state's at `dti = 0` and not round-off times `n`; over a
    step it moves from `1 − f_old` towards the steady share as `f̄` does."""
    import jax.numpy as jnp

    from jaxdd.solver import _trap_step

    Cn, Cp, nt0, pt0, n, p = 1e-14, 1e-14, 1e5, 1e5, 1e25, 1e3
    a = Cn * (n + nt0) + Cp * (p + pt0)
    exact = (Cn * nt0 + Cp * p) / a  # 1.0e-20
    mean, _, vacant = _trap_step(Cn, Cp, nt0, pt0, n, p, jnp.asarray(0.0), 0.0, empty=True)
    assert float(vacant) == pytest.approx(exact, rel=1e-12, abs=0)
    assert float(1.0 - mean) == 0.0  # what `1 − f̄` gives
    steady = Cp * (n * p - nt0 * pt0) / a  # the steady state's capture over N_t·C_n
    assert float(n * vacant - nt0 * mean) == pytest.approx(steady, rel=1e-12, abs=0)
    dti, f_old = 1e12, 0.5  # a step of 1 ps from half full
    share = -np.expm1(-a / dti) / (a / dti)
    mean, _, vacant = _trap_step(Cn, Cp, nt0, pt0, n, p, jnp.asarray(f_old), dti, empty=True)
    assert float(vacant) == pytest.approx(exact + (1 - f_old - exact) * share, rel=1e-12, abs=0)
    assert float(mean + vacant) == pytest.approx(1.0, rel=1e-14)
