"""Impedance in the frequency domain (`jaxdd.impedance`, docs/model.md "Impedance"): its limits
(the steady J-V curve's slope, the geometric capacitance) and, through implicit Euler's own
steps (`dt`), `solve_transient`'s response to a small sine, which it must be but for the sine's
amplitude and settling. ChargeTransport.jl's Ex103 cell (`test_chargetransport.device`), at
500 points."""

import numpy as np
import pytest

from jaxdd.batch import prepare, solve_states
from jaxdd.impedance import solve_impedance
from jaxdd.simss_input import TVG
from jaxdd.transient import solve_transient
from test_chargetransport import device
from test_free_surface import HTL, stack_setup

NP = 500


@pytest.mark.parametrize("ions", [0.0, 1e24])
def test_the_low_frequency_limit_is_the_steady_slope(ions) -> None:
    """At 1e-7 Hz the admittance is the steady current's slope in the voltage (`solve_states`
    at ±10 µV), with the cell's 1e21 m⁻³ of traps and, with ions, the vacancies at 1e-12 m²/Vs
    (settled in seconds): to 1e-5 of it, the difference's own accuracy."""
    d = device(ions, 1e25, 0.0, mu=1e-12, NP=NP)
    b = prepare([d])
    V0, dV = 0.6, 1e-5
    r = solve_impedance(b, [1e-7], voltage=V0, tol=1e-12)
    st = solve_states(b, [V0 - dV, V0 + dV], tol=1e-12, max_iter=200)
    slope = float((st["J"][0, 1] - st["J"][0, 0]) / (2 * dV))
    assert r.converged.all()
    assert r.Y[0, 0].real == pytest.approx(slope, rel=1e-5)
    assert abs(r.Y[0, 0].imag) < 1e-4 * slope


def test_the_high_frequency_limit_is_the_geometric_capacitance() -> None:
    """Far above every relaxation (1e14 Hz; the doped layers' dielectric one is near 1e11) the
    device is its layers' dielectrics in series, `1/∫dx/ε`, to 1e-5."""
    d = device(0.0, 1e25, 0.0, NP=NP)
    x, eps = np.asarray(d.x), np.asarray(d.eps)
    geometric = 1.0 / np.sum(np.diff(x) * 0.5 * (1 / eps[:-1] + 1 / eps[1:]))
    r = solve_impedance(prepare([d]), [1e6, 1e14])
    assert r.capacitance[0, 1] == pytest.approx(geometric, rel=1e-5)
    assert r.capacitance[0, 0] > 3 * geometric  # the depletion layers' at 1 MHz


def _sine_response(b, V0, f, amp=1e-4, per_period=40, periods=12, fitted=4):
    """`solve_transient`'s admittance at `f` from a sine of `amp` on `V0` at constant steps:
    `J = J₀ + amp·(Re Y·sin + Im Y·cos)` fitted over the last `fitted` periods."""
    dt = 1.0 / (f * per_period)
    k = np.arange(1, per_period * periods + 1)
    t = np.concatenate([[0.0], k * dt])
    n = len(t)
    tvg = TVG(t=t, Vext=np.concatenate([[V0], V0 + amp * np.sin(2 * np.pi * f * k * dt)]),
              oc=np.zeros(n, bool), G_frac=np.full(n, float(b["g_frac"][0])),
              track=np.zeros(n, int), dti=np.concatenate([[0.0], 1.0 / np.diff(t)]))  # fmt: skip
    r = solve_transient(b, tvg, tol=1e-12, max_iter=60)
    assert np.asarray(r.converged).all()
    m = k > per_period * (periods - fitted)
    phase = 2 * np.pi * f * k[m] * dt
    X = np.stack([np.ones(m.sum()), np.sin(phase), np.cos(phase)], -1)
    c = np.linalg.lstsq(X, np.asarray(r.J)[0, 1:][m], rcond=None)[0]
    return (c[1] + 1j * c[2]) / amp, dt


@pytest.mark.parametrize("ions, f, G_frac, tol", [
    (0.0, 1e4, 0.0, 1e-7),  # the traps' occupations as a transient solves them, over each step
    (0.0, 1e5, 1.0, 1e-7),
    (1e24, 1e2, 0.0, 1e-6),
])  # fmt: skip
def test_implicit_euler_steps_are_the_transients_response(ions, f, G_frac, tol) -> None:
    """With `dt` the admittance is what `solve_transient` sees through implicit Euler at steps of
    `dt` (40 a period, 12 periods): its response to a 0.1 mV sine at 0.6 V to 1e-7 (1.2e-8 dark,
    1.0e-8 lit: the sine's amplitude and settling; with ions 1.6e-7), the cell's traps and their
    occupations' dynamics, the ions' number kept as a transient keeps it, included. The
    continuous response differs from it by implicit Euler's error, 4–8 % at 40 steps a period."""
    d = device(ions, 1e25, G_frac, mu=1e-12, NP=NP)
    b = prepare([d])
    Y_t, dt = _sine_response(b, 0.6, f)
    discrete = solve_impedance(b, [f], voltage=0.6, dt=dt, tol=1e-12).Y[0, 0]
    continuous = solve_impedance(b, [f], voltage=0.6, tol=1e-12).Y[0, 0]
    assert abs(discrete / Y_t - 1) < tol
    assert abs(continuous / Y_t - 1) > 1e-2


@pytest.mark.parametrize("f, G_frac", [(1e4, 0.0), (1e5, 1.0)])
def test_interface_traps_follow_at_their_own_rates(f, G_frac) -> None:
    """Interface traps (1e16 m⁻² at both of the cell's interfaces) are each edge's occupation,
    coupling its two points (`impedance._interface_blocks`) and in the current through each
    point's rate (`tr["dRi"]`): with `dt` `solve_transient`'s response to a 10 µV sine to 1e-7
    (1.2e-8 dark at 10 kHz, 2.3e-8 lit at 100 kHz; at 0.1 mV the sine's own nonlinearity is
    1e-6 here). With their occupation instantaneous instead it was 3e-2 and 7e-2 off."""
    traps = [("L1.txt", "N_t_int", 1e16), ("L2.txt", "N_t_int", 1e16)]
    b = prepare([device(0.0, 1e25, G_frac, NP=NP, extra=traps)])
    Y_t, dt = _sine_response(b, 0.6, f, amp=1e-5)
    discrete = solve_impedance(b, [f], voltage=0.6, dt=dt, tol=1e-12).Y[0, 0]
    assert abs(discrete / Y_t - 1) < 1e-7


def test_the_series_resistance_and_the_shunt() -> None:
    """`Z = 1/(Y + G_shunt) + R_series`, `Y` the internal admittance."""
    d = device(0.0, 1e25, 0.0, NP=NP, extra=())
    b = prepare([d])
    plain = solve_impedance(b, [1e3, 1e6])
    b2 = dict(b, R_series=b["R_series"] + 1e-4, G_shunt=b["G_shunt"] + 10.0)
    r = solve_impedance(b2, [1e3, 1e6])
    np.testing.assert_allclose(r.Y, plain.Y, rtol=1e-12)
    np.testing.assert_allclose(r.Z, 1.0 / (plain.Y + b2["G_shunt"][:, None]) + 1e-4
                               + b["R_series"][:, None], rtol=1e-12)  # fmt: skip


def test_what_impedance_refuses() -> None:
    """A free end (no circuit), an unknown ion flux."""
    from jaxdd.device import build_device

    with pytest.raises(NotImplementedError, match="free end"):
        solve_impedance(prepare([build_device(stack_setup([HTL, {}], W_L=5.0))]), [1e3])
    with pytest.raises(ValueError, match="ion_flux"):
        solve_impedance(prepare([device(0.0, 1e25, 0.0, NP=NP)]), [1e3], ion_flux="drift")
