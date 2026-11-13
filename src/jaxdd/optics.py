"""The generation profile by the transfer-matrix method: SIMsalabim 5.36's `TransferMatrix` unit
(`genProfile = calc`), after Pettersson, Roman and Inganäs (J. Appl. Phys. 86, 487 (1999)).

Light enters from a substrate (glass, semi-infinite on its far side, its own front reflection
`R_glass` counted incoherently), passes a transparent electrode (`L_TCO`, may be 0), the device's
layers and a back electrode (`L_BE`). For each wavelength on a 1 nm grid the optical field at every
grid point of the device follows from 2×2 interface and layer matrices; the generation is

    G(x) = Σ_λ Δλ · α(x, λ) · n(x, λ) · AM(λ) · |E(x, λ)|² · λ / (h·c),    α = 4π·k/λ.

The refractive indices and the spectrum are read once and interpolated onto the wavelengths
(`optical_data`, NumPy); the profile on the grid (`generation`) runs in NumPy or `jax.numpy`, and in
JAX it is differentiable in the layers' thicknesses, which move both the grid and the optics.

SIMsalabim evaluates the field on its own grid extended by uniform 1 nm grids in the two
electrodes; the field at a point depends on that point alone, so only the device's points are
computed here. Its linear interpolation takes the two data points left of the bracketing interval
(docs/differences.md); on data at whole nanometres, as all of SIMsalabim's files are, that is the
plain interpolation used here.
"""

from __future__ import annotations

import io

import numpy as np

__all__ = ["LAMBDA_STEP", "generation", "optical_data"]

#: SIMsalabim's `lambdaStep` and `minDeltaLambda` (m), and its `h` and `c`.
LAMBDA_STEP = 1e-9
MIN_DELTA_LAMBDA = 0.5e-9
H_PLANCK = 6.62606957e-34
C_LIGHT = 2.99792458e8


def _table(text: str, columns: int) -> np.ndarray:
    """A whitespace table with one header line."""
    data = np.loadtxt(io.StringIO(text), skiprows=1, ndmin=2)
    if data.shape[1] < columns:
        raise ValueError(f"expected {columns} columns, got {data.shape[1]}")
    return data[:, :columns]


def _interpolate(x0: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear, extrapolated linearly beyond either end (SIMsalabim's `InterExtraPolation` with
    full extrapolation, on data whose points the wavelengths hit)."""
    inside = np.interp(x0, x, y)
    lo, hi = x0 < x[0], x0 > x[-1]
    inside[lo] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (x0[lo] - x[0])
    inside[hi] = y[-1] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (x0[hi] - x[-1])
    return inside


def _wavelengths(lambda_min: float, lambda_max: float) -> np.ndarray:
    """SIMsalabim's: `lambda_min + k·Δλ`, each rounded to 7 significant digits (`%.6e`)."""
    if lambda_min == lambda_max:
        return np.array([lambda_min])
    count = round((lambda_max - lambda_min) / LAMBDA_STEP) + 1
    return np.array([float(f"{lambda_min + k * LAMBDA_STEP:.6e}") for k in range(count)])


def _nk(text: str, lambdas: np.ndarray, name: str) -> np.ndarray:
    data = _table(text, 3)
    if np.any(np.diff(data[:, 0]) < 0.999 * MIN_DELTA_LAMBDA):
        raise ValueError(f"{name}: wavelengths closer than {MIN_DELTA_LAMBDA} m")
    return _interpolate(lambdas, data[:, 0], data[:, 1]) + 1j * _interpolate(
        lambdas, data[:, 0], data[:, 2]
    )


def optical_data(setup, layers) -> dict:
    """What the profile needs that the thicknesses do not change: the wavelengths, the complex
    refractive index of the substrate, the TCO (if `L_TCO` > 0), every layer and the back electrode
    on them, the spectrum, the substrate's reflection, and the electrodes' thicknesses. `setup` a
    `simss_input.Setup`, its files read with `Setup.read`."""
    f = setup.f
    if f("L_TCO") < 0 or f("L_BE") <= 0:
        raise ValueError("L_TCO must not be negative and L_BE must be positive")
    if f("lambda_min") <= 0 or f("lambda_min") > f("lambda_max"):
        raise ValueError("0 < lambda_min <= lambda_max is required")
    lambdas = _wavelengths(f("lambda_min"), f("lambda_max"))

    def nk(file: str) -> np.ndarray:
        return _nk(setup.read(file), lambdas, file)

    n_sub = nk(setup.raw["nkSubstrate"])
    stack = [nk(ly.raw["nkLayer"]) for ly in layers]
    tco = f("L_TCO") > 0
    n = np.stack(([nk(setup.raw["nkTCO"])] if tco else []) + stack + [nk(setup.raw["nkBE"])])
    spectrum = _table(setup.read(setup.raw["spectrum"]), 2)
    if np.any(np.diff(spectrum[:, 0]) < 0.999 * MIN_DELTA_LAMBDA):
        raise ValueError(f"spectrum: wavelengths closer than {MIN_DELTA_LAMBDA} m")
    AM = _interpolate(lambdas, spectrum[:, 0], spectrum[:, 1])
    # The glass's front reflection; SIMsalabim sets it to 0 for n = 1 exactly.
    R_glass = np.where(n_sub == 1.0, 0.0, np.abs(((1 - n_sub) / (1 + n_sub)) ** 2))
    return {"lambdas": lambdas, "n_sub": n_sub, "n": n, "AM": AM, "R_glass": R_glass,
            "L_TCO": f("L_TCO") if tco else None, "L_BE": f("L_BE")}  # fmt: skip


def _interface(n1, n2):
    """`I` of an interface from medium 1 to 2 (… × K), as `(a, b, c, d)` of [[a, b], [c, d]]."""
    t = 2 * n1 / (n1 + n2)
    r = (n1 - n2) / (n1 + n2)
    return 1 / t, r / t, r / t, 1 / t


def _layer(n, d, lambdas, xp):
    phase = 2 * np.pi * n / lambdas * d
    zero = xp.zeros_like(phase)
    return xp.exp(-1j * phase), zero, zero, xp.exp(1j * phase)


def _product(m1, m2):
    a, b, c, d = m1
    e, f, g, h = m2
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def _eye(like, xp):
    one, zero = xp.ones_like(like), xp.zeros_like(like)
    return one, zero, zero, one


def generation(optics: dict, x, lid, thicknesses, layer_gen, xp=np):
    """The generation profile (m⁻³ s⁻¹) at the device's grid points `x` (layer `lid` of each),
    for the layers' `thicknesses`: SIMsalabim's `Calc_E` and `Calc_Generation_Rate`; zero in a
    layer with `layerGen = 0`."""
    lambdas, n_sub, n = optics["lambdas"], optics["n_sub"], optics["n"]
    tco = optics["L_TCO"] is not None
    d = ([optics["L_TCO"]] if tco else []) + list(thicknesses) + [optics["L_BE"]]
    first = 1 if tco else 0  # the stack's index of the device's first layer
    count = len(d)

    # The whole stack (for the reflection): substrate | TCO | layers | BE.
    S = _interface(n_sub, n[0])
    for j in range(count - 1):
        S = _product(S, _product(_layer(n[j], d[j], lambdas, xp), _interface(n[j], n[j + 1])))
    R = xp.abs(S[2] / S[0]) ** 2
    T = xp.abs(2 / (1 + n_sub)) / xp.real(xp.sqrt(1 - optics["R_glass"] * R))

    # For each device layer m: S' up to its left side, S'' from its right side on.
    rows = []
    start = 0.0
    for k in range(len(thicknesses)):
        m = first + k
        Sp = _interface(n_sub, n[0])
        for j in range(1, m + 1):
            Sp = _product(Sp, _product(_layer(n[j - 1], d[j - 1], lambdas, xp),
                                       _interface(n[j - 1], n[j])))  # fmt: skip
        Spp = _eye(n[0], xp)
        for j in range(m, count - 1):
            Spp = _product(Spp, _product(_interface(n[j], n[j + 1]),
                                         _layer(n[j + 1], d[j + 1], lambdas, xp)))  # fmt: skip
        xi = 2 * np.pi * n[m] / lambdas
        dm = d[m]
        denom = Sp[0] * Spp[0] * xp.exp(-1j * xi * dm) + Sp[1] * Spp[2] * xp.exp(1j * xi * dm)
        rows.append((start, xi, dm, T * Spp[0] / denom, T * Spp[2] / denom, m))
        start = start + thicknesses[k]

    # E at each point from its own layer's row.
    def field(row, xs):
        s, xi, dm, a, b, _ = row
        rel = (xs - s)[:, None]
        return a * xp.exp(-1j * xi * (dm - rel)) + b * xp.exp(1j * xi * (dm - rel))

    alpha = 4 * np.pi * np.imag(n) / lambdas
    weight = LAMBDA_STEP * alpha * np.real(n) * optics["AM"] * lambdas / (H_PLANCK * C_LIGHT)
    G = xp.zeros_like(x)
    for k, row in enumerate(rows):
        E2 = xp.abs(field(row, x)) ** 2
        Gk = xp.sum(weight[row[5]] * E2, axis=1)
        G = xp.where((lid == k) & bool(layer_gen[k]), Gk, G)
    return G
