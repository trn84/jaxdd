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

jaxdd's own: each device layer's n,k may be shifted in photon energy by its `dE_nk` (eV; positive
moves the absorption edge to higher energy), `n(λ) = n_data(λ')` with `λ'` the wavelength of the
photon energy `h·c/λ − dE_nk`. The layers' tables are kept for it (`optical_data`) and
interpolated at `λ'` in NumPy or `jax.numpy` (`shifted_indices`), so that in JAX the profile is
differentiable in the shifts too. A zero shift gives `λ' = λ` exactly, and the stored n,k to the
bit.

SIMsalabim evaluates the field on its own grid extended by uniform 1 nm grids in the two
electrodes; the field at a point depends on that point alone, so only the device's points are
computed here. Its linear interpolation takes the two data points left of the bracketing interval
(docs/differences.md); on data at whole nanometres, as all of SIMsalabim's files are, that is the
plain interpolation used here.
"""

from __future__ import annotations

import io

import numpy as np

from jaxdd.constants import Q

__all__ = ["LAMBDA_STEP", "generation", "optical_data", "shifted_indices", "shifted_wavelengths"]

#: SIMsalabim's `lambdaStep` and `minDeltaLambda` (m), and its `h` and `c`.
LAMBDA_STEP = 1e-9
MIN_DELTA_LAMBDA = 0.5e-9
H_PLANCK = 6.62606957e-34
C_LIGHT = 2.99792458e8
#: `h·c/q` (eV m): a photon's energy in eV is `HC_Q/λ`.
HC_Q = H_PLANCK * C_LIGHT / Q


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


def _interpolate_xp(x0, x: np.ndarray, y: np.ndarray, xp=np):
    """`_interpolate` in NumPy or `jax.numpy`, at wavelengths `x0` that may be traced: in the
    interval `x[j] ≤ x0 < x[j+1]`, beyond either end along the first or the last interval, as
    `slope·(x0 − x[j]) + y[j]`, NumPy's own expression, so that on a data point it gives the
    point's value to the bit; differentiable in `x0` between the points."""
    slope = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    j = xp.clip(xp.searchsorted(x, x0, side="right") - 1, 0, len(x) - 1)
    return xp.take(slope, xp.minimum(j, len(x) - 2)) * (x0 - xp.take(x, j)) + xp.take(y, j)


def _wavelengths(lambda_min: float, lambda_max: float) -> np.ndarray:
    """SIMsalabim's: `lambda_min + k·Δλ`, each rounded to 7 significant digits (`%.6e`)."""
    if lambda_min == lambda_max:
        return np.array([lambda_min])
    count = round((lambda_max - lambda_min) / LAMBDA_STEP) + 1
    return np.array([float(f"{lambda_min + k * LAMBDA_STEP:.6e}") for k in range(count)])


def _nk_table(text: str, name: str) -> np.ndarray:
    """An n,k file's table: wavelength (m), n, k."""
    data = _table(text, 3)
    if np.any(np.diff(data[:, 0]) < 0.999 * MIN_DELTA_LAMBDA):
        raise ValueError(f"{name}: wavelengths closer than {MIN_DELTA_LAMBDA} m")
    return data


def _nk(data: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    return _interpolate(lambdas, data[:, 0], data[:, 1]) + 1j * _interpolate(
        lambdas, data[:, 0], data[:, 2]
    )


def shifted_wavelengths(lambdas, dE):
    """The wavelengths `λ'` whose photon energy is that of `lambdas` less `dE` (eV),
    `λ' = λ/(1 − dE·λ/(h·c/q))`: `λ` itself for `dE = 0`. Finite and positive for `dE` below
    the photon energy of the longest wavelength."""
    return lambdas / (1 - dE * lambdas / HC_Q)


def shifted_indices(optics: dict, dE, xp=np):
    """The stack's complex refractive index (`optical_data`'s `n`) with each device layer's n,k
    shifted in photon energy by its `dE` (eV, over the layers): `n(λ) = n_data(λ')`
    (`shifted_wavelengths`), the layer's table interpolated linearly at `λ'` and beyond either
    end extrapolated along its first or last interval, as on the wavelengths. The TCO and the
    back electrode keep theirs. Differentiable in `dE` in JAX, on a data point (with data on the
    wavelengths, every `λ'` at a zero shift) for an increasing shift; a zero shift gives
    `optics["n"]` to the bit."""
    lambdas, n = optics["lambdas"], optics["n"]
    first = 1 if optics["L_TCO"] is not None else 0
    rows = [n[j] for j in range(first)]
    for k, data in enumerate(optics["nk"]):
        lam = shifted_wavelengths(lambdas, dE[k])
        rows.append(_interpolate_xp(lam, data[:, 0], data[:, 1], xp)
                    + 1j * _interpolate_xp(lam, data[:, 0], data[:, 2], xp))  # fmt: skip
    rows.append(n[-1])
    return xp.stack(rows)


def optical_data(setup, layers) -> dict:
    """What the profile needs that the thicknesses do not change: the wavelengths, the complex
    refractive index of the substrate, the TCO (if `L_TCO` > 0), every layer and the back electrode
    on them, the spectrum, the substrate's reflection, and the electrodes' thicknesses; and each
    device layer's n,k table as read (`nk`: wavelength, n, k), for a shift in photon energy
    (`shifted_indices`). `setup` a `simss_input.Setup`, its files read with `Setup.read`."""
    f = setup.f
    if f("L_TCO") < 0 or f("L_BE") <= 0:
        raise ValueError("L_TCO must not be negative and L_BE must be positive")
    if f("lambda_min") <= 0 or f("lambda_min") > f("lambda_max"):
        raise ValueError("0 < lambda_min <= lambda_max is required")
    lambdas = _wavelengths(f("lambda_min"), f("lambda_max"))

    def table(file: str) -> np.ndarray:
        return _nk_table(setup.read(file), file)

    def nk(file: str) -> np.ndarray:
        return _nk(table(file), lambdas)

    n_sub = nk(setup.raw["nkSubstrate"])
    tables = tuple(table(ly.raw["nkLayer"]) for ly in layers)
    stack = [_nk(data, lambdas) for data in tables]
    tco = f("L_TCO") > 0
    n = np.stack(([nk(setup.raw["nkTCO"])] if tco else []) + stack + [nk(setup.raw["nkBE"])])
    spectrum = _table(setup.read(setup.raw["spectrum"]), 2)
    if np.any(np.diff(spectrum[:, 0]) < 0.999 * MIN_DELTA_LAMBDA):
        raise ValueError(f"spectrum: wavelengths closer than {MIN_DELTA_LAMBDA} m")
    AM = _interpolate(lambdas, spectrum[:, 0], spectrum[:, 1])
    # The glass's front reflection; SIMsalabim sets it to 0 for n = 1 exactly.
    R_glass = np.where(n_sub == 1.0, 0.0, np.abs(((1 - n_sub) / (1 + n_sub)) ** 2))
    return {"lambdas": lambdas, "n_sub": n_sub, "n": n, "AM": AM, "R_glass": R_glass,
            "L_TCO": f("L_TCO") if tco else None, "L_BE": f("L_BE"), "nk": tables}  # fmt: skip


def _interface(n1, n2):
    """`I` of an interface from medium 1 to 2 (… × K), as `(a, b, c, d)` of [[a, b], [c, d]]."""
    t = 2 * n1 / (n1 + n2)
    r = (n1 - n2) / (n1 + n2)
    return 1 / t, r / t, r / t, 1 / t


def _layer(xi, d, xp):
    phase = xi * d
    zero = xp.zeros_like(phase)
    return xp.exp(-1j * phase), zero, zero, xp.exp(1j * phase)


def _product(m1, m2):
    a, b, c, d = m1
    e, f, g, h = m2
    return a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h


def _eye(like, xp):
    one, zero = xp.ones_like(like), xp.zeros_like(like)
    return one, zero, zero, one


def _media(n_sub, n, lambdas, AM, xp):
    """What the profile takes of the stack's indices `n`: the interfaces from the substrate on
    (`_interface`; `[j + 1]` from medium j to j + 1), each medium's `ξ = 2π·n/λ`, and its
    generation's weight per wavelength, `Δλ·α·n·AM·λ/(h·c)`."""
    alpha = 4 * np.pi * xp.imag(n) / lambdas
    weight = LAMBDA_STEP * alpha * xp.real(n) * AM * lambdas / (H_PLANCK * C_LIGHT)
    return ([_interface(n_sub, n[0])] + [_interface(n[j], n[j + 1]) for j in range(len(n) - 1)],
            [2 * np.pi * n[j] / lambdas for j in range(len(n))], weight)  # fmt: skip


def _rows(optics: dict, n, thicknesses, xp):
    """For each device layer, the field's coefficients over the wavelengths (`Calc_E`): its
    start, `ξ`, thickness, the forward and the backward wave's amplitude and its index in the
    stack; and the media's weights (`_media`), for the stack's indices `n`."""
    faces, xis, weight = _media(optics["n_sub"], n, optics["lambdas"], optics["AM"],
                                np if isinstance(n, np.ndarray) else xp)  # fmt: skip
    tco = optics["L_TCO"] is not None
    d = ([optics["L_TCO"]] if tco else []) + list(thicknesses) + [optics["L_BE"]]
    first = 1 if tco else 0  # the stack's index of the device's first layer
    count = len(d)

    # The whole stack (for the reflection): substrate | TCO | layers | BE.
    S = faces[0]
    for j in range(count - 1):
        S = _product(S, _product(_layer(xis[j], d[j], xp), faces[j + 1]))
    R = xp.abs(S[2] / S[0]) ** 2
    T = xp.abs(2 / (1 + optics["n_sub"])) / xp.real(xp.sqrt(1 - optics["R_glass"] * R))

    # For each device layer m: S' up to its left side, S'' from its right side on.
    rows = []
    start = 0.0
    for k in range(len(thicknesses)):
        m = first + k
        Sp = faces[0]
        for j in range(1, m + 1):
            Sp = _product(Sp, _product(_layer(xis[j - 1], d[j - 1], xp), faces[j]))
        Spp = _eye(optics["n"][0], xp)
        for j in range(m, count - 1):
            Spp = _product(Spp, _product(faces[j + 1], _layer(xis[j + 1], d[j + 1], xp)))
        xi = xis[m]
        dm = d[m]
        denom = Sp[0] * Spp[0] * xp.exp(-1j * xi * dm) + Sp[1] * Spp[2] * xp.exp(1j * xi * dm)
        rows.append((start, xi, dm, T * Spp[0] / denom, T * Spp[2] / denom, m))
        start = start + thicknesses[k]
    return rows, weight


def generation(optics: dict, x, lid, thicknesses, layer_gen, xp=np, dE=None):
    """The generation profile (m⁻³ s⁻¹) at the device's grid points `x` (layer `lid` of each),
    for the layers' `thicknesses`: SIMsalabim's `Calc_E` and `Calc_Generation_Rate`; zero in a
    layer with `layerGen = 0`. With `dE` (eV, over the layers) each layer's n,k shifted in
    photon energy by it (`shifted_indices`). Where every shift is zero the profile and its
    derivatives in the thicknesses are those of the stored indices: shifts known to be zero are
    not used; traced ones (in JAX) give the field's coefficients from both indices and the
    stored ones' taken (`jax.numpy`'s complex division may differ from NumPy's in the last bit),
    with the derivatives in the shifts of the shifted ones, whose thicknesses are then held
    constant."""
    rows, weight = _rows(optics, optics["n"], thicknesses, xp)
    if dE is not None and xp is not np:
        import jax

        if not isinstance(dE, jax.core.Tracer) and not np.any(np.asarray(dE)):
            dE = None  # known to be zero: the stored indices' graph alone
    if dE is not None and xp is np:
        if np.any(dE):
            rows, weight = _rows(optics, shifted_indices(optics, dE), thicknesses, np)
    elif dE is not None:
        zero = xp.all(dE == 0)
        sg = jax.lax.stop_gradient

        def pick(s, t):  # at zero shift the stored value and its derivatives, plus t's in dE
            return xp.where(zero, s + (t - sg(t)), t)

        held = [xp.where(zero, sg(d), d) for d in thicknesses]  # at zero t's derivatives in dE only
        moved, moved_weight = _rows(optics, shifted_indices(optics, dE, xp), held, xp)
        weight = pick(weight, moved_weight)
        rows = [
            (s, pick(xi, xi_t), dm, pick(a, a_t), pick(b, b_t), m)
            for (s, xi, dm, a, b, m), (_, xi_t, _, a_t, b_t, _) in zip(rows, moved, strict=True)
        ]

    # E at each point from its own layer's row.
    def field(row, xs):
        s, xi, dm, a, b, _ = row
        rel = (xs - s)[:, None]
        return a * xp.exp(-1j * xi * (dm - rel)) + b * xp.exp(1j * xi * (dm - rel))

    G = xp.zeros_like(x)
    for k, row in enumerate(rows):
        E2 = xp.abs(field(row, x)) ** 2
        Gk = xp.sum(weight[row[5]] * E2, axis=1)
        G = xp.where((lid == k) & bool(layer_gen[k]), Gk, G)
    return G
