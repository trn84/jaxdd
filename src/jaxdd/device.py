"""A SIMsalabim device on SIMsalabim's grid: the static arrays the solver reads.

Follows SIMsalabim 5.36 (`Units/DDRoutines.pas`: `Make_Grid`, `Define_Layers`,
`Update_Gen_Pot`, `Init_Generation_Profile`, `Init_Trapping`, `Calc_Elec_Mob`), so that both
solve the same discrete equations. Plain NumPy: built once per device, then handed to JAX.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from jaxdd.constants import EPS_0, K_B, Q
from jaxdd.simss_input import Setup

__all__ = ["GRID_SPLITS", "Device", "build_device", "layer_points", "make_grid"]

#: SIMsalabim's `MinGridPointsPerLayer`.
MIN_GRID_POINTS_PER_LAYER = 5

#: How `layer_points` evaluates `Make_Grid`'s share of the points (docs/differences.md §5):
#: "exact" in rational arithmetic from the input's decimal values, "simss" as SimSS 5.36's
#: 10-byte reals round it.
GRID_SPLITS = ("exact", "simss")

#: The significand of SimSS's `myReal`, the x87 80-bit extended type.
_EXTENDED_BITS = 64


def _extended(q: Fraction) -> Fraction:
    """`q` rounded to the nearest 80-bit extended value, ties to even (x87's default)."""
    if q == 0:
        return q
    sign, q = (-1, -q) if q < 0 else (1, q)
    e = q.numerator.bit_length() - q.denominator.bit_length()
    if q < Fraction(2) ** e:
        e -= 1
    scaled = q / Fraction(2) ** (e - _EXTENDED_BITS + 1)  # in [2^63, 2^64)
    whole = scaled.numerator // scaled.denominator
    rest = scaled - whole
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and whole % 2):
        whole += 1
    return sign * whole * Fraction(2) ** (e - _EXTENDED_BITS + 1)


def layer_points(thicknesses: Sequence[str | float], NP: int, split: str = "exact") -> list[int]:
    """The points `Make_Grid` gives each layer beyond its first:
    `max(MinGridPointsPerLayer, FLOOR((NP − n_thin·MinGridPointsPerLayer)·L/(Ltot − L_thin)))`,
    a layer being thin where `NP·L/Ltot` is below `MinGridPointsPerLayer`.

    Where the share is a whole number in exact arithmetic (every layer but one thin, or
    thicknesses in simple ratios), `FLOOR` decides by the last bit: SimSS's 10-byte reals land
    below it on some stacks, float64 on others. `split` "exact" takes the share in rational
    arithmetic from the decimal values (a float is read as its shortest decimal, `repr`);
    "simss" repeats SimSS's operations in its order, each rounded to an 80-bit extended
    value, the input read as its nearest one.
    """
    if split not in GRID_SPLITS:
        raise ValueError(f"split {split!r}: one of {GRID_SPLITS}")
    exact = [Fraction(L if isinstance(L, str) else repr(float(L))) for L in thicknesses]
    r = (lambda q: q) if split == "exact" else _extended
    values = [r(L) for L in exact]
    total = Fraction(0)
    for L in values:
        total = r(total + L)
    thin = [L for L in values if r(r(NP * L) / total) < MIN_GRID_POINTS_PER_LAYER]
    thin_total = Fraction(0)
    for L in thin:
        thin_total = r(thin_total + L)
    free = NP - len(thin) * MIN_GRID_POINTS_PER_LAYER
    rest = r(total - thin_total)
    shares = [r(free * r(L / rest)) for L in values]
    return [max(MIN_GRID_POINTS_PER_LAYER, s.numerator // s.denominator) for s in shares]


def _sub_grid(h: np.ndarray, istart: int, ifinish: int, grad: float, span: float, total: float):
    """SIMsalabim's `Make_Sub_Grid`: an exponential grid, symmetric in the layer."""
    h[istart : ifinish + 1] = 1.0
    ending = round(0.5 * (ifinish - istart))
    for i in range(ending + 1):
        h[i + istart] = np.exp(grad * (1.0 / ending - 1.0 / (i + 1)))
        h[ifinish - i] = h[i + istart]
    norm = h[istart : ifinish + 1].sum()
    h[istart : ifinish + 1] *= span / (total * norm)


def make_grid(thicknesses: Sequence[str | float], NP: int, grad: float, split: str = "exact"):
    """`(x, h, i0, i1, lid)` as SIMsalabim's `Make_Grid`: `x` has `NP + 2` points, `h` is the
    spacing over the total length `L` (`h[i] = (x[i+1] − x[i])/L`), `i0`/`i1` each layer's first
    and last point, `lid` the layer of each point (0-based). The thicknesses in metres, as
    numbers or as the input's decimal strings; `split` as in `layer_points`.

    `L` is the sum of the layers, and `x[NP+1]` is not quite it: equalising the spacing
    across each interface leaves `Σh` slightly off 1 (0.006 % on a 1/500/1 nm stack).
    SIMsalabim uses `L` wherever it scales, so does this."""
    counts = layer_points(thicknesses, NP, split)
    thicknesses = [float(L) for L in thicknesses]
    total = sum(thicknesses)
    n_layers = len(thicknesses)
    h = np.zeros(NP + 2)
    lid = np.zeros(NP + 2, dtype=int)
    i0, i1 = [], []
    istart, xstart = 0, 0.0
    for k, L in enumerate(thicknesses):
        ifinish = istart + counts[k]
        i0.append(istart)
        i1.append(ifinish)
        if k == n_layers - 1:
            ifinish = NP + 1
        _sub_grid(h, istart, ifinish, grad, L, total)
        lid[istart : ifinish + 1] = k
        istart = ifinish + 1
        xstart += L
    i1[-1] = NP + 1
    for k in range(n_layers - 1):
        j = i0[k + 1]
        h[j - 2 : j + 1] = (h[j] + h[j - 1] + h[j - 2]) / 3.0
    x = np.zeros(NP + 2)
    for i in range(1, NP + 2):
        x[i] = x[i - 1] + total * h[i - 1]
    return x, h, i0, i1, lid


@dataclass(frozen=True)
class Device:
    """Every static array of one device, on its `NP + 2` grid points (SI units)."""

    x: np.ndarray
    h: np.ndarray
    L: float
    T: float
    Vt: float
    eps: np.ndarray
    E_c: np.ndarray
    E_v: np.ndarray
    N_c: np.ndarray
    ni: np.ndarray
    N_D: np.ndarray
    N_A: np.ndarray
    mu_n: np.ndarray  # on the interleaved mesh: mu_n[i] between x[i] and x[i+1]
    mu_p: np.ndarray
    k_direct: np.ndarray  # 0 on the electrodes
    srh_Nt: np.ndarray
    srh_Cn: np.ndarray
    srh_Cp: np.ndarray
    srh_nt0: np.ndarray
    srh_pt0: np.ndarray
    G: np.ndarray  # generation at G_frac = 1, 0 on the electrodes
    offset_n: np.ndarray  # Vgn − V
    offset_p: np.ndarray  # Vgp − V
    V0: float
    VL: float
    n_left: float
    n_right: float
    p_left: float
    p_right: float
    W_L: float
    W_R: float
    G_frac: float
    i0: tuple[int, ...]
    i1: tuple[int, ...]
    # Ions (SIMsalabim's `Init_Ionic_Region`, one layer per region in Stage A): per species, the
    # mobile region's mask and the number of ions per area it holds (N·L of the layer), or a
    # fixed density where they cannot move.
    anion_mask: np.ndarray
    cation_mask: np.ndarray
    anion_total: float
    cation_total: float
    anion_fixed: np.ndarray
    cation_fixed: np.ndarray
    # SIMsalabim's pre-bias: solve at `V_pre` with the ions free first (`preCond`), then hold
    # them for the scan (`fixIons`).
    pre_cond: bool
    V_pre: float
    fix_ions: bool


def _generation(setup: Setup, x: np.ndarray, lid: np.ndarray, layer_gen: np.ndarray,
                total: float) -> np.ndarray:  # fmt: skip
    """SIMsalabim's `Init_Generation_Profile`, `Use_gen_profile` 2: the file's profile, its
    x-axis stretched to the device, interpolated linearly; zero where a layer does not
    generate and on any point the interpolation loop does not reach."""
    table = np.loadtxt(setup.path("genProfile"), skiprows=1)
    a, gr = table[:, 0].copy(), table[:, 1]
    a *= total / a[-1]
    g = np.zeros_like(x)
    i = 0
    NP = len(x) - 2
    for j in range(len(a) - 1):
        while x[i] < a[j + 1] and i < NP + 1:
            if layer_gen[lid[i]]:
                g[i] = gr[j] + (x[i] - a[j]) * (gr[j + 1] - gr[j]) / (a[j + 1] - a[j])
            i += 1
    return g


def build_device(setup: Setup, grid_split: str = "exact") -> Device:
    """The device of a SIMsalabim input (Stage A, `check_supported` first). `grid_split`
    "simss" shares the grid points among the layers as SimSS's 10-byte reals do
    (`layer_points`; docs/differences.md §5), for a comparison point by point."""
    layers = setup.layers
    NP, grad = setup.i("NP"), setup.f("grad")
    thicknesses = [ly.f("L") for ly in layers]
    total = sum(thicknesses)
    x, h, i0, i1, lid = make_grid([ly.raw["L"] for ly in layers], NP, grad, grid_split)
    T = setup.f("T")
    Vt = K_B * T / Q

    def per_point(name: str) -> np.ndarray:
        values = np.array([ly.f(name) for ly in layers])
        return values[lid]

    E_c, E_v, N_c = per_point("E_c"), per_point("E_v"), per_point("N_c")
    ni = N_c * np.exp(-0.5 * (E_v - E_c) / Vt)
    k_direct = per_point("k_direct")
    k_direct[0] = k_direct[-1] = 0.0

    def interleaved(name: str, nu_name: str) -> np.ndarray:
        mu = per_point(name).copy()
        for j in range(len(layers) - 1):
            mu[i1[j]] = (total * h[i1[j]]) * layers[j].f(nu_name) / Vt
        return mu

    srh = {key: np.zeros(NP + 2) for key in ("Nt", "Cn", "Cp", "nt0", "pt0")}
    for j, ly in enumerate(layers):
        if ly.f("N_t_bulk") == 0:
            continue
        lo, hi = max(1, i0[j]), min(NP, i1[j])
        E_t = ly.f("E_t_bulk")
        srh["Nt"][lo : hi + 1] = ly.f("N_t_bulk")
        srh["Cn"][lo : hi + 1] = ly.f("C_n_bulk")
        srh["Cp"][lo : hi + 1] = ly.f("C_p_bulk")
        srh["nt0"][lo : hi + 1] = ly.f("N_c") * np.exp((ly.f("E_c") - E_t) / Vt)
        srh["pt0"][lo : hi + 1] = ly.f("N_c") * np.exp((E_t - ly.f("E_v")) / Vt)

    layer_gen = np.array([ly.i("layerGen") for ly in layers], dtype=bool)
    G = _generation(setup, x, lid, layer_gen, total)
    G[0] = G[-1] = 0.0

    ions = {}
    for species, conc_name, mob_name in (("anion", "N_anion", "mu_anion"),
                                         ("cation", "N_cation", "mu_cation")):  # fmt: skip
        mask = np.zeros(NP + 2, dtype=bool)
        fixed = np.zeros(NP + 2)
        ion_total = 0.0
        for j, ly in enumerate(layers):
            conc, mob = ly.f(conc_name), ly.f(mob_name)
            if conc == 0:
                continue
            if mob == 0:
                fixed[i0[j] : i1[j] + 1] = conc
                continue
            if mask.any():
                raise NotImplementedError(f"mobile {species}s in more than one layer")
            mask[i0[j] : i1[j] + 1] = True
            ion_total += ly.f("L") * conc
        ions[species] = (mask, ion_total, fixed)

    dos = Vt * np.log(N_c / N_c[0])
    W_L, W_R = setup.f("W_L"), setup.f("W_R")
    if setup.i("leftElec") != -1:
        raise NotImplementedError("leftElec = 1 (the left electrode as anode) is not in Stage A")
    return Device(
        x=x, h=h, L=total, T=T, Vt=Vt,
        eps=per_point("eps_r") * EPS_0, E_c=E_c, E_v=E_v, N_c=N_c, ni=ni,
        N_D=per_point("N_D"), N_A=per_point("N_A"),
        mu_n=interleaved("mu_n", "nu_int_n"), mu_p=interleaved("mu_p", "nu_int_p"),
        k_direct=k_direct,
        srh_Nt=srh["Nt"], srh_Cn=srh["Cn"], srh_Cp=srh["Cp"],
        srh_nt0=srh["nt0"], srh_pt0=srh["pt0"],
        G=G,
        offset_n=E_c - E_c[0] + dos, offset_p=E_v - E_v[0] - dos,
        V0=0.5 * (-W_L + W_R), VL=0.5 * (W_L - W_R),
        n_left=float(N_c[0] * np.exp(-(W_L - E_c[0]) / Vt)),
        n_right=float(N_c[-1] * np.exp(-(W_R - E_c[-1]) / Vt)),
        p_left=float(N_c[0] * np.exp((W_L - E_v[0]) / Vt)),
        p_right=float(N_c[-1] * np.exp(-(E_v[-1] - W_R) / Vt)),
        W_L=W_L, W_R=W_R, G_frac=setup.f("G_frac"),
        i0=tuple(i0), i1=tuple(i1),
        anion_mask=ions["anion"][0], cation_mask=ions["cation"][0],
        anion_total=ions["anion"][1], cation_total=ions["cation"][1],
        anion_fixed=ions["anion"][2], cation_fixed=ions["cation"][2],
        pre_cond=bool(setup.i("preCond")), V_pre=setup.f("Vpre"),
        fix_ions=bool(setup.i("fixIons")),
    )  # fmt: skip
