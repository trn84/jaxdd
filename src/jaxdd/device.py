"""A SIMsalabim device on SIMsalabim's grid: the static arrays the solver reads.

Follows SIMsalabim 5.36 (`Units/DDRoutines.pas`: `Make_Grid`, `Define_Layers`,
`Update_Gen_Pot`, `Init_Generation_Profile`, `Init_Trapping`, `Calc_Elec_Mob`), so that both
solve the same discrete equations.

A device is split in two (`device_parameters`): its **structure** — the grid's points per layer,
which layers generate light, hold traps or mobile ions, the pre-bias sequence — and its
**parameters**, the numbers (thicknesses, energies, densities, mobilities, …). One set of
formulas (`device_arrays`) makes the arrays from both, in NumPy (`build_device`: once per device,
then handed to JAX) or in `jax.numpy` (`batch.device_tree`: differentiable in the parameters).
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from jaxdd.constants import EPS_0, K_B, Q
from jaxdd.simss_input import Layer, Setup

__all__ = [
    "GRID_SPLITS",
    "Device",
    "Structure",
    "build_device",
    "device_arrays",
    "device_parameters",
    "layer_points",
    "make_grid",
]

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


def _sub_grid(m: int, grad: float) -> np.ndarray:
    """SIMsalabim's `Make_Sub_Grid` before its scaling: an exponential grid of `m + 1` points,
    symmetric in the layer."""
    h = np.ones(m + 1)
    ending = round(0.5 * m)
    for i in range(ending + 1):
        h[i] = np.exp(grad * (1.0 / ending - 1.0 / (i + 1)))
        h[m - i] = h[i]
    return h


def _layout(thicknesses: Sequence[str | float], NP: int, grad: float, split: str):
    """What `make_grid` takes from the thicknesses that is not smooth in them: the points per
    layer, `i0`, `i1`, `lid`, and each point's unscaled spacing with its layer's sum."""
    counts = layer_points(thicknesses, NP, split)
    n_layers = len(counts)
    raw = np.zeros(NP + 2)
    norm = np.zeros(n_layers)
    lid = np.zeros(NP + 2, dtype=int)
    i0, i1 = [], []
    istart = 0
    for k in range(n_layers):
        ifinish = istart + counts[k]
        i0.append(istart)
        i1.append(ifinish)
        if k == n_layers - 1:
            ifinish = NP + 1
        raw[istart : ifinish + 1] = _sub_grid(ifinish - istart, grad)
        norm[k] = raw[istart : ifinish + 1].sum()
        lid[istart : ifinish + 1] = k
        istart = ifinish + 1
    i1[-1] = NP + 1
    return i0, i1, lid, raw, norm


def _set(xp, a, idx, value):
    """`a` with `a[idx] = value`, in NumPy (a copy) or JAX."""
    if xp is np:
        a = a.copy()
        a[idx] = value
        return a
    return a.at[idx].set(value)


def _spacing(L, total, i0, lid, raw, norm, xp=np):
    """`(x, h)` on a layout (`_layout`) for the thicknesses `L` (per layer) and their sum: each
    layer's spacing scaled to its thickness, equalised across each interface (smooth in `L`)."""
    h = raw * (L[lid] / (total * norm[lid]))
    for j in i0[1:]:
        h = _set(xp, h, slice(j - 2, j + 1), (h[j] + h[j - 1] + h[j - 2]) / 3.0)
    x = xp.concatenate([xp.zeros(1), xp.cumsum(total * h[:-1])])
    return x, h


def make_grid(thicknesses: Sequence[str | float], NP: int, grad: float, split: str = "exact"):
    """`(x, h, i0, i1, lid)` as SIMsalabim's `Make_Grid`: `x` has `NP + 2` points, `h` is the
    spacing over the total length `L` (`h[i] = (x[i+1] − x[i])/L`), `i0`/`i1` each layer's first
    and last point, `lid` the layer of each point (0-based). The thicknesses in metres, as
    numbers or as the input's decimal strings; `split` as in `layer_points`.

    `L` is the sum of the layers, and `x[NP+1]` is not quite it: equalising the spacing
    across each interface leaves `Σh` slightly off 1 (0.006 % on a 1/500/1 nm stack).
    SIMsalabim uses `L` wherever it scales, so does this."""
    i0, i1, lid, raw, norm = _layout(thicknesses, NP, grad, split)
    L = np.array([float(L) for L in thicknesses])
    x, h = _spacing(L, _sum(L), i0, lid, raw, norm)
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
    # Interface traps (SIMsalabim's `N_t_int` of a layer: at its interface to the next), one
    # level each, on the `NP + 1` edges: edge `e` lies between points `e` and `e + 1`, and an
    # interface between the last point of a layer and the first of the next. None without any.
    # `Nt` m⁻², `Cn`, `Cp` m³/s, `nt0_lo`/`pt0_lo` (`nt0_hi`/`pt0_hi`) SRH's densities of the
    # layer on the edge's left (right), `iw` the inverse of the interface's width, `charge`
    # 1 for a charged type (donor or acceptor), `cwe` its charge when empty (1 a donor).
    interface: dict | None = None
    # The external circuit (SimSS's: `J_ext = J + V_int/R_shunt`, `V_ext = V_int + J_ext·R_series`),
    # Ω m²; `R_shunt` ≤ 0 none.
    R_series: float = 0.0
    R_shunt: float = -1.0


@dataclass(frozen=True)
class Structure:
    """What a device's parameters do not change: the grid's layout (`_layout`: points per layer,
    each point's layer and unscaled spacing), which layers generate light, hold traps, or hold
    mobile or fixed ions, and the pre-bias sequence. Devices of one structure batch by `vmap`
    over their parameters (`batch.device_tree`)."""

    NP: int
    i0: tuple[int, ...]
    i1: tuple[int, ...]
    lid: np.ndarray
    raw: np.ndarray
    norm: np.ndarray
    layer_gen: np.ndarray  # per layer
    traps: np.ndarray  # per layer: bulk traps
    anion_mobile: np.ndarray  # per layer: N > 0 and μ > 0
    anion_fixed: np.ndarray  # per layer: N > 0 and μ = 0
    cation_mobile: np.ndarray
    cation_fixed: np.ndarray
    pre_cond: bool
    fix_ions: bool
    int_traps: np.ndarray = None  # per layer: traps at its interface to the next (not the last)
    int_trap_type: np.ndarray = None  # per layer: -1 acceptor, 0 neutral, 1 donor


#: A device's parameters per layer (`device_parameters`), as SIMsalabim names them.
LAYER_PARAMETERS = (
    "L", "eps_r", "E_c", "E_v", "N_c", "N_D", "N_A", "mu_n", "mu_p", "nu_int_n", "nu_int_p",
    "k_direct", "N_t_bulk", "C_n_bulk", "C_p_bulk", "E_t_bulk", "N_anion", "N_cation",
    "N_t_int", "E_t_int", "C_n_int", "C_p_int",
)  # fmt: skip


def device_parameters(setup: Setup, grid_split: str = "exact") -> tuple[Structure, dict]:
    """A SIMsalabim input (Stage A, `check_supported` first) as its `Structure` and its
    parameters: a dict of `LAYER_PARAMETERS`, each an array over the layers, and `T`, `W_L`,
    `W_R`, `G_frac`, `V_pre`, the generation profile's file (`gen_x`, `gen_G`) and the external
    circuit's `R_series` and `R_shunt` (Ω m², `R_shunt` ≤ 0 none). `grid_split`
    as in `build_device`. Changing a parameter keeps the structure: a trap density set on a
    layer without traps, or an ion density on a layer without ions, does nothing; the points
    per layer stay those of the input's thicknesses."""
    layers = setup.layers
    if setup.i("leftElec") != -1:
        raise NotImplementedError("leftElec = 1 (the left electrode as anode) is not in Stage A")
    NP, grad = setup.i("NP"), setup.f("grad")
    i0, i1, lid, raw, norm = _layout([ly.raw["L"] for ly in layers], NP, grad, grid_split)

    def flags(test):
        return np.array([bool(test(ly)) for ly in layers])

    structure = Structure(
        NP=NP, i0=tuple(i0), i1=tuple(i1), lid=lid, raw=raw, norm=norm,
        layer_gen=flags(lambda ly: ly.i("layerGen")),
        traps=flags(lambda ly: ly.f("N_t_bulk") != 0),
        anion_mobile=flags(lambda ly: ly.f("N_anion") != 0 and ly.f("mu_anion") != 0),
        anion_fixed=flags(lambda ly: ly.f("N_anion") != 0 and ly.f("mu_anion") == 0),
        cation_mobile=flags(lambda ly: ly.f("N_cation") != 0 and ly.f("mu_cation") != 0),
        cation_fixed=flags(lambda ly: ly.f("N_cation") != 0 and ly.f("mu_cation") == 0),
        pre_cond=bool(setup.i("preCond")), fix_ions=bool(setup.i("fixIons")),
        int_traps=np.array([float(ly.raw.get("N_t_int", "0")) != 0 for ly in layers[:-1]]
                           + [False]),
        int_trap_type=np.array([int(float(ly.raw.get("intTrapType", "0"))) for ly in layers]),
    )  # fmt: skip
    for j in np.flatnonzero(structure.int_traps):
        lo, hi = layers[j], layers[j + 1]
        E_t = lo.f("E_t_int")
        if not max(lo.f("E_c"), hi.f("E_c")) < E_t < min(lo.f("E_v"), hi.f("E_v")):
            raise ValueError(f"interface trap between layers {j + 1} and {j + 2} at {E_t} eV: "
                             "not within the gaps of both")  # fmt: skip
        if structure.int_trap_type[j] not in (-1, 0, 1):
            raise ValueError(f"layer {j + 1}: intTrapType {structure.int_trap_type[j]}")
    for species in ("anion", "cation"):
        if getattr(structure, f"{species}_mobile").sum() > 1:
            raise NotImplementedError(f"mobile {species}s in more than one layer")
    table = np.loadtxt(io.StringIO(setup.text("genProfile")), skiprows=1)
    # Needed only on some layers: the interface velocities (not on the last), the trap's.
    optional = {
        "nu_int_n",
        "nu_int_p",
        "C_n_bulk",
        "C_p_bulk",
        "E_t_bulk",
        "N_t_int",
        "E_t_int",
        "C_n_int",
        "C_p_int",
    }

    def value(ly: Layer, name: str) -> float:
        return float(ly.raw.get(name, "0")) if name in optional else ly.f(name)

    params = {name: np.array([value(ly, name) for ly in layers]) for name in LAYER_PARAMETERS}
    params.update(
        T=setup.f("T"), W_L=setup.f("W_L"), W_R=setup.f("W_R"), G_frac=setup.f("G_frac"),
        V_pre=setup.f("Vpre"), gen_x=table[:, 0].copy(), gen_G=table[:, 1].copy(),
        R_series=float(setup.raw.get("R_series", "0")),
        R_shunt=float(setup.raw.get("R_shunt", "-1")),
    )  # fmt: skip
    return structure, params


def _generation(gen_x, gen_G, x, lid, layer_gen, total, xp=np):
    """SIMsalabim's `Init_Generation_Profile`, `Use_gen_profile` 2: the file's profile, its
    x-axis stretched to the device, interpolated linearly; zero where a layer does not
    generate and on any point the interpolation loop does not reach (at or beyond the
    stretched axis's end; the electrodes are zeroed in `device_arrays`). Each point takes the
    first interval whose end lies beyond it, as the loop does."""
    a = gen_x * (total / gen_x[-1])
    j = xp.searchsorted(a[1:], x, side="right")
    jj = xp.minimum(j, len(a) - 2)
    g = gen_G[jj] + (x - a[jj]) * (gen_G[jj + 1] - gen_G[jj]) / (a[jj + 1] - a[jj])
    return xp.where((j < len(a) - 1) & layer_gen[lid], g, 0.0)


def _sum(values, xp=np):
    """`sum(values)` as Python (3.12 and later) sums floats, compensated (Neumaier), so that the
    total thickness is the one `make_grid` takes, to the bit; in JAX too."""
    total, comp = 0.0, 0.0
    for v in values:
        s = total + v
        comp = comp + xp.where(abs(total) >= abs(v), (total - s) + v, (v - s) + total)
        total = s
    return total + comp


def _interface(st: Structure, p: dict, h, total, Vt, xp=np) -> dict:
    """`Device.interface` (SIMsalabim's `Init_Trapping` and `Calc_Trap_Filling_Charge`): each
    layer's interface traps on the edge after its last point."""
    j = np.flatnonzero(st.int_traps)  # layers with traps at their right interface
    e = np.array(st.i1)[j]  # their edges
    E_t = p["E_t_int"][j]

    def nt0(k):
        return p["N_c"][k] * xp.exp((p["E_c"][k] - E_t) / Vt)

    def pt0(k):
        return p["N_c"][k] * xp.exp((E_t - p["E_v"][k]) / Vt)

    width = 0.5 * total * (0.5 * h[e + 1] + h[e] + 0.5 * h[e - 1])
    kind = st.int_trap_type[j]
    edges = xp.zeros(st.NP + 1)

    def on_edges(values):
        return _set(xp, edges, e, values)

    return {
        "Nt": on_edges(p["N_t_int"][j]), "Cn": on_edges(p["C_n_int"][j]),
        "Cp": on_edges(p["C_p_int"][j]),
        "nt0_lo": on_edges(nt0(j)), "pt0_lo": on_edges(pt0(j)),
        "nt0_hi": on_edges(nt0(j + 1)), "pt0_hi": on_edges(pt0(j + 1)),
        "iw": on_edges(1.0 / width),
        "charge": on_edges(np.abs(kind).astype(float)),
        "cwe": on_edges((kind == 1).astype(float)),
    }  # fmt: skip


def device_arrays(structure: Structure, p: dict, xp=np) -> Device:
    """Every array of a device from its structure and parameters (`device_parameters`), in
    NumPy (`xp=np`) or `jax.numpy`, by the same formulas: in JAX differentiable in `p`."""
    st = structure
    NP, lid, i0, i1 = st.NP, st.lid, st.i0, st.i1
    total = _sum(p["L"], xp)
    x, h = _spacing(p["L"], total, i0, lid, st.raw, st.norm, xp)
    T = p["T"]
    Vt = K_B * T / Q
    interior = np.zeros(NP + 2, dtype=bool)
    interior[1:-1] = True

    E_c, E_v, N_c = p["E_c"][lid], p["E_v"][lid], p["N_c"][lid]
    ni = N_c * xp.exp(-0.5 * (E_v - E_c) / Vt)
    k_direct = xp.where(interior, p["k_direct"][lid], 0.0)

    # Interface mobilities: on the last point of each layer but the last, the interval to the
    # next layer, `L·h·ν_int/V_T`.
    iface = np.zeros(NP + 2, dtype=bool)
    iface[list(i1[:-1])] = True

    def interleaved(name: str, nu_name: str):
        return xp.where(iface, (total * h) * p[nu_name][lid] / Vt, p[name][lid])

    trap = interior & st.traps[lid]
    nt0 = p["N_c"] * xp.exp((p["E_c"] - p["E_t_bulk"]) / Vt)
    pt0 = p["N_c"] * xp.exp((p["E_t_bulk"] - p["E_v"]) / Vt)
    srh = {key: xp.where(trap, v[lid], 0.0) for key, v in (
        ("Nt", p["N_t_bulk"]), ("Cn", p["C_n_bulk"]), ("Cp", p["C_p_bulk"]),
        ("nt0", nt0), ("pt0", pt0))}  # fmt: skip

    G = _generation(p["gen_x"], p["gen_G"], x, lid, st.layer_gen, total, xp)
    G = xp.where(interior, G, 0.0)

    ions = {}
    for species in ("anion", "cation"):
        conc = p[f"N_{species}"]
        mobile, fixed = getattr(st, f"{species}_mobile"), getattr(st, f"{species}_fixed")
        ions[species] = (
            mobile[lid],
            xp.sum(xp.where(mobile, p["L"] * conc, 0.0)),
            xp.where(fixed[lid], conc[lid], 0.0),
        )

    dos = Vt * xp.log(N_c / N_c[0])
    W_L, W_R = p["W_L"], p["W_R"]
    interface = None
    if st.int_traps is not None and st.int_traps.any():
        interface = _interface(st, p, h, total, Vt, xp)
    return Device(
        x=x, h=h, L=total, T=T, Vt=Vt,
        eps=p["eps_r"][lid] * EPS_0, E_c=E_c, E_v=E_v, N_c=N_c, ni=ni,
        N_D=p["N_D"][lid], N_A=p["N_A"][lid],
        mu_n=interleaved("mu_n", "nu_int_n"), mu_p=interleaved("mu_p", "nu_int_p"),
        k_direct=k_direct,
        srh_Nt=srh["Nt"], srh_Cn=srh["Cn"], srh_Cp=srh["Cp"],
        srh_nt0=srh["nt0"], srh_pt0=srh["pt0"],
        G=G,
        offset_n=E_c - E_c[0] + dos, offset_p=E_v - E_v[0] - dos,
        V0=0.5 * (-W_L + W_R), VL=0.5 * (W_L - W_R),
        n_left=N_c[0] * xp.exp(-(W_L - E_c[0]) / Vt),
        n_right=N_c[-1] * xp.exp(-(W_R - E_c[-1]) / Vt),
        p_left=N_c[0] * xp.exp((W_L - E_v[0]) / Vt),
        p_right=N_c[-1] * xp.exp(-(E_v[-1] - W_R) / Vt),
        W_L=W_L, W_R=W_R, G_frac=p["G_frac"],
        i0=tuple(i0), i1=tuple(i1),
        anion_mask=ions["anion"][0], cation_mask=ions["cation"][0],
        anion_total=ions["anion"][1], cation_total=ions["cation"][1],
        anion_fixed=ions["anion"][2], cation_fixed=ions["cation"][2],
        pre_cond=st.pre_cond, V_pre=p["V_pre"], fix_ions=st.fix_ions, interface=interface,
        R_series=p.get("R_series", 0.0), R_shunt=p.get("R_shunt", -1.0),
    )  # fmt: skip


def build_device(setup: Setup, grid_split: str = "exact") -> Device:
    """The device of a SIMsalabim input (Stage A, `check_supported` first), in NumPy. `grid_split`
    "simss" shares the grid points among the layers as SimSS's 10-byte reals do
    (`layer_points`; docs/differences.md §5), for a comparison point by point."""
    return device_arrays(*device_parameters(setup, grid_split))
