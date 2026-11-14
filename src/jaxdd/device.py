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

import dataclasses
import io
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from jaxdd.constants import EPS_0, K_B, Q
from jaxdd.optics import generation as optics_generation
from jaxdd.optics import optical_data
from jaxdd.simss_input import Layer, Setup

__all__ = [
    "GRID_SPLITS",
    "LANGEVIN_MOBILITIES",
    "SURFACE_MODELS",
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
    # Bulk traps (SRH), 0 outside layers with traps and on the electrodes. One level: `srh_Nt`,
    # `srh_nt0`, `srh_pt0` over the points; several (a trap file): points × levels, a level of
    # density 0 none.
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
    # level each (or several from a trap file: `Nt` and the SRH densities edges × levels), on
    # the `NP + 1` edges: edge `e` lies between points `e` and `e + 1`, and an
    # interface between the last point of a layer and the first of the next. None without any.
    # `Nt` m⁻², `Cn`, `Cp` m³/s, `nt0_lo`/`pt0_lo` (`nt0_hi`/`pt0_hi`) SRH's densities of the
    # layer on the edge's left (right), `iw` the inverse of the interface's width, `charge`
    # 1 for a charged type (donor or acceptor), `cwe` its charge when empty (1 a donor).
    interface: dict | None = None
    # The external circuit (SimSS's: `J_ext = J + V_int/R_shunt`, `V_ext = V_int + J_ext·R_series`),
    # Ω m²; `R_shunt` ≤ 0 none.
    R_series: float = 0.0
    R_shunt: float = -1.0
    left_elec: int = -1  # SIMsalabim's `leftElec`: −1 the left electrode the cathode, 1 the anode
    # Charged bulk traps (`bulkTrapType` ±1), shaped as `srh_Nt`: each level's density where it
    # is charged (else 0), and its charge when empty (1 a donor, 0 an acceptor). None without.
    trap_charge: np.ndarray | None = None
    trap_cwe: np.ndarray | None = None
    # The field-dependent mobility (SIMsalabim's `mobnDep`, `mobpDep` = 1, Poole–Frenkel):
    # μ = μ₀·exp(γ·√|F|) on each interval, `F` its field; here `(k_n, k_p)` on the interleaved
    # mesh, `k = γ/√(L·h)` (so `γ·√|F| = k·√|ΔV|`), 0 at the interfaces (their transfer mobility)
    # and in layers with a constant mobility. None without any.
    mob_field: tuple[np.ndarray, np.ndarray] | None = None
    # Finite surface recombination at the contacts (SIMsalabim's `S_n_L`, `S_p_L`, `S_n_R`,
    # `S_p_R`, m/s, negative for infinite): `S` the four, `a` the first and last interval's
    # `V_T·μ/(L·h)` for electrons and holes in the same order, `both` 1 where the condition holds
    # in both directions (`surface = "robin"`), 0 only for extraction (SimSS's manual). None
    # where every one is infinite.
    surface: dict | None = None


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
    bulk_trap_type: np.ndarray = None  # per layer: -1 acceptor, 0 neutral, 1 donor
    bulk_trap_file: np.ndarray = None  # per layer: the levels from `bulkTrapFile`
    int_trap_file: np.ndarray = None  # per layer: the levels from `intTrapFile` (not the last)
    #: The contacts with finite surface recombination: None (all infinite), "extraction" or
    #: "robin" (`SURFACE_MODELS`).
    surface: str | None = None
    sfb: tuple[bool, bool] = (False, False)  # `W_L`, `W_R` = 'sfb': from the adjacent layer
    langevin: np.ndarray = None  # per layer: `useLangevin`, `k_direct` from the mobilities
    langevin_mobility: str = "layer"  # `LANGEVIN_MOBILITIES`
    mob_dep: np.ndarray = None  # per layer × (electrons, holes): `mobnDep`, `mobpDep` = 1
    #: The generation: "file" (a profile, `gen_x`, `gen_G`), "none" (uniform, each layer's
    #: `G_ehp`) or "calc" (the transfer-matrix method, `optics.optical_data`).
    gen_mode: str = "file"
    optics: dict | None = None
    left_elec: int = -1


#: How finite surface recombination at a contact (`S_n_L`, …, ≥ 0) acts: "extraction" as
#: SIMsalabim's manual states it, `J·n̂ = q·S·(n − n_eq)` while the carrier is extracted there
#: and the density `n_eq` while it is injected; "robin" the same condition in both directions
#: (thermionic emission at a Schottky contact). Both with the physical sign at all four ends,
#: which SimSS 5.36 has at two of them (docs/differences.md).
SURFACE_MODELS = ("extraction", "robin")

#: The mobilities in the Langevin rate `preLangevin·q·(μ_n + μ_p)/ε` at a point: "layer" its
#: layer's; "simss" those of the interval after it, as SimSS 5.36 takes them, which at the last
#: point of each layer but the last is the interface's transfer mobility `L·h·ν_int/V_T`
#: (docs/differences.md §8).
LANGEVIN_MOBILITIES = ("layer", "simss")

#: The surface recombination velocities, in `S` order (`Device.surface`).
SURFACE_PARAMETERS = ("S_n_L", "S_p_L", "S_n_R", "S_p_R")

#: A device's parameters per layer (`device_parameters`), as SIMsalabim names them.
LAYER_PARAMETERS = (
    "L", "eps_r", "E_c", "E_v", "N_c", "N_D", "N_A", "mu_n", "mu_p", "nu_int_n", "nu_int_p",
    "gamma_n", "gamma_p", "k_direct", "preLangevin", "N_t_bulk", "C_n_bulk", "C_p_bulk",
    "E_t_bulk", "N_anion", "N_cation", "N_t_int", "E_t_int", "C_n_int", "C_p_int", "G_ehp",
)  # fmt: skip


def _named(layer: Layer, name: str) -> bool:
    """Whether a layer's file parameter (`bulkTrapFile`, `intTrapFile`) names a file."""
    return layer.raw.get(name, "none").strip().lower() != "none"


def _trap_levels(text: str, what: str) -> tuple[np.ndarray, np.ndarray]:
    """A trap file's levels (SIMsalabim's `Init_Traps_From_File`): a header (`E Ntrap`), then
    each level's energy (eV) and density (m⁻³ in the bulk, m⁻² at an interface), absolute."""
    table = np.atleast_2d(np.loadtxt(io.StringIO(text), skiprows=1))
    if table.size == 0:
        raise ValueError(f"{what}: no trap levels")
    E, N = table[:, 0].copy(), table[:, 1].copy()
    if (N == 0).any():
        raise ValueError(f"{what}: a trap level of zero density (level {np.argmin(N != 0) + 1})")
    return E, N


def _level_table(layers, files, single_E, read, kind):
    """Each layer's trap levels from its file, padded to the most any file has: `(E, N)`,
    layers × levels, a padded level of density 0 at the file's first energy; zero rows for the
    layers without a file (their one level is the parameter file's, `device_arrays`)."""
    levels = {k: _trap_levels(read(layers[k], k), f"layer {k + 1}: {kind}")
              for k in np.flatnonzero(files)}  # fmt: skip
    K = max(len(E) for E, _ in levels.values())
    E_tab = np.repeat(np.asarray(single_E, dtype=float)[:, None], K, axis=1)
    N_tab = np.zeros((len(layers), K))
    for k, (E, N) in levels.items():
        E_tab[k] = E[0]
        E_tab[k, : len(E)], N_tab[k, : len(N)] = E, N
    return E_tab, N_tab


def _sfb(N_A, N_D, E_c, E_v, N_c, Vt, xp=np):
    """SIMsalabim's `ObtainWorkFunction` for 'sfb' (semi-flat band): the adjacent layer's
    equilibrium Fermi level from its net doping, mid-gap where there is none."""
    net = N_A - N_D
    safe = xp.where(net == 0, 1.0, xp.abs(net))
    n_type = xp.where(net < 0, E_c + Vt * xp.log(N_c / safe), 0.5 * (E_c + E_v))
    return xp.where(net > 0, E_v - Vt * xp.log(N_c / safe), n_type)


def device_parameters(setup: Setup, grid_split: str = "exact", surface: str = "extraction",
                      langevin: str = "layer") -> tuple[Structure, dict]:  # fmt: skip
    """A SIMsalabim input (Stage A, `check_supported` first) as its `Structure` and its
    parameters: a dict of `LAYER_PARAMETERS`, each an array over the layers, and `T`, `W_L`,
    `W_R`, `G_frac`, `V_pre`, the generation profile's file (`gen_x`, `gen_G`) and the external
    circuit's `R_series` and `R_shunt` (Ω m², `R_shunt` ≤ 0 none). Trap files add their levels,
    layers × levels: `bulk_trap_E`, `bulk_trap_N` (eV, m⁻³) and `int_trap_E`, `int_trap_N`
    (m⁻²), used on the layers whose file names them; the surface recombination velocities
    `S_n_L`, `S_p_L`, `S_n_R`, `S_p_R` (m/s, negative infinite), acting as `surface` says
    (`SURFACE_MODELS`); `langevin` which mobilities the Langevin rate takes
    (`LANGEVIN_MOBILITIES`). `grid_split`
    as in `build_device`. Changing a parameter keeps the structure: a trap density set on a
    layer without traps, or an ion density on a layer without ions, does nothing; the points
    per layer stay those of the input's thicknesses."""
    layers = setup.layers
    if surface not in SURFACE_MODELS:
        raise ValueError(f"surface {surface!r}: one of {SURFACE_MODELS}")
    if langevin not in LANGEVIN_MOBILITIES:
        raise ValueError(f"langevin {langevin!r}: one of {LANGEVIN_MOBILITIES}")
    if setup.i("leftElec") not in (-1, 1):
        raise ValueError("leftElec must be -1 (the left electrode the cathode) or 1 (the anode)")
    NP, grad = setup.i("NP"), setup.f("grad")
    i0, i1, lid, raw, norm = _layout([ly.raw["L"] for ly in layers], NP, grad, grid_split)

    def flags(test):
        return np.array([bool(test(ly)) for ly in layers])

    bulk_file = flags(lambda ly: _named(ly, "bulkTrapFile"))
    int_file = np.array([_named(ly, "intTrapFile") for ly in layers[:-1]] + [False])

    structure = Structure(
        NP=NP, i0=tuple(i0), i1=tuple(i1), lid=lid, raw=raw, norm=norm,
        layer_gen=flags(lambda ly: ly.i("layerGen")),
        traps=flags(lambda ly: ly.f("N_t_bulk") != 0) | bulk_file,
        anion_mobile=flags(lambda ly: ly.f("N_anion") != 0 and ly.f("mu_anion") != 0),
        anion_fixed=flags(lambda ly: ly.f("N_anion") != 0 and ly.f("mu_anion") == 0),
        cation_mobile=flags(lambda ly: ly.f("N_cation") != 0 and ly.f("mu_cation") != 0),
        cation_fixed=flags(lambda ly: ly.f("N_cation") != 0 and ly.f("mu_cation") == 0),
        pre_cond=bool(setup.i("preCond")), fix_ions=bool(setup.i("fixIons")),
        int_traps=np.array([float(ly.raw.get("N_t_int", "0")) != 0 for ly in layers[:-1]]
                           + [False]) | int_file,
        int_trap_type=np.array([int(float(ly.raw.get("intTrapType", "0"))) for ly in layers]),
        bulk_trap_type=np.array([int(float(ly.raw.get("bulkTrapType", "0"))) for ly in layers]),
        bulk_trap_file=bulk_file, int_trap_file=int_file,
        surface=surface if any(float(setup.raw.get(k, "-1")) >= 0 for k in SURFACE_PARAMETERS)
        else None,
        sfb=tuple(setup.raw[k].strip().lower() == "sfb" for k in ("W_L", "W_R")),
        langevin=flags(lambda ly: int(float(ly.raw.get("useLangevin", "0"))) == 1),
        langevin_mobility=langevin,
        mob_dep=np.array([[int(float(ly.raw.get(k, "0"))) for k in ("mobnDep", "mobpDep")]
                          for ly in layers]),
        left_elec=setup.i("leftElec"),
    )  # fmt: skip
    # SIMsalabim's checks: each work function within its layer's gap.
    for k, ly, sfb in (("W_L", layers[0], structure.sfb[0]), ("W_R", layers[-1], structure.sfb[1])):
        if not sfb and not ly.f("E_c") <= setup.f(k) <= ly.f("E_v"):
            raise ValueError(f"{k} = {setup.f(k)} eV: outside the gap of its layer "
                             f"({ly.f('E_c')} to {ly.f('E_v')} eV)")  # fmt: skip
    # SIMsalabim's: `mobnDep`, `mobpDep` 0 or 1. With Langevin the rate would follow the field
    # too (SimSS takes the interval's mobility): not yet.
    if not np.isin(structure.mob_dep, (0, 1)).all():
        raise ValueError(f"mobnDep, mobpDep: 0 or 1, got {structure.mob_dep.tolist()}")
    both = structure.langevin & structure.mob_dep.any(axis=1)
    if both.any():
        raise NotImplementedError(f"layer {np.argmax(both) + 1}: useLangevin with a "
                                  "field-dependent mobility")  # fmt: skip
    extra = {}
    if bulk_file.any():
        extra["bulk_trap_E"], extra["bulk_trap_N"] = _level_table(
            layers, bulk_file, [float(ly.raw.get("E_t_bulk", "0")) for ly in layers],
            lambda ly, k: setup.read(ly.raw["bulkTrapFile"].strip(), "bulkTrapFile"),
            "bulkTrapFile")  # fmt: skip
    if int_file.any():
        extra["int_trap_E"], extra["int_trap_N"] = _level_table(
            layers, int_file, [float(ly.raw.get("E_t_int", "0")) for ly in layers],
            lambda ly, k: setup.read(ly.raw["intTrapFile"].strip(), "intTrapFile"),
            "intTrapFile")  # fmt: skip

    def levels(k, file, kind):  # a layer's levels (energies of non-zero density) and their name
        if file[k]:
            E, N = extra[f"{kind}_trap_E"][k], extra[f"{kind}_trap_N"][k]
            return E[N != 0]
        return np.array([float(layers[k].raw.get(f"E_t_{kind}", "0"))])

    # SIMsalabim's checks (`Init_Trapping`): every level within its layer's gap, an interface's
    # within both layers' gaps (strictly).
    for j in np.flatnonzero(structure.traps):
        ly = layers[j]
        e_t = levels(j, bulk_file, "bulk")
        if not ((e_t > ly.f("E_c")) & (e_t < ly.f("E_v"))).all():
            raise ValueError(f"layer {j + 1}: a bulk trap level outside the gap ({e_t} eV)")
        if structure.bulk_trap_type[j] not in (-1, 0, 1):
            raise ValueError(f"layer {j + 1}: bulkTrapType {structure.bulk_trap_type[j]}")
    for j in np.flatnonzero(structure.int_traps):
        lo, hi = layers[j], layers[j + 1]
        e_t = levels(j, int_file, "int")
        gap = max(lo.f("E_c"), hi.f("E_c")), min(lo.f("E_v"), hi.f("E_v"))
        if not ((e_t > gap[0]) & (e_t < gap[1])).all():
            raise ValueError(f"interface trap between layers {j + 1} and {j + 2} at {e_t} eV: "
                             "not within the gaps of both")  # fmt: skip
        if structure.int_trap_type[j] not in (-1, 0, 1):
            raise ValueError(f"layer {j + 1}: intTrapType {structure.int_trap_type[j]}")
    for species in ("anion", "cation"):
        if getattr(structure, f"{species}_mobile").sum() > 1:
            raise NotImplementedError(f"mobile {species}s in more than one layer")
    mode = {"none": "none", "calc": "calc"}.get(setup.raw.get("genProfile", "none").lower(), "file")
    structure = dataclasses.replace(
        structure, gen_mode=mode, optics=optical_data(setup, layers) if mode == "calc" else None
    )
    table = (np.loadtxt(io.StringIO(setup.text("genProfile")), skiprows=1) if mode == "file"
             else np.zeros((2, 2)))  # fmt: skip
    # Needed only on some layers: the interface velocities (not on the last), the trap's.
    optional = {
        "nu_int_n",
        "nu_int_p",
        "gamma_n",
        "gamma_p",
        "C_n_bulk",
        "C_p_bulk",
        "E_t_bulk",
        "N_t_int",
        "E_t_int",
        "C_n_int",
        "C_p_int",
        "G_ehp",
    }

    def value(ly: Layer, name: str) -> float:
        if name == "preLangevin":
            return float(ly.raw.get(name, "1"))
        return float(ly.raw.get(name, "0")) if name in optional else ly.f(name)

    params = {name: np.array([value(ly, name) for ly in layers]) for name in LAYER_PARAMETERS}
    params.update(
        T=setup.f("T"), G_frac=setup.f("G_frac"),
        # 'sfb' (`Structure.sfb`) is computed in `device_arrays`; the value here is unused.
        W_L=0.0 if structure.sfb[0] else setup.f("W_L"),
        W_R=0.0 if structure.sfb[1] else setup.f("W_R"),
        **{k: float(setup.raw.get(k, "-1")) for k in SURFACE_PARAMETERS},
        V_pre=setup.f("Vpre"), gen_x=table[:, 0].copy(), gen_G=table[:, 1].copy(),
        R_series=float(setup.raw.get("R_series", "0")),
        R_shunt=float(setup.raw.get("R_shunt", "-1")),
        **extra,
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
    E_t, N_t = p["E_t_int"][j], p["N_t_int"][j]
    levels = st.int_trap_file is not None and st.int_trap_file.any()
    if levels:  # interfaces × levels, from a trap file
        E_t, N_t = _levels(
            st.int_trap_file[j], p["int_trap_E"][j], p["int_trap_N"][j], E_t, N_t, xp
        )

    def col(v):  # per interface, or a column against the levels
        return v[:, None] if levels else v

    def nt0(k):
        return col(p["N_c"][k]) * xp.exp((col(p["E_c"][k]) - E_t) / Vt)

    def pt0(k):
        return col(p["N_c"][k]) * xp.exp((E_t - col(p["E_v"][k])) / Vt)

    width = 0.5 * total * (0.5 * h[e + 1] + h[e] + 0.5 * h[e - 1])
    kind = st.int_trap_type[j]

    def on_edges(values):
        return _set(xp, xp.zeros((st.NP + 1, *values.shape[1:])), e, values)

    return {
        "Nt": on_edges(N_t), "Cn": on_edges(p["C_n_int"][j]),
        "Cp": on_edges(p["C_p_int"][j]),
        "nt0_lo": on_edges(nt0(j)), "pt0_lo": on_edges(pt0(j)),
        "nt0_hi": on_edges(nt0(j + 1)), "pt0_hi": on_edges(pt0(j + 1)),
        "iw": on_edges(1.0 / width),
        "charge": on_edges(np.abs(kind).astype(float)),
        "cwe": on_edges((kind == 1).astype(float)),
    }  # fmt: skip


def _levels(from_file, E_file, N_file, E_one, N_one, xp=np):
    """Trap levels, rows × levels: a trap file's where `from_file`, else the one level of the
    parameter file (`E_one`, `N_one`) in the first column, the others of density 0 at its energy."""
    first = np.arange(E_file.shape[1]) == 0
    E = xp.where(from_file[:, None], E_file, E_one[:, None] + 0.0 * E_file)
    N = xp.where(from_file[:, None], N_file, xp.where(first, N_one[:, None], 0.0))
    return E, N


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

    # Interface mobilities: on the last point of each layer but the last, the interval to the
    # next layer, `L·h·ν_int/V_T`.
    iface = np.zeros(NP + 2, dtype=bool)
    iface[list(i1[:-1])] = True

    def interleaved(name: str, nu_name: str):
        return xp.where(iface, (total * h) * p[nu_name][lid] / Vt, p[name][lid])

    # Direct recombination: `k_direct`, or in layers with `useLangevin` the Langevin rate
    # (`Calc_Langevin_Factor`); none on the electrodes.
    k_direct = p["k_direct"][lid]
    if st.langevin is not None and st.langevin.any():
        if st.langevin_mobility == "simss":
            mob = interleaved("mu_n", "nu_int_n") + interleaved("mu_p", "nu_int_p")
        else:
            mob = p["mu_n"][lid] + p["mu_p"][lid]
        rate = p.get("preLangevin", xp.ones_like(p["k_direct"]))[lid] * Q * mob / (
            p["eps_r"][lid] * EPS_0)  # fmt: skip
        k_direct = xp.where(st.langevin[lid], rate, k_direct)
    k_direct = xp.where(interior, k_direct, 0.0)

    trap = interior & st.traps[lid]
    E_t, N_t, on = p["E_t_bulk"], p["N_t_bulk"], trap
    levels = st.bulk_trap_file is not None and st.bulk_trap_file.any()
    if levels:  # layers × levels, from a trap file
        E_t, N_t = _levels(st.bulk_trap_file, p["bulk_trap_E"], p["bulk_trap_N"], E_t, N_t, xp)
        on = trap[:, None]

    def col(v):  # per layer or point, or a column against the levels
        return v[:, None] if levels else v

    nt0 = col(p["N_c"]) * xp.exp((col(p["E_c"]) - E_t) / Vt)
    pt0 = col(p["N_c"]) * xp.exp((E_t - col(p["E_v"])) / Vt)
    srh = {key: xp.where(where, v[lid], 0.0) for key, v, where in (
        ("Nt", N_t, on), ("Cn", p["C_n_bulk"], trap), ("Cp", p["C_p_bulk"], trap),
        ("nt0", nt0, on), ("pt0", pt0, on))}  # fmt: skip
    trap_charge = trap_cwe = None
    kind = st.bulk_trap_type if st.bulk_trap_type is not None else np.zeros(len(st.traps), int)
    charged = st.traps & (kind != 0)
    if charged.any():  # SIMsalabim's `Calc_Trap_Filling_Charge`: |type|·(cwe − f)·N_t per level
        q = on & col(charged[lid])
        trap_charge = xp.where(q, srh["Nt"], 0.0)
        trap_cwe = xp.where(q & col(kind[lid] == 1), 1.0, 0.0) + 0.0 * srh["Nt"]

    if st.gen_mode == "calc":
        G = optics_generation(st.optics, x, lid, [p["L"][k] for k in range(len(st.layer_gen))],
                              st.layer_gen, xp)  # fmt: skip
    elif st.gen_mode == "none":
        G = p["G_ehp"][lid]  # SIMsalabim's: each layer's own, whatever its `layerGen`
    else:
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
    for side, k in ((0, 0), (1, -1)):
        if st.sfb[side]:
            W = _sfb(p["N_A"][k], p["N_D"][k], p["E_c"][k], p["E_v"][k], p["N_c"][k], Vt, xp)
            W_L, W_R = (W, W_R) if side == 0 else (W_L, W)
    surface = None
    if st.surface is not None:
        mu_n, mu_p = interleaved("mu_n", "nu_int_n"), interleaved("mu_p", "nu_int_p")
        a_L, a_R = Vt / (total * h[0]), Vt / (total * h[NP])
        surface = {
            "S": xp.stack([xp.asarray(p.get(k, -1.0), dtype=float) for k in SURFACE_PARAMETERS]),
            "a": xp.stack([a_L * mu_n[0], a_L * mu_p[0], a_R * mu_n[NP], a_R * mu_p[NP]]),
            "both": xp.asarray(1.0 if st.surface == "robin" else 0.0),
        }
    mob_field = None
    if st.mob_dep is not None and st.mob_dep.any():  # SIMsalabim's `Calc_Elec_Mob`, `Calc_Hole_Mob`
        dep = xp.asarray(st.mob_dep[lid]) & ~iface[:, None]
        mob_field = tuple(xp.where(dep[:, c], p[g][lid] / xp.sqrt(total * h), 0.0)
                          for c, g in enumerate(("gamma_n", "gamma_p")))  # fmt: skip
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
        R_series=p.get("R_series", 0.0), R_shunt=p.get("R_shunt", -1.0), left_elec=st.left_elec,
        trap_charge=trap_charge, trap_cwe=trap_cwe, surface=surface,
        mob_field=mob_field,
    )  # fmt: skip


def build_device(setup: Setup, grid_split: str = "exact", surface: str = "extraction",
                 langevin: str = "layer") -> Device:  # fmt: skip
    """The device of a SIMsalabim input (Stage A, `check_supported` first), in NumPy. `grid_split`
    "simss" shares the grid points among the layers as SimSS's 10-byte reals do
    (`layer_points`; docs/differences.md §5), for a comparison point by point; `surface` how
    finite surface recombination at the contacts acts (`SURFACE_MODELS`); `langevin` which
    mobilities the Langevin rate takes (`LANGEVIN_MOBILITIES`)."""
    return device_arrays(*device_parameters(setup, grid_split, surface, langevin))
