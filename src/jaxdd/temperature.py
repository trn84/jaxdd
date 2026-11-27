"""Temperature laws: a device's parameters (`device.device_parameters`' dict) at another
temperature.

jaxdd's `T`, as SIMsalabim's, enters a device only through `V_T = kT/q`; every other parameter
is a constant. `at_temperature(params, T, laws, gap)` maps the dict from its own `T`, the
reference `T0`, to `T`: each parameter a law names scaled by it (`Law`, or the caller's own
function), the bands moved by the gap's change (`Gap`) and the absolute energies that follow
them (`FOLLOW`), and `T` set. A pure function in NumPy or `jax.numpy` (JAX's where the dict, `T`
or a law's coefficients are JAX arrays), in front of `batch.device_tree`: differentiable in `T`
and in the laws' coefficients, batched by `vmap` over `T`. At `T = T0` every value comes back
unchanged, to the bit. The forms are jaxdd's, the coefficients the caller's; nothing changes
with `T` unless a law says so. `check_gaps` checks a mapped dict's energies against its gaps,
as `device_parameters` checks the input's (docs/model.md, "Temperature laws").
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.constants import K_B, Q
from jaxdd.device import SURFACE_PARAMETERS, Structure

__all__ = ["ENERGIES", "FOLLOW", "Gap", "Law", "at_temperature", "check_gaps", "textbook_laws"]

#: The absolute energies of the dict (eV below the vacuum level), which the bands' move carries
#: or leaves (`at_temperature`'s `traps`, `contacts`, `edge`) and no law scales.
ENERGIES = ("E_c", "E_v", "E_t_bulk", "E_t_int", "bulk_trap_E", "int_trap_E", "W_L", "W_R",
            "dE_nk")  # fmt: skip

#: What a trap level or a work function follows when its layer's bands move: nothing (its
#: absolute energy stays), the conduction band (its distance to `E_c` stays), the valence band,
#: or the middle of the gap.
FOLLOW = ("fixed", "E_c", "E_v", "midgap")

#: A law on a layer's capture coefficient acts on its trap file's levels' own as well
#: (`device.TRAP_FILE_CAPTURE`), unless one is given for those.
_LEVELS = {"C_n_bulk": "bulk_trap_Cn", "C_p_bulk": "bulk_trap_Cp",
           "C_n_int": "int_trap_Cn", "C_p_int": "int_trap_Cp"}  # fmt: skip


def _xp(*values):
    """`jax.numpy` where any value is a JAX array (a tracer too), else NumPy."""
    return jnp if any(isinstance(v, jax.Array) for v in values) else np


def _column(v, ndim: int):
    """A coefficient (a number, or one per layer) against a parameter of `ndim` axes, its layers
    first: a layer's coefficient for each of its levels."""
    if np.ndim(v) == 0:
        return v
    v = v if isinstance(v, jax.Array | np.ndarray) else np.asarray(v, dtype=float)
    return v.reshape(v.shape + (1,) * (ndim - v.ndim))


@dataclass(frozen=True, eq=False)
class Law:
    """A parameter's law, `value·(T/T0)^m·exp(−(E/k)·(1/T − 1/T0))`: a power `m` of the
    temperature and an activation energy `E` (eV), each a number or one per layer (an array over
    the layers; against a trap file's levels, each layer's for all of its levels). The defaults
    change nothing. At `T = T0` the factor is 1 exactly, so the value comes back to the bit. As
    any law, a callable `law(value, T, T0) -> value`."""

    m: float | np.ndarray = 0.0
    E: float | np.ndarray = 0.0

    def factor(self, T, T0, ndim: int = 0):
        """The factor at `T` against `T0`, shaped for a parameter of `ndim` axes."""
        xp = _xp(T, T0, self.m, self.E)
        m, E = _column(self.m, ndim), _column(self.E, ndim)
        return (T / T0) ** m * xp.exp(-(E * (Q / K_B)) * (1.0 / T - 1.0 / T0))

    def __call__(self, value, T, T0):
        return value * self.factor(T, T0, np.ndim(value))


@dataclass(frozen=True, eq=False)
class Gap:
    """The band gap's change with the temperature, per layer: linear, `slope·(T − T0)` (eV/K),
    or the caller's `Eg(T)` (eV), whose change `Eg(T) − Eg(T0)` counts (Varshni's, a phase
    transition's step). `share` (0 to 1) of the change is taken by `E_v`, the rest by `E_c`: a
    gap that grows moves `E_v` down from the vacuum level (its number up) and `E_c` towards it.
    `slope`, `share` and `Eg`'s value each a number or one per layer."""

    slope: float | np.ndarray = 0.0
    share: float | np.ndarray = 0.5
    Eg: Callable | None = None

    def change(self, T, T0):
        """The gap's change from `T0` to `T` (eV); 0 exactly at `T = T0`."""
        if self.Eg is not None:
            return self.Eg(T) - self.Eg(T0)
        return self.slope * (T - T0)


def textbook_laws(E_anion=0.0, E_cation=0.0) -> dict[str, Law]:
    """The textbook laws, to pass to `at_temperature` (alone, or changed and added to): `N_c`
    as a parabolic band's density of states, `T^{3/2}`; the interface velocities `nu_int_n`,
    `nu_int_p` as a thermal velocity, `T^{1/2}`; and each ion species' mobility as an Arrhenius
    diffusion `D ∝ e^{−E/kT}` with its activation energy (eV, a number or one per layer; 0, a
    constant `D`, by default), `μ = D/V_T ∝ T^{−1}·e^{−E/kT}`."""
    return {"N_c": Law(m=1.5), "nu_int_n": Law(m=0.5), "nu_int_p": Law(m=0.5),
            "mu_anion": Law(m=-1.0, E=E_anion), "mu_cation": Law(m=-1.0, E=E_cation)}


def _per_layer(option: str | Sequence[str], n: int, what: str) -> np.ndarray:
    """`traps`' or a contact's option, one per layer (or per contact)."""
    options = [option] * n if isinstance(option, str) else list(option)
    if len(options) != n or any(o not in FOLLOW for o in options):
        raise ValueError(f"{what} {option!r}: one of {FOLLOW}, or one per layer")
    return np.array(options)


def _shaped(new, value, name: str):
    """A law's result, which has its value's shape (static, under a trace too)."""
    if np.shape(new) != np.shape(value):
        raise ValueError(f"a law for {name}: a value of shape {np.shape(new)}, not "
                         f"{np.shape(value)} as the parameter's (one per layer on a single "
                         "value?)")  # fmt: skip
    return new


def at_temperature(params: dict, T, laws: Mapping[str, Callable] | None = None,
                   gap: Gap | None = None, traps: str | Sequence[str] = "fixed",
                   contacts: str | tuple[str, str] = "fixed", edge: bool = False,
                   structure: Structure | None = None) -> dict:  # fmt: skip
    """The parameters `params` (`device.device_parameters`' dict) at the temperature `T`, from
    the dict's own `T`, the reference `T0`. A new dict; the input is not changed.

    `laws`: by a parameter's name, its law, a `Law` or any `law(value, T, T0) -> value` (it
    should return the value at `T0` unchanged, as `Law` does to the bit; `v * (g(T) / g(T0))` is
    so for any `g`). Each sees the input's value. Any parameter of the dict but `T` and the
    energies (`ENERGIES`, which `gap` moves): a law on `C_n_bulk`, `C_p_bulk`, `C_n_int` or
    `C_p_int` acts on the trap files' levels' own (`bulk_trap_Cn`, …, layers × levels) as well,
    unless one is given for those: a `Law` each layer's coefficients for all of its levels, a
    caller's function on them transposed (levels × layers, so that a factor per layer
    broadcasts over the layers). A law keeps a value's shape: one per layer on a single value
    is refused. A negative (infinite) surface recombination velocity stays as it is.

    `gap` (`Gap`): the bands' move with `T`. Then `traps` says what every trap level of a layer
    follows (`FOLLOW`, one for all or one per layer): `E_t_bulk`, `bulk_trap_E`, and a layer's
    interface level to the next (`E_t_int`, `int_trap_E`), which is the layer's on its left;
    `"fixed"` by default. `contacts` (one for both, or (left, right)) what `W_L` and `W_R`
    follow, their adjacent layer's bands; `"fixed"` by default, the electrode's work function.
    With `structure` (the dict's `Structure`) an 'sfb' or 'free' end's `W` (unused) stays, and
    `k_direct` in a layer with Langevin's rate (which the device does not read). `edge`: where the
    dict has `dE_nk` (the shift of a layer's n,k in photon energy, eV), the gap's change is added
    to it, so the computed absorption follows the gap.

    The `Structure` stays the input's: a law cannot switch on what the input lacks (a trap level
    on a layer without traps, Auger in a device without any, a finite `S` where it is
    infinite). `check_gaps` checks the result's energies against its gaps."""
    T0 = params["T"]
    laws = dict(laws or {})
    for name in laws:
        if name == "T" or name in ENERGIES:
            raise ValueError(f"a law for {name!r}: the temperature and the energies "
                             f"{ENERGIES} have none (`gap` moves the bands)")  # fmt: skip
        if name not in params and name not in _LEVELS.values():
            raise ValueError(f"a law for {name!r}: not a parameter of the dict")
    langevin = None
    if structure is not None and structure.langevin is not None and structure.langevin.any():
        langevin = structure.langevin
    p = dict(params)
    for name, law in laws.items():
        if name not in params:  # the levels' own capture coefficients, which this input lacks
            continue
        value = params[name]
        new = _shaped(law(value, T, T0), value, name)
        if name in SURFACE_PARAMETERS:
            new = _xp(value, new).where(value < 0, value, new)
        if name == "k_direct" and langevin is not None:
            new = _xp(value, new).where(langevin, value, new)
        p[name] = new
        levels = _LEVELS.get(name)
        if levels in params and levels not in laws:
            v = params[levels]  # layers × levels
            new = law(v, T, T0) if isinstance(law, Law) else law(v.T, T, T0).T
            p[levels] = _shaped(new, v, f"{name} (on {levels})")
    n = len(params["E_c"])
    on_traps = _per_layer(traps, n, "traps")
    on_contacts = _per_layer(contacts, 2, "contacts")
    if gap is not None:
        change = gap.change(T, T0)
        xp = _xp(change, gap.share, params["E_c"])
        zero = xp.zeros(n)
        dE_v = gap.share * change + zero
        dE_c = -((1.0 - gap.share) * change) + zero
        moves = {"E_c": dE_c, "E_v": dE_v, "midgap": 0.5 * (dE_c + dE_v)}

        def shift(options):  # per layer: the move of what follows each option
            s = zero
            for option, dE in moves.items():
                s = xp.where(options == option, dE, s)
            return s

        p["E_c"] = params["E_c"] + dE_c
        p["E_v"] = params["E_v"] + dE_v
        if (on_traps != "fixed").any():
            s = shift(on_traps)
            for name in ("E_t_bulk", "E_t_int"):
                p[name] = params[name] + s
            for name in ("bulk_trap_E", "int_trap_E"):
                if name in params:
                    p[name] = params[name] + s[:, None]
        sfb = structure.sfb if structure is not None else (False, False)
        for side, (name, k) in enumerate((("W_L", 0), ("W_R", n - 1))):
            if on_contacts[side] != "fixed" and not sfb[side]:
                p[name] = params[name] + moves[on_contacts[side]][k]
        if edge and "dE_nk" in params:
            p["dE_nk"] = params["dE_nk"] + (change + zero)
    p["T"] = T
    return p


def check_gaps(structure: Structure, params: dict) -> None:
    """The energies of a dict of `structure` (`at_temperature`'s, say): each layer's gap open
    (`E_v > E_c`), and `device_parameters`' checks of the input, each work function not 'sfb' or
    'free' within its layer's gap, each trap level of a layer with traps strictly inside its gap
    and an interface's inside both of its layers' (a trap file's levels of density 0 aside). On
    concrete values, outside any trace: NumPy's or JAX's, of one device or stacked (several
    temperatures by `vmap`, say), every one checked. `ValueError` naming what is outside."""

    def a(name):
        return np.asarray(params[name], dtype=float)

    E_c, E_v = a("E_c"), a("E_v")
    problems = []
    for k in range(E_c.shape[-1]):
        if not (E_v[..., k] > E_c[..., k]).all():
            problems.append(f"layer {k + 1}: no gap (E_c {E_c[..., k]} ≥ E_v {E_v[..., k]} eV)")
    for side, (name, k) in enumerate((("W_L", 0), ("W_R", E_c.shape[-1] - 1))):
        if structure.sfb[side]:
            continue
        W = a(name)
        if not ((E_c[..., k] <= W) & (E_v[..., k] >= W)).all():
            problems.append(f"{name} = {W} eV: outside the gap of its layer "
                            f"({E_c[..., k]} to {E_v[..., k]} eV)")  # fmt: skip

    def levels(kind, j, files):  # a layer's levels (..., levels) and where they count
        if files is not None and files[j]:
            return a(f"{kind}_trap_E")[..., j, :], a(f"{kind}_trap_N")[..., j, :] != 0
        E = a(f"E_t_{kind}")[..., j, None]
        return E, np.ones(E.shape, dtype=bool)

    def outside(E, on, lo, hi):  # the levels that count and are not strictly inside (lo, hi)
        return np.unique(E[on & ~((lo[..., None] < E) & (hi[..., None] > E))])

    for j in np.flatnonzero(structure.traps):
        bad = outside(*levels("bulk", j, structure.bulk_trap_file), E_c[..., j], E_v[..., j])
        if bad.size:
            problems.append(f"layer {j + 1}: a bulk trap level outside the gap ({bad} eV)")
    if structure.int_traps is not None:
        for j in np.flatnonzero(structure.int_traps):
            lo = np.maximum(E_c[..., j], E_c[..., j + 1])
            hi = np.minimum(E_v[..., j], E_v[..., j + 1])
            bad = outside(*levels("int", j, structure.int_trap_file), lo, hi)
            if bad.size:
                problems.append(f"interface trap between layers {j + 1} and {j + 2} at {bad} "
                                "eV: not within the gaps of both")  # fmt: skip
    if problems:
        raise ValueError("; ".join(problems))
