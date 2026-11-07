"""Read SIMsalabim's input files: `simulation_setup.txt` and the layer files it names.

The format is SIMsalabim's: one `name = value * comment` per line, `*` comments, order free
for reading. Values are kept as strings here and converted where they are used, so that a
parameter `jaxdd` does not implement can be named in an error instead of silently dropped
(`check_supported`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["Layer", "Setup", "check_supported", "read_parameter_file", "read_setup"]


def read_parameter_file(path: Path) -> dict[str, str]:
    """`{name: value}` of a SIMsalabim parameter file."""
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        body = line.split("*", 1)[0].strip()
        if "=" not in body:
            continue
        name, value = (part.strip() for part in body.split("=", 1))
        if name:
            values[name] = value
    return values


@dataclass(frozen=True)
class Layer:
    """One layer's parameters, as SIMsalabim reads them (SI units, energies in eV)."""

    raw: dict[str, str]

    def f(self, name: str) -> float:
        return float(self.raw[name])

    def i(self, name: str) -> int:
        return int(float(self.raw[name]))


@dataclass(frozen=True)
class Setup:
    """`simulation_setup.txt` with its layers, in order from the left electrode."""

    raw: dict[str, str]
    layers: tuple[Layer, ...]
    directory: Path

    def f(self, name: str) -> float:
        return float(self.raw[name])

    def i(self, name: str) -> int:
        return int(float(self.raw[name]))

    def path(self, name: str) -> Path:
        return self.directory / self.raw[name]


def read_setup(path: Path) -> Setup:
    """The setup file and every layer file `l1`, `l2`, … it names."""
    raw = read_parameter_file(path)
    layers = []
    k = 1
    while f"l{k}" in raw:
        layers.append(Layer(read_parameter_file(path.parent / raw[f"l{k}"])))
        k += 1
    return Setup(raw=raw, layers=tuple(layers), directory=path.parent)


#: Stage A (PLAN.md §2.1): the values a supported input must have. Ions are supported in one
#: layer per region, steady (free at every voltage) or held after a pre-bias (`preCond`,
#: `fixIons`, with FauxPV's patch that keeps Poisson from moving them). Anything else is refused by
#: name, never approximated.
_SETUP_REQUIRED = {
    "S_n_L": -1.0,
    "S_p_L": -1.0,
    "S_n_R": -1.0,
    "S_p_R": -1.0,
    "R_shunt": -1.0,
    "R_series": 0.0,
    "Vdist": 1.0,
    "currDiffInt": 2.0,
}
_LAYER_REQUIRED = {
    "mobnDep": 0.0,
    "mobpDep": 0.0,
    "N_t_int": 0.0,
    "fieldDepG": 0.0,
    "useLangevin": 0.0,
}


def check_supported(setup: Setup) -> None:
    """Raise `NotImplementedError` naming every parameter outside Stage A without ions."""
    problems = [
        f"{name} = {setup.raw.get(name)} (supported: {value:g})"
        for name, value in _SETUP_REQUIRED.items()
        if float(setup.raw.get(name, value)) != value
    ]
    if setup.raw.get("genProfile", "none") in ("none", "calc"):
        problems.append(f"genProfile = {setup.raw.get('genProfile')} (supported: a file)")
    for k, layer in enumerate(setup.layers, start=1):
        problems += [
            f"layer {k}: {name} = {layer.raw.get(name)} (supported: {value:g})"
            for name, value in _LAYER_REQUIRED.items()
            if float(layer.raw.get(name, value)) != value
        ]
        if layer.f("N_t_bulk") != 0 and layer.i("bulkTrapType") != 0:
            problems.append(
                f"layer {k}: charged bulk traps (bulkTrapType {layer.i('bulkTrapType')})"
            )
        if layer.i("ionsMayEnter") and k > 1 and setup.layers[k - 2].i("ionsMayEnter"):
            problems.append(f"layers {k - 1}–{k}: an ionic region across layers")
        if layer.raw.get("bulkTrapFile", "none") != "none":
            problems.append(f"layer {k}: bulkTrapFile = {layer.raw['bulkTrapFile']}")
    mobile = any(ly.f("N_anion") * ly.f("mu_anion") > 0 or ly.f("N_cation") * ly.f("mu_cation") > 0
                 for ly in setup.layers)  # fmt: skip
    if mobile and setup.i("fixIons") and not setup.i("preCond"):
        problems.append("fixIons = 1 without preCond (ions held at their initial uniform density)")
    if problems:
        raise NotImplementedError("not in jaxdd's Stage A:\n  " + "\n  ".join(problems))
