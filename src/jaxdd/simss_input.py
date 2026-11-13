"""Read SIMsalabim's input files: `simulation_setup.txt` and the layer files it names.

The format is SIMsalabim's: one `name = value * comment` per line, `*` comments, order free
for reading. Values are kept as strings here and converted where they are used, so that a
parameter `jaxdd` does not implement can be named in an error instead of silently dropped
(`check_supported`).

The files come from a directory (`read_setup`) or from a mapping of file names to texts
(`setup_from_texts`), for sweeps that build devices without writing files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "Layer",
    "Setup",
    "check_supported",
    "parse_parameters",
    "read_parameter_file",
    "read_setup",
    "setup_from_texts",
]


def parse_parameters(text: str) -> dict[str, str]:
    """`{name: value}` of a SIMsalabim parameter file's text."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        body = line.split("*", 1)[0].strip()
        if "=" not in body:
            continue
        name, value = (part.strip() for part in body.split("=", 1))
        if name:
            values[name] = value
    return values


def read_parameter_file(path: Path) -> dict[str, str]:
    """`{name: value}` of a SIMsalabim parameter file."""
    return parse_parameters(path.read_text())


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
    """`simulation_setup.txt` with its layers, in order from the left electrode. The files it
    names (the generation profile) are read from `files`, file name to text, where given there,
    else from `directory`."""

    raw: dict[str, str]
    layers: tuple[Layer, ...]
    directory: Path | None = None
    files: Mapping[str, str] | None = None

    def f(self, name: str) -> float:
        return float(self.raw[name])

    def i(self, name: str) -> int:
        return int(float(self.raw[name]))

    def text(self, name: str) -> str:
        """The text of the file the parameter `name` names."""
        file = self.raw[name]
        if self.files is not None and file in self.files:
            return self.files[file]
        if self.directory is None:
            raise FileNotFoundError(f"{name} = {file}: not among the setup's files")
        return (self.directory / file).read_text()


def _layers(raw: dict[str, str], read) -> tuple[Layer, ...]:
    layers = []
    k = 1
    while f"l{k}" in raw:
        layers.append(Layer(parse_parameters(read(raw[f"l{k}"]))))
        k += 1
    return tuple(layers)


def read_setup(path: Path) -> Setup:
    """The setup file and every layer file `l1`, `l2`, … it names."""
    raw = read_parameter_file(path)
    layers = _layers(raw, lambda file: (path.parent / file).read_text())
    return Setup(raw=raw, layers=layers, directory=path.parent)


def setup_from_texts(files: Mapping[str, str], setup: str = "simulation_setup.txt") -> Setup:
    """`read_setup` without files: `files` maps each file name (as the setup names it) to its
    text, the setup file under `setup`."""
    raw = parse_parameters(files[setup])
    return Setup(raw=raw, layers=_layers(raw, lambda file: files[file]), files=dict(files))


#: Stage A (docs/overview.md, "Scope") and the modules of Stage B so far (interface traps, one
#: level each): the values a supported input must have. Ions are
#: supported in one layer per region, steady (free at every voltage) or held after a pre-bias
#: (`preCond`, `fixIons`, as SimSS with `simss/fixed_ions.patch`: Poisson leaves them). Anything
#: else is refused by name, never approximated.
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
        if layer.raw.get("intTrapFile", "none") != "none" and k < len(setup.layers):
            problems.append(f"layer {k}: intTrapFile = {layer.raw['intTrapFile']}")
    mobile = any(ly.f("N_anion") * ly.f("mu_anion") > 0 or ly.f("N_cation") * ly.f("mu_cation") > 0
                 for ly in setup.layers)  # fmt: skip
    if mobile and setup.i("fixIons") and not setup.i("preCond"):
        problems.append("fixIons = 1 without preCond (ions held at their initial uniform density)")
    if problems:
        raise NotImplementedError("not in jaxdd's Stage A:\n  " + "\n  ".join(problems))
