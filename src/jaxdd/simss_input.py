"""Read SIMsalabim's input files: `simulation_setup.txt` and the layer files it names.

The format is SIMsalabim's: one `name = value * comment` per line, `*` comments, order free
for reading. Values are kept as strings here and converted where they are used, so that a
parameter `jaxdd` does not implement can be named in an error instead of silently dropped
(`check_supported`).

The files come from a directory (`read_setup`) or from a mapping of file names to texts
(`setup_from_texts`), for sweeps that build devices without writing files.
"""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "Layer",
    "Setup",
    "check_supported",
    "parse_parameters",
    "read_parameter_file",
    "read_setup",
    "setup_from_texts",
    "simss_voltages",
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
        return self.read(self.raw[name], name)

    def read(self, file: str, name: str | None = None) -> str:
        """The text of a file the input names (a generation profile, n,k data, a spectrum): from
        `files` if given there, else relative to `directory`. `name`, the parameter that names it,
        for the error."""
        if self.files is not None and file in self.files:
            return self.files[file]
        if self.directory is None:
            what = f"{name} = {file}" if name else file
            raise FileNotFoundError(f"{what}: not among the setup's files")
        return (self.directory / file).read_text()


def simss_voltages(setup: Setup) -> np.ndarray:
    """The internal voltages SimSS simulates, in its order (`Init_Voltages_and_Tasks`): from
    `Vmin` to `Vmax` in steps of `Vstep` (`Vdist = 1`) or `NJV` of them accumulating towards
    `Vacc` (`Vdist = 2`), reversed for `Vscan = -1`; with `useExpData` the `Vext` column of
    `expJV` instead, in the file's order. The pre-bias `Vpre` is not among them (`solve_jv`
    takes it from the device). With `untilVoc` SimSS stops past the open circuit; these are all
    the voltages it would reach without stopping. SimSS's checks on them raise `ValueError`."""
    if "useExpData" in setup.raw and setup.i("useExpData"):
        table = np.genfromtxt(io.StringIO(setup.text("expJV")), names=True)
        if "Vext" not in (table.dtype.names or ()):
            raise ValueError(f"expJV = {setup.raw['expJV']}: no column Vext")
        V = np.atleast_1d(table["Vext"]).astype(float)
        if V.size < 4:  # SimSS's minExpData
            raise ValueError("Not enough experimental data points (SimSS needs 4).")
        return V
    vdist, vscan = setup.i("Vdist"), setup.i("Vscan")
    vmin, vmax = setup.f("Vmin"), setup.f("Vmax")
    if vdist not in (1, 2):
        raise ValueError(f"Vdist = {vdist}: SimSS has 1 (uniform) and 2 (logarithmic)")
    if abs(vscan) != 1:
        raise ValueError("Vscan must be either -1 or 1.")
    if vmin > vmax:
        raise ValueError("Vmin should not be greater than Vmax.")
    if vdist == 1:
        step = setup.f("Vstep")
        if step <= 0:
            raise ValueError("Vstep should be positive.")
        njv = int(np.trunc((vmax - vmin) / step + 1e-10)) + 1
        V = vmin + step * np.arange(njv)
    else:
        acc, njv = setup.f("Vacc"), setup.i("NJV")
        if vmin <= acc <= vmax:
            raise ValueError("Invalid Vacc selected, must be outside [Vmin, Vmax].")
        if abs(vmin - vmax) < 1e-10:
            raise ValueError("Do not use Vdist=2 when Vmin = Vmax.")
        if njv < 2:
            raise ValueError(f"NJV = {njv}: Vdist = 2 needs at least 2 voltages")
        d = acc - vmax
        V = acc - d * np.exp((1 - np.arange(njv) / (njv - 1)) * np.log((acc - vmin) / d))
    return V[::-1].copy() if vscan == -1 else V


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


#: Stage A (docs/overview.md, "Scope") and the modules of Stage B so far (bulk and interface
#: traps, neutral or charged, one level or several from a trap file; series and shunt
#: resistance; the generation profile; the left electrode as the anode; finite surface
#: recombination at the contacts and 'sfb' work functions; Langevin recombination; the
#: field-dependent mobility and generation; either form of the current, `currDiffInt` 1 or
#: 2). Ions are supported in one layer per region, steady (free at every voltage) or held
#: after a pre-bias (`preCond`, `fixIons`, as SimSS with `simss/fixed_ions.patch`: Poisson
#: leaves them). The voltages are the caller's (`simss_voltages` gives SimSS's, for either
#: `Vdist`). Anything else is refused by name, never approximated.
def check_supported(setup: Setup) -> None:
    """Raise `NotImplementedError` naming every parameter outside what jaxdd supports."""
    problems = []
    if float(setup.raw.get("currDiffInt", 2)) not in (1.0, 2.0):
        problems.append(f"currDiffInt = {setup.raw['currDiffInt']} (SimSS has 1 and 2)")
    if float(setup.raw.get("R_series", 0)) < 0 or float(setup.raw.get("R_shunt", -1)) == 0:
        problems.append("R_series < 0 or R_shunt = 0 (SimSS refuses both)")
    for k, layer in enumerate(setup.layers, start=1):
        if layer.i("ionsMayEnter") and k > 1 and setup.layers[k - 2].i("ionsMayEnter"):
            problems.append(f"layers {k - 1}–{k}: an ionic region across layers")
    mobile = any(ly.f("N_anion") * ly.f("mu_anion") > 0 or ly.f("N_cation") * ly.f("mu_cation") > 0
                 for ly in setup.layers)  # fmt: skip
    if mobile and setup.i("fixIons") and not setup.i("preCond"):
        problems.append("fixIons = 1 without preCond (ions held at their initial uniform density)")
    if problems:
        raise NotImplementedError("not in jaxdd's Stage A:\n  " + "\n  ".join(problems))
