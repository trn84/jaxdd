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
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from jaxdd.constants import K_B, Q

__all__ = [
    "TVG",
    "Layer",
    "Setup",
    "check_supported",
    "ion_region_problems",
    "ion_regions",
    "parse_parameters",
    "read_parameter_file",
    "read_setup",
    "read_tvg",
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


def read_parameter_file(path: str | os.PathLike) -> dict[str, str]:
    """`{name: value}` of a SIMsalabim parameter file."""
    return parse_parameters(Path(path).read_text())


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

    def f(self, name: str, default: float | None = None) -> float:
        """The parameter `name` as a number; `default` where the input has none (a parameter
        only SimSS or only ZimT reads)."""
        if default is not None and name not in self.raw:
            return float(default)
        return float(self.raw[name])

    def i(self, name: str, default: int | None = None) -> int:
        return int(self.f(name, default))

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


#: ln of the largest extended float (FPC's on x86-64), SIMsalabim's `myReal`: its bound on V/Vt.
_LN_MAX_REAL = 11356.523406294143


@dataclass(frozen=True)
class TVG:
    """ZimT's time steps (`tVGFile`): at each time `t` (s) the voltage `Vext` (V; NaN where
    `oc`, the open circuit), `G_frac` and `track` (0–3, ZimT's automatic stop). `dti` is
    1/Δt from the step before, 0 at `t = 0` (a steady state: the first step, or a return
    to it). ZimT takes Δt from the last step it accepted, so where it skips a failed step
    (`failureMode = 2`) the next one is longer (`transient.solve_transient` does as ZimT).
    With them the setup's `failureMode` (0 stop at a failed step, 1 keep it, 2 skip it),
    `convVar` (what two states compare by, `Track` 2 and 3) and the tolerances that
    comparison takes (`tolPois` V, `tolDens`, `tolCurr` A/m²)."""

    t: np.ndarray
    Vext: np.ndarray
    oc: np.ndarray
    G_frac: np.ndarray
    track: np.ndarray
    dti: np.ndarray
    failure_mode: int = 1
    conv_var: int = 1
    tol_pois: float = 1e-6
    tol_dens: float = 1e-6
    tol_curr: float = 1e-3


def read_tvg(setup: Setup) -> TVG:
    """ZimT 5.36's `tVGFile` (`Open_and_Read_tVG_file`, `Read_tVG`): lines before the header
    `t Vext G_frac Track` are skipped, then one step per line, up to the end or the first
    empty line; columns beyond the fourth are ignored. ZimT's checks raise `ValueError`: the
    header, a first step at `t = 0`, times that go forward or back to 0, a voltage within
    SIMsalabim's range, `Track` one of 0–3 and a 1 before any 2 or 3."""
    name = setup.raw["tVGFile"]
    lines = iter(setup.text("tVGFile").splitlines())
    for line in lines:
        if "".join(line.lower().split())[:16] == "tvextg_fractrack":
            break
    else:
        raise ValueError(f"Could not find correct header 't Vext G_frac Track' in {name}.")
    vti = Q / (K_B * setup.f("T"))
    rows, previous, reference = [], 0.0, False
    for line in lines:
        fields = line.split()
        if not fields:
            break
        try:
            t, v, g, track = (fields + [""] * 4)[:4]
            t, g = float(t), float(g)
            oc = v.lower() == "oc"
            v = np.nan if oc else float(v)
            if track not in ("0", "1", "2", "3"):
                raise ValueError
        except ValueError:
            raise ValueError(
                f"Error while reading from file {name}; offending line: {line!r}"
            ) from None
        if t > previous:
            dti = 1 / (t - previous)
        elif t == 0:
            dti = 0.0
        else:
            raise ValueError("Time steps must either be positive, or you should go back to t=0 "
                             f"(steady-state); at t = {t}")  # fmt: skip
        if not oc and abs(v * vti) > 1.95 * _LN_MAX_REAL:
            raise ValueError(f"V is too {'small' if v < 0 else 'large'}; at t = {t}")
        reference |= track == "1"
        if not reference and track in ("2", "3"):
            raise ValueError("Error in tVGFile: before setting Track=2 or 3, you need to have at "
                             "least one Track=1")  # fmt: skip
        rows.append((t, v, oc, g, int(track), dti))
        previous = t
        if len(rows) == 1:  # ZimT forgets the first line's Track=1 (its main loop resets it)
            reference = False
    if not rows or rows[0][0] != 0:
        raise ValueError("tVGFile did not specify steady-state (t=0)")
    t, v, oc, g, track, dti = (np.array(c) for c in zip(*rows, strict=True))
    raw = setup.raw
    failure_mode, conv_var = int(raw.get("failureMode", 1)), int(raw.get("convVar", 1))
    if failure_mode not in (0, 1, 2):
        raise ValueError("Invalid failureMode selected.")
    if conv_var not in (1, 2, 3, 4):
        raise ValueError("Invalid convVar selected.")
    return TVG(t=t, Vext=v, oc=oc.astype(bool), G_frac=g, track=track, dti=dti,
               failure_mode=failure_mode, conv_var=conv_var,
               tol_pois=float(raw.get("tolPois", 1e-6)), tol_dens=float(raw.get("tolDens", 1e-6)),
               tol_curr=float(raw.get("tolCurr", 1e-3)))  # fmt: skip


def ion_regions(setup: Setup, species: str) -> list[tuple[int, int, bool]]:
    """SIMsalabim's ionic regions of `species` ("anion" or "cation", `Init_Ionic_Region`):
    `(first, last, mobile)`, layer indices from 0. A region is a run of adjacent layers each with
    `ionsMayEnter` (or one layer of immobile ions), holding some ions; it ends at a layer whose
    ions do not move or before a layer they may not enter, and is mobile if the mobility of its
    last layer is not zero. As in SimSS 5.36, a region found does not reset the start: one that
    follows directly begins where the one before began. SimSS's check: a layer with mobile ions
    must let them enter (`ValueError`)."""
    regions, start, total = [], None, 0.0
    layers = setup.layers
    for j, ly in enumerate(layers):
        mob, conc = ly.f(f"mu_{species}"), ly.f(f"N_{species}")
        if mob * conc > 0 and not ly.i("ionsMayEnter"):
            raise ValueError(f"layer {j + 1}: If a layer contains mobile ions, then ionsMayEnter "
                             "(of that layer) must be 1.")  # fmt: skip
        if ly.i("ionsMayEnter") or (mob == 0 and conc > 0):
            total += ly.f("L") * conc
            start = j if start is None else start
        else:
            start, total = None, 0.0
            continue
        if j == len(layers) - 1 or not layers[j + 1].i("ionsMayEnter") or mob == 0:
            if total != 0:
                regions.append((start, j, mob > 0))
            total = 0.0
    return regions


def _layers(raw: dict[str, str], read) -> tuple[Layer, ...]:
    layers = []
    k = 1
    while f"l{k}" in raw:
        layers.append(Layer(parse_parameters(read(raw[f"l{k}"]))))
        k += 1
    return tuple(layers)


def read_setup(path: str | os.PathLike) -> Setup:
    """The setup file and every layer file `l1`, `l2`, … it names, found beside it."""
    path = Path(path)
    raw = read_parameter_file(path)
    layers = _layers(raw, lambda file: (path.parent / file).read_text())
    return Setup(raw=raw, layers=layers, directory=path.parent)


def setup_from_texts(files: Mapping[str, str], setup: str = "simulation_setup.txt") -> Setup:
    """`read_setup` without files: `files` maps each file name (as the setup names it) to its
    text, the setup file under `setup`."""
    raw = parse_parameters(files[setup])
    return Setup(raw=raw, layers=_layers(raw, lambda file: files[file]), files=dict(files))


def ion_region_problems(setup: Setup) -> list[str]:
    """What jaxdd does not take of the ionic regions (`ion_regions`): SimSS 5.36's two quirks
    with regions of several layers, which it would solve differently from what the input says
    (docs/differences.md §10)."""
    problems = []
    for species in ("anion", "cation"):
        for first, last, mobile in ion_regions(setup, species):
            if first == last:
                continue
            span = [setup.layers[k] for k in range(first, last + 1)]
            where = f"{species}s: SimSS's region of layers {first + 1}–{last + 1}"
            closed = [first + 1 + k for k, ly in enumerate(span) if not ly.i("ionsMayEnter")]
            if closed:
                problems.append(f"{where} takes in layer {closed[0]}, which they may not enter")
            if not mobile and any(ly.f(f"mu_{species}") * ly.f(f"N_{species}") > 0 for ly in span):
                problems.append(f"{where} has mobility 0 in its last layer, so it holds them all")
    return problems


def check_supported(setup: Setup) -> None:
    """Raise `NotImplementedError` naming every parameter outside what jaxdd solves.

    jaxdd solves SimSS 5.36's steady state with all of its modules (docs/overview.md, "Scope"):
    traps in the bulk and at interfaces, neutral or charged, one level or several from a trap
    file; series and shunt resistance; the generation profile from a file, uniform or by the
    transfer-matrix method; either electrode the cathode; finite surface recombination and
    'sfb' work functions; Langevin recombination; field-dependent mobility and generation;
    either form of the current (`currDiffInt` 1 or 2); mobile ions in regions of one layer or
    several (`ion_regions`), free at every voltage, frozen after a pre-bias (`preCond`,
    `fixIons`, as SimSS with `simss/fixed_ions.patch`) or held from the start (`fixIons`
    alone); and ZimT 5.36's transients of the same devices. What it refuses are the inputs SimSS
    refuses itself and two of SimSS's ionic regions it would solve against what the input says
    (`ion_region_problems`, docs/differences.md §10); anything else is refused by name, never
    approximated."""
    problems = []
    if float(setup.raw.get("currDiffInt", 2)) not in (1.0, 2.0):
        problems.append(f"currDiffInt = {setup.raw['currDiffInt']} (SimSS has 1 and 2)")
    if float(setup.raw.get("R_series", 0)) < 0 or float(setup.raw.get("R_shunt", -1)) == 0:
        problems.append("R_series < 0 or R_shunt = 0 (SimSS refuses both)")
    problems += ion_region_problems(setup)
    if problems:
        raise NotImplementedError("outside what jaxdd solves:\n  " + "\n  ".join(problems))
