"""TRPL of perovskite films and half-stacks without electrodes: the devices, the laser pulse,
the measurement's binning, the data of Kober-Czerny et al. (2025) and ZimT 5.36's cases for the
same devices (docs/validation.md, "TRPL without electrodes").

The perovskite (600 nm FAPbI3) is as Kober-Czerny et al., PRX Energy 4, 013001 (2025) found it
(0.41 cm²/Vs, k_rad 7.5e-11 cm³/s, a photodoping of 2e15 cm⁻³, a deep and a shallow trap level);
the transport layers (30 nm SnO2 below it, 200 nm Spiro-OMeTAD above it) take typical literature
values, assumed, not fitted. Their interfaces with the perovskite recombine at the authors' SRV
(9 and 100 cm/s) through interface traps, `S = N_t·C`.

The pulse is a generation step of `T_PULSE` with the profile `exp(−α·z)` from the lit side;
the PL is `∫k_rad·n·p dz` (the `Jdir` columns), averaged over the 2 ns bins of the measurement
and normalised to its first (`as_measured`).
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from jaxdd.constants import K_B, Q
from jaxdd.simss_input import TVG, setup_from_texts

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "validation" / "data" / "koberczerny2025"
_SETUP = (REPO / "tests/data/none_light/simulation_setup.txt").read_text()
_LAYER = (REPO / "tests/data/none_light/L2_parameters.txt").read_text()
_ZIMT = (REPO / "tests/data/zimt_photo_surface_light/simulation_setup.txt").read_text()

ALPHA = 3e7  # 1/m, at 398 nm
FLUENCE = 4.8e15  # photons/m² (4.8e11 cm⁻²), the high fluence
T_PULSE = 1e-11  # s
BIN = 2e-9  # s, the TCSPC bins
C = 1e-14  # m³/s, the traps' capture coefficient
K_B_T = K_B * 295 / Q  # V, the thermal voltage at the stacks' 295 K

PEROVSKITE = {
    "L": "600e-9", "E_c": 4.0, "E_v": 5.53, "N_c": "2.2e24", "eps_r": 30, "layerGen": 1,
    "mu_n": "4.1e-5", "mu_p": "4.1e-5", "k_direct": "7.5e-17", "N_A": "2.07e21", "N_D": 0,
    "N_t_bulk": "1e20", "C_n_bulk": C, "C_p_bulk": C, "bulkTrapFile": "traps.txt",
    "bulkTrapType": -1, "nu_int_n": "1e5", "nu_int_p": "1e5",
}  # fmt: skip
# The authors' k_deep (1.6e4 /s) and k_capture (1.3e6 /s) as N_t·C, k_emission (7.5e5 /s) as the
# shallow level's depth below E_c.
TRAPS = f"E Ntrap\n4.765 {1.6e4 / C:.4e}\n4.261 {1.3e6 / C:.4e}\n"
# The same with each level's own capture coefficients (`device.TRAP_FILE_CAPTURE`): the shallow
# level takes no holes, so it traps electrons and gives them back without recombining.
TRAPS_OWN = (f"E Ntrap C_n C_p\n4.765 {1.6e4 / C:.4e} {C} {C}\n"
             f"4.261 {1.3e6 / C:.4e} {C} 0\n")  # fmt: skip
SNO2 = {
    "L": "30e-9", "E_c": 4.1, "E_v": 7.6, "N_c": "2.2e24", "eps_r": 9, "N_D": "1e24", "N_A": 0,
    "mu_n": "1e-4", "mu_p": "1e-4", "k_direct": 0, "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0,
    "bulkTrapFile": "none", "nu_int_n": "1e5", "nu_int_p": "1e5",
}  # fmt: skip
SPIRO = {
    "L": "200e-9", "E_c": 2.0, "E_v": 5.2, "N_c": "2.2e24", "eps_r": 3, "N_A": "1e24", "N_D": 0,
    "mu_n": "1e-7", "mu_p": "1e-7", "k_direct": 0, "layerGen": 0, "G_ehp": 0, "N_t_bulk": 0,
    "bulkTrapFile": "none",
}  # fmt: skip


def interface(S_cm):
    """Interface traps on a layer's right interface giving it the SRV `S_cm` (cm/s)."""
    return {"N_t_int": f"{S_cm * 1e-2 / C:.4e}", "C_n_int": C, "C_p_int": C, "E_t_int": 4.765,
            "intTrapType": 0}  # fmt: skip


#: name: (layers from the glass up, index of the perovskite, or a tuple of the indices of the
#: layers it is made of)
STACKS = {
    "film": ([PEROVSKITE], 0),
    "SnO2": ([{**SNO2, **interface(9)}, PEROVSKITE], 1),
    "Spiro": ([{**PEROVSKITE, **interface(100)}, SPIRO], 0),
}


def with_bottom_zone(thickness):
    """`STACKS` with a zone of `thickness` (m) at each perovskite's bottom, a layer of its own
    (the same bands, its own traps): the stacks, and each stack's zone (its layer index)."""
    zone = {**PEROVSKITE, "L": f"{thickness:.6e}", "nu_int_n": "1e7", "nu_int_p": "1e7"}
    rest = {**PEROVSKITE, "L": f"{600e-9 - thickness:.6e}"}
    stacks = {
        "film": ([zone, rest], (0, 1)),
        "SnO2": ([{**SNO2, **interface(9)}, zone, rest], (1, 2)),
        "Spiro": ([zone, {**rest, **interface(100)}, SPIRO], (0, 1)),
    }
    return stacks, {"film": 0, "SnO2": 1, "Spiro": 0}


def with_sublayers(count):
    """`STACKS` with each perovskite split into `count` equal layers (the same material, their
    interfaces open: `nu_int` 1e7 m/s, no traps), so that the solver's per-layer columns give
    the radiative recombination by depth (`detected`): the stacks, and each stack's perovskite
    layers from the glass up."""
    thick = float(PEROVSKITE["L"]) / count
    part = {**PEROVSKITE, "L": f"{thick:.6e}", "nu_int_n": "1e7", "nu_int_p": "1e7"}
    parts = [dict(part) for _ in range(count)]
    stacks = {
        "film": (parts, tuple(range(count))),
        "SnO2": ([{**SNO2, **interface(9)}, *parts], tuple(range(1, count + 1))),
        "Spiro": ([*parts[:-1], {**parts[-1], **interface(100)}, SPIRO], tuple(range(count))),
    }
    return stacks


def detected(columns, layers, alpha, R_back, from_glass, thickness=600e-9, boost=0.0,
             reach=50e-9):  # fmt: skip
    """The PL a detector sees at the wavelength where the perovskite absorbs `alpha` (1/m): each
    perovskite layer's radiative recombination (`JdirL<k>`, layers from the glass up) weighed by
    its light's escape towards the detection side (the glass side or the top), directly and after
    a reflection `R_back` at the far side; reabsorbed light is lost (no photon recycling). With
    `boost`, light from near the bottom leaves better, by `1 + boost·exp(−z/reach)` at a height `z`
    above it, whichever side is lit (a scattering layer under the film)."""
    n = len(layers)
    depth = (np.arange(n) + 0.5) * thickness / n  # each layer's centre, from the glass side
    d = depth if from_glass else thickness - depth
    w = np.exp(-alpha * d) + R_back * np.exp(-alpha * (2 * thickness - d))
    w = w * (1 + boost * np.exp(-depth / reach))
    return sum(w[i] * columns[f"JdirL{k + 1}"] for i, k in enumerate(layers))


def _set(text, key, value):
    new, k = re.subn(rf"^{key}\s*=\s*\S+", f"{key} = {value}", text, flags=re.M)
    if k != 1:
        raise KeyError(key)
    return new


def _files(name, lit_from_glass, fluence, traps=None):
    """The layer files, the traps and the generation profile (G at G_frac = 1, m⁻³ s⁻¹)."""
    layers, k = STACKS[name]
    files = {"traps.txt": TRAPS if traps is None else traps}
    for i, ly in enumerate(layers):
        text = _LAYER
        for key, v in ly.items():
            text = _set(text, key, v)
        files[f"L{i + 1}.txt"] = text
    thick = [float(ly["L"]) for ly in layers]
    first, last = (k[0], k[-1]) if isinstance(k, tuple) else (k, k)  # the absorbing layers
    x0, x1 = sum(thick[:first]), sum(thick[: last + 1])
    x = np.linspace(0, sum(thick), 2000)
    depth = np.clip((x - x0) if lit_from_glass else (x1 - x), 0, None)
    G = np.where((x >= x0) & (x <= x1), fluence * ALPHA * np.exp(-ALPHA * depth) / T_PULSE, 0.0)
    rows = (f"{a:.9e} {b:.9e}" for a, b in zip(x, G, strict=True))
    files["gen.txt"] = "x G_ehp\n" + "\n".join(rows)
    return files


def _setup_text(template, name, W, S, NP, extra=None):
    s = template
    for k in ("l1", "l2", "l3"):
        s = re.sub(rf"^{k} = .*\n", "", s, flags=re.M)
    n = len(STACKS[name][0])
    lines = "".join(f"l{i + 1} = L{i + 1}.txt * layer\n" for i in range(n))
    s = s.replace("**Contacts", lines + "**Contacts", 1)
    vals = {"W_L": W[0], "W_R": W[1], "NP": NP, "genProfile": "gen.txt", "T": 295, **(extra or {})}
    vals.update(zip(("S_n_L", "S_p_L", "S_n_R", "S_p_R"), S, strict=True))
    for k, v in vals.items():
        s = _set(s, k, v)
    return s


def setup(name, W=("free", "free"), S=(0, 0, 0, 0), NP=400, lit_from_glass=True,
          fluence=FLUENCE, traps=None):  # fmt: skip
    """A stack of `STACKS` for jaxdd: its ends free, or electrodes at the work functions `W`
    (eV or 'sfb') with the surface recombination velocities `S` (m/s, SIMsalabim's order);
    `traps` the perovskite's trap file (`TRAPS` by default)."""
    files = _files(name, lit_from_glass, fluence, traps)
    files["simulation_setup.txt"] = _setup_text(_SETUP, name, W, S, NP)
    return setup_from_texts(files)


def times(n=140, t_end=2e-5):
    """The steps: the dark steady state, the pulse, then log-spaced to `t_end`."""
    return np.r_[0.0, T_PULSE, T_PULSE + np.logspace(-11, np.log10(t_end), n)]


def fine_times(n_early=200, n=160, t_end=2e-5):
    """`times` resolving the first bins evenly (for `as_measured`)."""
    early = np.linspace(0, 4e-9, n_early)[1:]
    late = np.logspace(np.log10(4e-9), np.log10(t_end), n)[1:]
    return np.r_[0.0, T_PULSE, T_PULSE + np.unique(np.r_[early, late])]


def pulse(t, V=0.0, oc=False):
    """`tVG` for `t`: light during the pulse, at `V` or the open circuit (`oc`: a bool, or one
    per step)."""
    G = np.where((t > 0) & (t <= T_PULSE * 1.0000001), 1.0, 0.0)
    return TVG(t=t, Vext=np.full(len(t), float(V)), oc=np.broadcast_to(oc, t.shape).copy(),
               G_frac=G, track=np.zeros(len(t), int), dti=np.r_[0.0, 1 / np.diff(t)])  # fmt: skip


def pl_of(columns, n_layers):
    """PL (the direct recombination over every layer) after the pulse, over its value at the end
    of the pulse, and the times after the pulse."""
    pl = sum(columns[f"JdirL{j + 1}"] for j in range(n_layers))
    return pl[1:] / pl[1]


def bin_average(t, pl, starts):
    """Mean of `pl(t)` over `[s, s + BIN)` for each start, from the trapezoidal integral."""
    integral = np.r_[0.0, np.cumsum(0.5 * (pl[1:] + pl[:-1]) * np.diff(t))]
    return (np.interp(starts + BIN, t, integral) - np.interp(starts, t, integral)) / BIN


def as_measured(t, pl, data_t):
    """The model at the measured bins `data_t`, normalised as the data are (to the bin at 0)."""
    return bin_average(t, pl, data_t) / bin_average(t, pl, np.array([0.0]))[0]


def log_bins(t, t_max, bins):
    """Each point's log-spaced bin (index), the first bin [0, 2 ns)."""
    edges = np.r_[0.0, np.logspace(np.log10(BIN), np.log10(t_max), bins)]
    return np.digitize(t, edges) - 1


def binned(y, which):
    """`y` (curves × points) averaged in the bins `which`."""
    out = np.full((y.shape[0], which.max() + 1), np.nan)
    for b in np.unique(which):
        out[:, b] = np.nanmean(y[:, which == b], axis=1)
    return out


def film_data(t_max=1.5e-5):
    """The film's decays: t (s) and 4 × points (glass 4.8e11, glass 4.8e10, film 4.8e11, film
    4.8e10 photons/cm²)."""
    a = np.loadtxt(DATA / "Seo_FAPI_nuts_data_norm.dat", skiprows=1)
    m = a[:, 1] * 1e-9 < t_max
    return a[m, 1] * 1e-9, a[m, 2:].T


def sno2_data(t_max=1.5e-5, background=False):
    """The SnO2 half-stack's decays: t (s) and 3 × points (glass 4.8e11, glass 1.5e11, film 4.8e11
    photons/cm²). With `background` each decay's background taken off: its mean over 15–20 µs,
    beyond the fitted times, where all three are flat (the second only from about 12 µs; the 19
    bins before the pulse's rise show the same level, too few to measure it: 1.4e-3 ± 0.3e-3
    against 1.73e-3 lit from the film, which has a fifth to a quarter of the others' counts at
    its peak, 780 against 3100–3600)."""
    a = np.genfromtxt(DATA / "Seo_SnO2_Glass_0_processed_ode.txt", delimiter="\t", skip_header=1)
    y = a[:, 2:5].T
    if background:
        tail = a[:, 1] * 1e-9 >= 1.5e-5
        y = y - np.nanmean(y[:, tail], axis=1, keepdims=True)
    m = (a[:, 1] >= 0) & (a[:, 1] * 1e-9 < t_max)
    return a[m, 1] * 1e-9, y[:, m]


def spiro_data(t_max=4.5e-6):
    """The Spiro half-stack's raw decay, its background taken off, normalised to its peak, from
    the peak on: t (s) and 1 × points."""
    rows = []
    for line in (DATA / "Seo_Spiro_Perov.dat").read_text(encoding="latin-1").splitlines():
        q = line.split()
        if len(q) >= 2:
            with contextlib.suppress(ValueError):  # the instrument's header
                rows.append((float(q[0]), float(q[1])))
    a = np.array(rows)
    pk = int(np.argmax(a[:, 1]))
    bg = np.median(a[: pk - 5, 1])
    t = (a[pk:, 0] - a[pk, 0]) * 1e-9
    y = ((a[pk:, 1] - bg) / (a[pk, 1] - bg))[None]
    m = t < t_max
    return t[m], y[:, m]


def zimt_case(directory, name, W, S, t, oc_after=False, NP=400):
    """The same stack as a ZimT 5.36 case in `directory`: electrodes at `W` with `S`, the pulse,
    0 V at t = 0 and after it 0 V or the open circuit (`oc_after`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    extra = {"failureMode": 1, "tolPois": "1E-12", "tolDens": "1E-12"}
    (d / "simulation_setup.txt").write_text(_setup_text(_ZIMT, name, W, S, NP, extra))
    for f, text in _files(name, True, FLUENCE).items():
        (d / f).write_text(text)
    rows = ["t Vext G_frac Track"]
    for tt in t:
        g = 1 if 0 < tt <= T_PULSE * 1.0000001 else 0
        rows.append(f"{tt:.9e} {'oc' if (oc_after and tt > 0) else 0} {g} 0")
    (d / "tVG.txt").write_text("\n".join(rows) + "\n")
    return d


def run_zimt(directory, timeout=1800, image="jaxdd/simss:5.36"):
    """ZimT in Docker (`simss/`) on a case: its `tj.dat` as a record array (None if it wrote
    nothing), whether it finished within `timeout` s, and its output."""
    d = Path(directory).resolve()
    name = f"jaxdd-zimt-{os.getpid()}-{abs(hash(str(d))) % 10**8}"
    cmd = ["docker", "run", "--rm", "--name", name, "-u", f"{os.getuid()}:{os.getgid()}",
           "-v", f"{d}:/work", "--entrypoint", "zimt", image, "simulation_setup.txt"]  # fmt: skip
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, finished = p.stdout + p.stderr, True
    except subprocess.TimeoutExpired as e:
        out, finished = (e.stdout or b"").decode(errors="replace"), False
        # The container itself: the timeout stops only the Docker client.
        subprocess.run(["docker", "kill", name], capture_output=True)
    tj = d / "tj.dat"
    text = tj.read_text() if tj.exists() else ""
    if text.count("\n") < 2:
        return None, finished, out
    return np.genfromtxt(io.StringIO(text), names=True), finished, out
