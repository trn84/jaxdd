"""ZimT 5.36 (SIMsalabim) itself on TRPL without electrodes, in Docker (`simss/`).

1. A decay with an exact answer: a uniform film, radiative recombination alone, both contacts
   blocking (S = 0) at 'sfb', 0 V. Without field or current the film stays uniform, so each
   implicit-Euler step of dn/dt = −k·(n² − n_i²) is known in closed form. jaxdd and ZimT against
   it, and the current ZimT lets through its blocking contacts.
2. SIMsalabim's best case for the two half-stacks (`trpl_halfstacks.py`): electrodes at the free
   stack's dark Fermi level, blocking, 0 V at t = 0, then the open circuit; each ZimT run stopped
   after `--cap` seconds.

    .venv/bin/python validation/trpl_zimt.py [--cap 1800] [--workdir DIR]

writes validation/trpl_zimt.json.

docs/validation.md, "TRPL without electrodes"; docs/differences.md §7.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
from trpl_common import (
    _LAYER,
    _SETUP,
    _ZIMT,
    REPO,
    T_PULSE,
    _set,
    pulse,
    run_zimt,
    setup,
    times,
    zimt_case,
)  # fmt: skip
from trpl_halfstacks import dark_fermi_levels, transient

from jaxdd.batch import prepare
from jaxdd.device import build_device
from jaxdd.simss_input import setup_from_texts
from jaxdd.transient import solve_transient, tj_columns

K_RAD, G0, L = 1e-16, 1e33, 600e-9  # m³/s; m⁻³ s⁻¹ in the pulse (n0 = G0·T_PULSE = 1e22); m


def exact_case(template):
    """The uniform film's setup and layer, as `template` (SimSS's or ZimT's setup) has them."""
    s = template
    for k in ("l2", "l3"):
        s = s.replace(next(line for line in s.splitlines() if line.startswith(k + " ")) + "\n", "")
    vals = {"l1": "L1.txt", "W_L": "sfb", "W_R": "sfb", "S_n_L": 0, "S_p_L": 0, "S_n_R": 0,
            "S_p_R": 0, "T": 295, "NP": 300, "genProfile": "none"}  # fmt: skip
    if template is _ZIMT:
        vals.update(failureMode=1, tolPois="1E-12", tolDens="1E-12")
    for k, v in vals.items():
        s = _set(s, k, v)
    ly = _LAYER
    for k, v in {"L": L, "E_c": 4.0, "E_v": 5.53, "N_c": "2.2e24", "eps_r": 30, "N_t_bulk": 0,
                 "layerGen": 1, "mu_n": "4e-5", "mu_p": "4e-5", "k_direct": K_RAD,
                 "G_ehp": f"{G0:.6e}"}.items():  # fmt: skip
        ly = _set(ly, k, v)
    return s, ly


def exact_decay(t, ni):
    """Implicit Euler in closed form: the pulse's step from the dark, then the decay."""
    G = np.where((t > 0) & (t <= T_PULSE * 1.0000001), G0, 0.0)
    n = [ni]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        c = -(n[-1] + dt * G[i] + dt * K_RAD * ni**2)
        n.append((-1 + np.sqrt(1 - 4 * dt * K_RAD * c)) / (2 * dt * K_RAD))
    n = np.array(n)
    return 1.602176634e-19 * K_RAD * (n**2 - ni**2) * L  # Jdir, A/m²


def part_exact(work, cap):
    t = np.r_[0.0, T_PULSE, T_PULSE + np.logspace(-11, -5, 60)]
    s, ly = exact_case(_SETUP)
    dev = build_device(setup_from_texts({"simulation_setup.txt": s, "L1.txt": ly}))
    exact = exact_decay(t, float(dev.ni[1]))
    r = solve_transient(prepare([dev]), pulse(t))
    jx = tj_columns(r, 0)["JdirL1"]
    d = Path(work) / "exact"
    d.mkdir(parents=True, exist_ok=True)
    zs, zly = exact_case(_ZIMT)
    (d / "simulation_setup.txt").write_text(zs)
    (d / "L1.txt").write_text(zly)
    G = np.where((t > 0) & (t <= T_PULSE * 1.0000001), 1, 0)
    rows = ["t Vext G_frac Track"] + [f"{a:.9e} 0 {g} 0" for a, g in zip(t, G, strict=True)]
    (d / "tVG.txt").write_text("\n".join(rows) + "\n")
    t0 = time.time()
    a, finished, _ = run_zimt(d, cap)
    dev_jaxdd = float(np.max(np.abs(jx[1:] / exact[1:] - 1)))
    out = {"t": (t[1:] - T_PULSE).tolist(), "exact": (exact[1:] / exact[1]).tolist(),
           "jaxdd": (jx[1:] / exact[1]).tolist(), "jaxdd_max_rel": dev_jaxdd,
           "zimt_seconds": time.time() - t0, "zimt_finished": finished}  # fmt: skip
    print(f"exact decay: jaxdd within {out['jaxdd_max_rel']:.1e} of the exact answer")
    if a is not None:
        n = len(a)
        out["zimt"] = (a["JdirL1"][1:] / exact[1]).tolist()
        out["zimt_leak"] = float(np.max(np.abs(np.r_[a["JminLeft"], a["JminRight"]])))
        for i in (2, 20, 30, 40, n - 1):
            print(f"   t {t[i]:.2e}: ZimT {a['JdirL1'][i] / exact[i] - 1:+.2f} of the exact answer")
        print(f"   ZimT's current through its blocking contacts: up to {out['zimt_leak']:.1e} A/m²")
    return out


def part_best(work, cap):
    t = times()
    out = {}
    for name in ("SnO2", "Spiro"):
        WL, WR = dark_fermi_levels(name)
        W = (f"{WL:.6f}", f"{WR:.6f}")
        free, _ = transient(setup(name, S=(0,) * 4), t)
        d = zimt_case(Path(work) / f"best_{name}", name, W, (0,) * 4, t, oc_after=True)
        t0 = time.time()
        a, finished, text = run_zimt(d, cap)
        res = {"seconds": time.time() - t0, "finished": finished, "free": free.tolist(),
               "steps": 0 if a is None else len(a), "of": len(t)}  # fmt: skip
        if a is not None:
            pl = sum(a[k] for k in a.dtype.names if k.startswith("JdirL"))
            res["zimt_t"] = (a["t"][1:] - T_PULSE).tolist()
            res["zimt"] = (pl[1:] / pl[1]).tolist()
            res["zimt_leak"] = float(np.max(np.abs(np.r_[a["JminLeft"], a["JminRight"]])))
        lines = text.splitlines()
        for i, line in enumerate(lines):  # FPC's two lines: where, then which exception
            if "exception occurred" in line.lower():
                res["error"] = " ".join(x.strip() for x in lines[i : i + 2])
                break
        out[name] = res
        msg = res.get("error") or f"{res['steps']} of {len(t)} steps in {res['seconds']:.0f} s"
        print(f"{name}, best case: ZimT {msg}", "" if finished else f"(stopped after {cap} s)")
        if "zimt" in res:
            tz = np.array(res["zimt_t"])
            k = min(np.searchsorted(tz, 1e-8), len(tz) - 1)
            mine = np.interp(tz[k], t[1:] - T_PULSE, free)
            print(f"   at {tz[k]:.1e} s: ZimT {res['zimt'][k]:.4f}, jaxdd (free ends) {mine:.4f};"
                  f" leak up to {res['zimt_leak']:.1e} A/m²")  # fmt: skip
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", type=float, default=1800.0, help="seconds per ZimT run")
    ap.add_argument("--workdir", help="where ZimT's cases go (default: a temporary directory)")
    a = ap.parse_args()
    work = a.workdir or tempfile.mkdtemp(prefix="trpl_zimt_")
    out = {"exact": part_exact(work, a.cap), "best": part_best(work, a.cap)}
    path = REPO / "validation" / "trpl_zimt.json"
    path.write_text(json.dumps(out))
    print("wrote", path.relative_to(REPO), "; ZimT's cases in", work)


if __name__ == "__main__":
    main()
