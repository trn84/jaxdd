"""TRPL of the two half-stacks without electrodes, jaxdd with free ends, against SIMsalabim's
electrodes under the choices they force: the work function at the outer surface, S, the voltage;
SIMsalabim's equations as jaxdd solves them (its conditions at the electrodes physically right,
not ZimT 5.36's, docs/differences.md §7). And SIMsalabim's best case: electrodes at the free
stack's own dark Fermi level, blocking, 0 V at t = 0, then the open circuit.

    .venv/bin/python validation/trpl_halfstacks.py      # writes validation/trpl_halfstacks.json

docs/validation.md, "TRPL without electrodes".
"""

import json

import numpy as np
from trpl_common import K_B_T, REPO, pl_of, pulse, setup, times

from jaxdd.batch import prepare
from jaxdd.device import build_device
from jaxdd.solver import solve_state
from jaxdd.transient import solve_transient, tj_columns


def transient(su, t, V=0.0, oc=False):
    """One stack's PL after the pulse and its steps' convergence."""
    r = solve_transient(prepare([build_device(su)]), pulse(t, V, oc), vint=True)
    return pl_of(tj_columns(r, 0), int(r.n_layers[0])), np.asarray(r.converged[0])


def dark_fermi_levels(name):
    """The free stack's dark Fermi level at each end, as an electrode's work function (eV)."""
    dev = build_device(setup(name))
    r = solve_state(dev, g_frac=0.0)
    return tuple(float(dev.E_c[k] + K_B_T * np.log(dev.N_c[k] / r.n[k])) for k in (0, -1))


def main():
    t = times()
    tt = t[1:] - t[1]
    out = {"t": tt.tolist()}
    marks = (1e-9, 1e-8, 1e-7, 1e-6)
    for name, side, works in (("SnO2", 0, (4.1, 4.4, 4.8)), ("Spiro", 1, (5.2, 4.9, 4.6))):
        S01, sfb = (0.1,) * 4, ("sfb", "sfb")
        cases = {"free ends": (dict(S=S01), 0.0, False)}
        for W in works:
            pair = (W, "sfb") if side == 0 else ("sfb", W)
            cases[f"electrodes, outer W = {W} eV, S = 0.1 m/s"] = (dict(W=pair, S=S01), 0.0, False)
        cases["electrodes sfb, ohmic"] = (dict(W=sfb, S=(-1,) * 4), 0.0, False)
        cases["electrodes sfb, S = 0.1 m/s, +0.3 V"] = (dict(W=sfb, S=S01), 0.3, False)
        WL, WR = dark_fermi_levels(name)
        best_W = (f"{WL:.6f}", f"{WR:.6f}")
        best = {"free ends, S = 0": (dict(S=(0,) * 4), 0.0, False),
                "best electrodes": (dict(W=best_W, S=(0,) * 4), 0.0, t > 0)}  # fmt: skip
        out[name] = {"best_W": [WL, WR], "curves": {}}
        print(f"== {name}  (the free stack's dark Fermi level: W_L {WL:.4f}, W_R {WR:.4f} eV)")
        print(f"   {'':44s}" + "".join(f"{m:>10.0e}" for m in marks))
        for label, (kw, V, oc) in {**cases, **best}.items():
            pl, ok = transient(setup(name, **kw), t, V, oc)
            good = np.cumprod(ok[1:]).astype(bool)  # up to the first step not converged
            out[name]["curves"][label] = {"pl": pl.tolist(), "converged_to": int(good.sum())}
            vals = [np.interp(m, tt[good], pl[good]) if good.any() and m <= tt[good][-1] else np.nan
                    for m in marks]  # fmt: skip
            print(f"   {label:44s}" + "".join(f"{v:10.4f}" for v in vals))
        free, el = (np.array(out[name]["curves"][k]["pl"]) for k in best)
        dev = float(np.max(np.abs(np.log10(free / el))))
        out[name]["best_max_log_deviation"] = dev
        print(f"   best electrodes against free ends: max |log10| {dev:.1e}")
    path = REPO / "validation" / "trpl_halfstacks.json"
    path.write_text(json.dumps(out))
    print("wrote", path.relative_to(REPO))


if __name__ == "__main__":
    main()
