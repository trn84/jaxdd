"""Do the shortcuts cost accuracy? The batched solver with them against the same solver strict.

    .venv/bin/python validation/shortcuts.py CASE_DIR [CASE_DIR ...]

The shortcuts (`jaxdd.batch`): the quadratic stop (a small step a hundredth of the one before)
and the settled current; the extrapolated start changes only where Newton begins. `strict=True`
runs Newton to its 1e-10 step instead. Per device, the largest difference in J over the 61
voltages, relative to the device's generation current (the scale `η` is measured in); and the
Newton steps both took.
"""

import sys
from pathlib import Path

import numpy as np

from jaxdd.batch import prepare, solve_batch
from jaxdd.constants import Q
from jaxdd.device import build_device
from jaxdd.simss_input import read_setup

V = np.round(np.arange(-0.2, 1.3 + 1e-9, 0.025), 6)
BATCH = 256


def main(cases: list[Path]) -> None:
    devices = [build_device(read_setup(c / "simulation_setup.txt")) for c in cases]
    gaps, steps, steps_strict, failed = [], [], [], 0
    for i in range(0, len(devices), BATCH):
        part = devices[i : i + BATCH]
        batch = prepare(part)
        J, ok, its = (np.asarray(a) for a in solve_batch(batch, V))
        Js, oks, its_s = (np.asarray(a) for a in solve_batch(batch, V, strict=True))
        failed += int((~ok).sum() + (~oks).sum())
        for k, dev in enumerate(part):
            generated = Q * dev.L * np.sum(dev.h[:-1] * dev.G[:-1]) / np.sum(dev.h[:-1])
            scale = max(generated, np.max(np.abs(Js[k])) * 1e-3, 1.0)
            gaps.append(np.max(np.abs(J[k] - Js[k])) / scale)
        steps.append(its.mean())
        steps_strict.append(its_s.mean())
        print(f"{min(i + BATCH, len(devices))}/{len(devices)}", flush=True)
    gaps = np.array(gaps)
    worst = int(np.argmax(gaps))
    print(f"{len(devices)} devices; shortcuts against strict, |ΔJ| / J_gen: max {gaps.max():.1e}, "
          f"99th pct {np.percentile(gaps, 99):.1e}, median {np.median(gaps):.1e}; "
          f"beyond 1e-6: {(gaps > 1e-6).sum()}, beyond 1e-4: {(gaps > 1e-4).sum()}")  # fmt: skip
    print(f"worst: {cases[worst]}")
    print(f"Newton steps per voltage: {np.mean(steps):.2f} with the shortcuts, "
          f"{np.mean(steps_strict):.2f} strict; unconverged points: {failed}")  # fmt: skip


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
