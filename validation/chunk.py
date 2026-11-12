"""A batch saved to a file, solved and timed: a real workload carried to another machine.

    python validation/chunk.py batch.npz [first_max_iter]

`save` writes a batch (`jaxdd.batch.prepare`) and its voltages to one `.npz`, on the machine
that builds the devices; this script loads it on the machine that solves it, calls
`solve_batch` twice (the first call compiles) and reports the second: seconds, curves per
second, the Newton steps a voltage needed on average and paid in lockstep, and the voltages
left unconverged. `first_max_iter` as `solve_batch`'s (default 6: two passes); `JAXDD_SWEEP`
chooses the sweep as everywhere.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxdd.batch import solve_batch

_SEP = "/"


def save(path, batch: dict, voltages) -> None:
    """The batch's leaves under their paths in the tree (`rec/kd`, …), and the voltages."""
    leaves = {}

    def put(prefix, tree):
        for key, value in tree.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                put(name + _SEP, value)
            else:
                leaves[name] = np.asarray(value)

    put("", batch)
    np.savez_compressed(path, __voltages__=np.asarray(voltages), **leaves)


def load(path) -> tuple[dict, np.ndarray]:
    """`save`'s batch, on the default device, and its voltages."""
    batch: dict = {}
    with np.load(path) as f:
        voltages = f["__voltages__"]
        for name in f.files:
            if name == "__voltages__":
                continue
            *parents, leaf = name.split(_SEP)
            node = batch
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = jnp.asarray(f[name])
    return batch, voltages


def main(path: str, first_max_iter: int = 6) -> None:
    batch, V = load(path)
    n = batch["x"].shape[0]
    print(f"JAX {jax.__version__} on {jax.devices()}; {n} devices, {batch['x'].shape[1]} grid "
          f"points, {V.size} voltages, first_max_iter {first_max_iter}")  # fmt: skip
    for label in ("first call (compiles)", "second call"):
        start = time.perf_counter()
        J, ok, its = solve_batch(batch, V, first_max_iter=first_max_iter)
        J, ok, its = np.asarray(J), np.asarray(ok), np.asarray(its)
        seconds = time.perf_counter() - start
        print(f"  {label:22s} {seconds:7.1f} s  {n / seconds:7.0f} curves/s")
    needed, paid = its.mean(axis=0).sum() / V.size, its.max(axis=0).sum() / V.size
    print(
        f"  Newton steps per voltage: {needed:.2f} needed, {paid:.2f} paid; "
        f"max {its.max()} at one voltage; unconverged voltages {int((~ok).sum())}"
    )


if __name__ == "__main__":
    main(sys.argv[1], *(int(a) for a in sys.argv[2:3]))
