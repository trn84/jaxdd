"""Where a Newton step's time goes, part by part, on whatever JAX runs (for the A100).

    python validation/breakdown.py [batch]      # e.g. 16384 on an A100, 256 on a laptop

Each part of one Newton step of `jaxdd.batch` is jitted on its own over a batch of `bench.py`'s
devices and timed after compilation: the residual, the Jacobian blocks and the ions' columns
(autodiff of each row), the ions' constraint, the linear solve (sweep, three right-hand sides),
and the current; the solver assembles the first three by edges (`_assemble`), and the
autodiff of each row it replaced is timed beside it; on a GPU the sweep both as the scan and
as the Triton kernel (jaxdd.sweep) at three tiles and reading the blocks in the solver's layout
(`transposed=False`), and as the Mosaic GPU kernel at two tiles in both layouts. The part that
dominates is the candidate for a kernel of its own (PLAN.md, section 1.2).
"""

import functools
import sys
import time

import jax
import jax.numpy as jnp
from bench import variants

from jaxdd.batch import _bounds, _current, _initial, prepare
from jaxdd.solver import BERNOULLI, _assemble, _block_thomas, _ion_constraints, _kernels
from jaxdd.sweep import (
    MOSAIC_TILE,
    SWEEP_TILE,
    kernel_available,
    mosaic_sweep,
    pallas_sweep,
    sweep_kind,
)

REPEATS = 5


def timed(name, fn, *args):
    jax.block_until_ready(fn(*args))  # compiles
    start = time.perf_counter()
    for _ in range(REPEATS):
        out = fn(*args)
    jax.block_until_ready(out)
    ms = (time.perf_counter() - start) / REPEATS * 1e3
    print(f"  {name:38s} {ms:9.2f} ms")
    return ms


def main(batch_size: int = 256) -> None:
    print(f"JAX {jax.__version__} on {jax.devices()}; batch {batch_size}")
    batch = prepare(variants(batch_size))
    row_val, row_jac, row_ions = _kernels(BERNOULLI["exact"])
    Vint = 0.5

    def state(d):
        bounds = _bounds(d, Vint)
        y = _initial(d, Vint)[1:-1]
        full = jnp.concatenate([bounds[0][None], y, bounds[1][None]])
        return full, jnp.zeros(2)

    def residual(d):
        full, logs = state(d)
        args = (full[:-2], full[1:-1], full[2:], d["s"], d["Vt"], d["g_frac"], logs[0], logs[1])
        return row_val(*args)

    def jacobian(d):
        full, logs = state(d)
        args = (full[:-2], full[1:-1], full[2:], d["s"], d["Vt"], d["g_frac"], logs[0], logs[1])
        return row_jac(*args), row_ions(*args)

    def assembled(d):
        full, logs = state(d)
        return _assemble(full, d["s"], d["Vt"], d["g_frac"], logs[0], logs[1],
                         BERNOULLI["exact"])  # fmt: skip

    def constraint(d):
        full, logs = state(d)

        def g(V_interior, logs_):
            V_full = jnp.concatenate([full[:1, 0], V_interior, full[-1:, 0]])
            return _ion_constraints(V_full, logs_[0], logs_[1], d["ion"])

        return jax.jacrev(g, argnums=(0, 1))(full[1:-1, 0], logs)

    blocks = jax.jit(jax.vmap(jacobian))(batch)
    (A, B, C), U = blocks
    F = jax.jit(jax.vmap(residual))(batch)
    rhs = jnp.concatenate([-F[..., None], jnp.stack(U, -1)], axis=-1)

    def current(d):
        full, _ = state(d)
        return _current(d, full, d["g_frac"])

    timed("residual (rows, the reference)", jax.jit(jax.vmap(residual)), batch)
    timed("blocks + ion columns (jacfwd, reference)", jax.jit(jax.vmap(jacobian)), batch)
    total = 0.0
    total += timed("residual, blocks, ion columns (edges)", jax.jit(jax.vmap(assembled)), batch)
    total += timed("ion constraint (jacrev)", jax.jit(jax.vmap(constraint)), batch)
    scan = timed("linear solve (block sweep, scan)", jax.jit(jax.vmap(_block_thomas)), A, B, C, rhs)
    tiles, mosaic = {}, {}
    if kernel_available("pallas"):
        tiles = {
            tile: timed(f"linear solve (sweep kernel, tile {tile})",
                        jax.jit(functools.partial(pallas_sweep, tile=tile)), A, B, C, rhs)
            for tile in (64, 128, 256)
        }  # fmt: skip
        timed(f"linear solve (kernel, {SWEEP_TILE}, solver's layout)",
              jax.jit(functools.partial(pallas_sweep, transposed=False)), A, B, C, rhs)  # fmt: skip
    if kernel_available("mosaic"):
        for tile in (128, 256):
            for transposed in (True, False):
                layout = "" if transposed else ", solver's layout"
                mosaic[tile, transposed] = timed(
                    f"linear solve (Mosaic, tile {tile}{layout})",
                    jax.jit(functools.partial(mosaic_sweep, tile=tile, transposed=transposed)),
                    A, B, C, rhs)  # fmt: skip
    used = {"pallas": tiles.get(SWEEP_TILE), "mosaic": mosaic.get((MOSAIC_TILE, True))}
    total += used.get(sweep_kind()) or scan
    total += timed("current", jax.jit(jax.vmap(current)), batch)
    print(f"  {'sum of the parts (the sweep the solver uses)':38s} {total:9.2f} ms")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
