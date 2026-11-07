"""Where a Newton step's time goes, part by part, on whatever JAX runs (for the A100).

    python validation/breakdown.py [batch]      # e.g. 16384 on an A100, 256 on a laptop

Each part of one Newton step of `jaxdd.batch` is jitted on its own over a batch of `bench.py`'s
devices and timed after compilation: the residual, the Jacobian blocks and the ions' columns
(autodiff of each row), the ions' constraint, the linear solve (sweep, three right-hand sides),
the current, and the whole step. The part that dominates is the candidate for a kernel of its
own (PLAN.md §8a).
"""

import sys
import time

import jax
import jax.numpy as jnp
from bench import variants

from jaxdd.batch import _bounds, _current, _initial, prepare
from jaxdd.solver import BERNOULLI, LINEAR_SOLVERS, _ion_constraints, _kernels

REPEATS = 5


def timed(name, fn, *args):
    out = fn(*args)
    jax.block_until_ready(out)
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

    total = 0.0
    total += timed("residual (rows)", jax.jit(jax.vmap(residual)), batch)
    total += timed("Jacobian blocks + ion columns (jacfwd)", jax.jit(jax.vmap(jacobian)), batch)
    total += timed("ion constraint (jacrev)", jax.jit(jax.vmap(constraint)), batch)
    for name, solve in LINEAR_SOLVERS.items():
        ms = timed(f"linear solve: {name}", jax.jit(jax.vmap(solve)), A, B, C, rhs)
        if name == "thomas":
            total += ms
    total += timed("current", jax.jit(jax.vmap(current)), batch)
    print(f"  {'sum of the parts (with the sweep)':38s} {total:9.2f} ms")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
