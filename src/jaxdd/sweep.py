"""The block Thomas sweep of a whole batch in one GPU kernel (Pallas), and the switch to it.

`solver._block_thomas` solves one device's block-tridiagonal system by a `lax.scan` over its
grid points. Under `vmap` that scan becomes two XLA while loops of about 50 iterations each
(400 points, unrolled by 8), and every iteration launches some 30 kernels: about 1450 launches
per sweep, which on an A100 cost more than the arithmetic (docs/solver.md, "The block sweep").
Here the whole sweep, forward and back, is one kernel: each program takes a tile of devices and
runs over every grid point itself, with the carry in registers.

The kernel wants the devices on the last axis, so that a tile's loads are contiguous: each
block's nine entries (and the right-hand sides') become `(entries, NP, devices)`. The transposes
in and out are part of its cost, and it is still faster (docs/solver.md).

`block_sweep` is the solver's sweep: for one device the scan; under `vmap`, on an NVIDIA GPU,
the kernel (`custom_vmap`), and the scan everywhere else (CPU, ROCm). `JAXDD_SWEEP` overrides
the choice: `scan` or `pallas` (`auto` by default).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from jaxdd.solver import _block_thomas

__all__ = ["SWEEP_TILE", "block_sweep", "pallas_sweep", "sweep_kind"]

#: Devices per kernel program. 64 and 32 were equal on an RTX 4070 Ti SUPER, 128 and 256
#: slower at batch 4096 (validation/breakdown.py times the tiles on the machine at hand).
SWEEP_TILE = 64
#: Warps per program: one thread per device of the tile.
_WARPS = {32: 1, 64: 2, 128: 4, 256: 8}


def sweep_kind() -> str:
    """`"pallas"` or `"scan"`: `JAXDD_SWEEP` if set, else the kernel on an NVIDIA GPU."""
    choice = os.environ.get("JAXDD_SWEEP", "auto")
    if choice in ("scan", "pallas"):
        return choice
    if choice != "auto":
        raise ValueError(f"JAXDD_SWEEP = {choice!r}: auto, scan or pallas")
    device = jax.devices()[0]
    cuda = device.platform == "gpu" and "cuda" in device.client.platform_version.lower()
    return "pallas" if cuda and _triton_available() else "scan"


def _triton_available() -> bool:
    """Pallas's Triton backend, which JAX 0.12 marks deprecated: without it, the scan."""
    try:
        from jax.experimental.pallas import triton  # noqa: F401
    except ImportError:
        return False
    return True


# Lists of (tile,) vectors: one per entry of a 3×3 block, or of 3×k right-hand sides.
def _load(ref, i, cols):
    return [[ref[cols * r + c, i, :] for c in range(cols)] for r in range(3)]


def _store(ref, i, m):
    for r, row in enumerate(m):
        for c, v in enumerate(row):
            ref[len(row) * r + c, i, :] = v


def _mm(a, b):
    return [[a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c] for c in range(len(b[0]))]
            for r in range(3)]  # fmt: skip


def _sub(a, b):
    return [[x - y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def _inverse3(m):
    """The adjugate over the determinant, as `solver._inverse3`."""
    (a, b, c), (d, e, f), (g, h, i) = m
    co = [[e * i - f * h, c * h - b * i, b * f - c * e],
          [f * g - d * i, a * i - c * g, c * d - a * f],
          [d * h - e * g, b * g - a * h, a * e - b * d]]  # fmt: skip
    inv = 1.0 / (a * co[0][0] + b * co[1][0] + c * co[2][0])
    return [[x * inv for x in row] for row in co]


def _flat(m):
    return tuple(x for row in m for x in row)


def _rows(t, cols):
    return [list(t[cols * r : cols * r + cols]) for r in range(3)]


def _kernel(NP, k):
    """Forward: `m = B − A·c'`, `c' = m⁻¹·C`, `d' = m⁻¹·(d − A·d')`, kept in the outputs; back:
    `x = d' − c'·x_next`, over `d'` in place (`solver._block_thomas`)."""

    def kernel(a_ref, b_ref, c_ref, d_ref, x_ref, cp_ref):
        zero = jnp.zeros((a_ref.shape[-1],), a_ref.dtype)

        def forward(i, carry):
            cp, dp = _rows(carry[:9], 3), _rows(carry[9:], k)
            a = _load(a_ref, i, 3)
            m_inv = _inverse3(_sub(_load(b_ref, i, 3), _mm(a, cp)))
            cn = _mm(m_inv, _load(c_ref, i, 3))
            dn = _mm(m_inv, _sub(_load(d_ref, i, k), _mm(a, dp)))
            _store(cp_ref, i, cn)
            _store(x_ref, i, dn)
            return _flat(cn) + _flat(dn)

        jax.lax.fori_loop(0, NP, forward, (zero,) * (9 + 3 * k))

        def backward(j, x_next):
            i = NP - 1 - j
            x = _sub(_load(x_ref, i, k), _mm(_load(cp_ref, i, 3), _rows(x_next, k)))
            _store(x_ref, i, x)
            return _flat(x)

        jax.lax.fori_loop(0, NP, backward, (zero,) * (3 * k))

    return kernel


def pallas_sweep(A, B, C, d, tile: int = SWEEP_TILE, interpret: bool = False):
    """`_block_thomas` for a batch in one kernel: `A, B, C` (n × NP × 3 × 3), `d`
    (n × NP × 3 × k); returns `x` like `d`. `interpret` runs it without a GPU (tests)."""
    from jax.experimental import pallas as pl

    n, NP, _, k = d.shape
    pad = (-n) % tile
    # The blocks once, in the solver's layout: without the barrier XLA fuses their whole
    # assembly into each of the four transposes below, and computes it four times.
    barrier = getattr(jax.lax, "optimization_barrier", lambda x: x)  # not in JAX 0.4.31
    A, B, C, d = barrier((A, B, C, d))

    def lay(M, cols):  # n × NP × 3 × cols → 3·cols × NP × (n + pad)
        M = M.reshape(n, NP, 3 * cols)
        if pad:
            M = jnp.concatenate([M, jnp.broadcast_to(M[-1:], (pad, NP, 3 * cols))])
        return M.transpose(2, 1, 0)

    N = n + pad

    def spec(rows):
        return pl.BlockSpec((rows, NP, tile), lambda j: (0, 0, j))

    params = {}
    if not interpret:
        from jax.experimental.pallas import triton as plgpu

        params["compiler_params"] = plgpu.CompilerParams(num_warps=_WARPS.get(tile, 2))
    x, _ = pl.pallas_call(
        _kernel(NP, k),
        out_shape=(
            jax.ShapeDtypeStruct((3 * k, NP, N), d.dtype),
            jax.ShapeDtypeStruct((9, NP, N), d.dtype),
        ),
        grid=(N // tile,),
        in_specs=[spec(9), spec(9), spec(9), spec(3 * k)],
        out_specs=(spec(3 * k), spec(9)),
        interpret=interpret,
        **params,
    )(lay(A, 3), lay(B, 3), lay(C, 3), lay(d, k))
    return x.transpose(2, 1, 0)[:n].reshape(n, NP, 3, k)


@jax.custom_batching.custom_vmap
def block_sweep(A, B, C, d):
    """The solver's block sweep: `solver._block_thomas` for one device; for a batch the kernel
    or the scan (`sweep_kind`)."""
    return _block_thomas(A, B, C, d)


@block_sweep.def_vmap
def _block_sweep_vmap(axis_size, in_batched, A, B, C, d):
    A, B, C, d = (x if batched else jnp.broadcast_to(x, (axis_size, *x.shape))
                  for x, batched in zip((A, B, C, d), in_batched, strict=True))  # fmt: skip
    if sweep_kind() == "pallas":
        return pallas_sweep(A, B, C, d), True
    return jax.vmap(_block_thomas)(A, B, C, d), True
