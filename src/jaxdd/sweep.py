"""The block Thomas sweep of a whole batch in one GPU kernel (Pallas), and the switch to it.

`solver._block_thomas` solves one device's block-tridiagonal system by a `lax.scan` over its
grid points. Under `vmap` that scan becomes two XLA while loops of about 50 iterations each
(400 points, unrolled by 8), and every iteration launches some 30 kernels: about 1450 launches
per sweep, which on an A100 cost more than the arithmetic (docs/solver.md, "The block sweep").
Here the whole sweep, forward and back, is one kernel: each program takes a tile of devices and
runs over every grid point itself, with the carry in registers.

The kernel wants the devices on the last axis, so that a tile's loads are contiguous: each
block's nine entries (and the right-hand sides') become `(entries, NP, devices)`. The transposes
in and out are half of its time, and it is still faster than the scan. With
`transposed=False` it reads the blocks where the solver wrote them, `(devices, NP, entries)`,
by strided loads and without the transposes: faster on an RTX 4070 Ti SUPER up to batch 8192
and even at 16 384, slower on an A100 (9.4 against 7.1 ms at 16 384), hence not the default
(docs/solver.md; validation/breakdown.py times both).

The same kernel body runs in two Pallas backends: Triton (`pallas_sweep`, which JAX 0.12 marks
deprecated) and Mosaic GPU (`mosaic_sweep`, its successor; NVIDIA only, at least one warpgroup
of 128 devices per program).

`block_sweep` is the solver's sweep: for one device the scan; under `vmap`, on an NVIDIA GPU,
the Triton kernel (`custom_vmap`), and the scan everywhere else (CPU, ROCm). `JAXDD_SWEEP`
overrides the choice: `scan`, `pallas` (Triton) or `mosaic` (`auto` by default).
"""

from __future__ import annotations

import contextlib
import os

import jax
import jax.numpy as jnp

# Pallas here, not where the kernel is built: its import registers configuration states that
# enter the cache key of every `jax.jit`. Imported during the solver's first trace, it made
# that trace's key stale, and the second call of the solver compiled it once more
# (docs/solver.md, "The block Thomas sweep").
with contextlib.suppress(ImportError):  # a JAX without Pallas: the scan
    import jax.experimental.pallas
with contextlib.suppress(ImportError):  # a JAX without Mosaic GPU: Triton or the scan
    import jax.experimental.pallas.mosaic_gpu

from jaxdd.solver import _block_thomas

__all__ = [
    "MOSAIC_TILE",
    "SWEEP_TILE",
    "SWEEP_TRANSPOSED",
    "block_sweep",
    "mosaic_sweep",
    "pallas_sweep",
    "sweep_kind",
]

#: Devices per kernel program. 64 and 32 were equal on an RTX 4070 Ti SUPER, 128 and 256
#: slower at batch 4096 (validation/breakdown.py times the tiles on the machine at hand).
SWEEP_TILE = 64
#: The blocks transposed to `(entries, NP, devices)` for the kernel (contiguous loads), or read
#: as the solver has them (strided loads, no transposes).
SWEEP_TRANSPOSED = True
#: Devices per Mosaic GPU program: one warpgroup (128 threads), the least it takes. 256 was
#: equal at batch 16 384 on an RTX 4070 Ti SUPER and slower at 4096.
MOSAIC_TILE = 128
#: Warps per program: one thread per device of the tile.
_WARPS = {32: 1, 64: 2, 128: 4, 256: 8}


def sweep_kind() -> str:
    """`"pallas"`, `"mosaic"` or `"scan"`: `JAXDD_SWEEP` if set, else the Triton kernel on an
    NVIDIA GPU."""
    choice = os.environ.get("JAXDD_SWEEP", "auto")
    if choice in ("scan", "pallas", "mosaic"):
        return choice
    if choice != "auto":
        raise ValueError(f"JAXDD_SWEEP = {choice!r}: auto, scan, pallas or mosaic")
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


# Lists of (tile,) vectors: one per entry of a 3×3 block, or of 3×k right-hand sides. `at`
# gives the index of entry `e` at grid point `i` in the layout at hand; `cols` selects the
# program's devices (all of a Triton program's block, a slice of the whole array in Mosaic).
def _at(transposed, cols=slice(None)):
    if transposed:
        return lambda e, i: (e, i, cols)
    return lambda e, i: (cols, i, e)


def _load(ref, i, cols, at):
    return [[ref[at(cols * r + c, i)] for c in range(cols)] for r in range(3)]


def _store(ref, i, m, at):
    for r, row in enumerate(m):
        for c, v in enumerate(row):
            ref[at(len(row) * r + c, i)] = v


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


def _kernel(NP, k, transposed, columns=None):
    """Forward: `m = B − A·c'`, `c' = m⁻¹·C`, `d' = m⁻¹·(d − A·d')`, kept in the outputs; back:
    `x = d' − c'·x_next`, over `d'` in place (`solver._block_thomas`). `columns`, called in the
    kernel, gives the program's devices and their number (default: the whole block)."""

    def kernel(a_ref, b_ref, c_ref, d_ref, x_ref, cp_ref):
        if columns is None:
            cols, tile = slice(None), a_ref.shape[-1 if transposed else 0]
        else:
            cols, tile = columns()
        at = _at(transposed, cols)
        zero = jnp.zeros((tile,), a_ref.dtype)

        def forward(i, carry):
            cp, dp = _rows(carry[:9], 3), _rows(carry[9:], k)
            a = _load(a_ref, i, 3, at)
            m_inv = _inverse3(_sub(_load(b_ref, i, 3, at), _mm(a, cp)))
            cn = _mm(m_inv, _load(c_ref, i, 3, at))
            dn = _mm(m_inv, _sub(_load(d_ref, i, k, at), _mm(a, dp)))
            _store(cp_ref, i, cn, at)
            _store(x_ref, i, dn, at)
            return _flat(cn) + _flat(dn)

        jax.lax.fori_loop(0, NP, forward, (zero,) * (9 + 3 * k))

        def backward(j, x_next):
            i = NP - 1 - j
            x = _sub(_load(x_ref, i, k, at), _mm(_load(cp_ref, i, 3, at), _rows(x_next, k)))
            _store(x_ref, i, x, at)
            return _flat(x)

        jax.lax.fori_loop(0, NP, backward, (zero,) * (3 * k))

    return kernel


def pallas_sweep(A, B, C, d, tile: int = SWEEP_TILE, interpret: bool = False,
                 transposed: bool | None = None):  # fmt: skip
    """`_block_thomas` for a batch in one kernel: `A, B, C` (n × NP × 3 × 3), `d`
    (n × NP × 3 × k); returns `x` like `d`. `interpret` runs it without a GPU (tests);
    `transposed` as `SWEEP_TRANSPOSED`, which is its default."""
    from jax.experimental import pallas as pl

    transposed = SWEEP_TRANSPOSED if transposed is None else transposed
    _, NP, _, k = d.shape
    ins, N, shape, unlay = _lay(A, B, C, d, tile, transposed)

    def spec(rows):
        if transposed:
            return pl.BlockSpec((rows, NP, tile), lambda j: (0, 0, j))
        return pl.BlockSpec((tile, NP, rows), lambda j: (j, 0, 0))

    params = {}
    if not interpret:
        from jax.experimental.pallas import triton as plgpu

        params["compiler_params"] = plgpu.CompilerParams(num_warps=_WARPS.get(tile, 2))
    x, _ = pl.pallas_call(
        _kernel(NP, k, transposed),
        out_shape=(
            jax.ShapeDtypeStruct(shape(3 * k), d.dtype),
            jax.ShapeDtypeStruct(shape(9), d.dtype),
        ),
        grid=(N // tile,),
        in_specs=[spec(9), spec(9), spec(9), spec(3 * k)],
        out_specs=(spec(3 * k), spec(9)),
        interpret=interpret,
        **params,
    )(*ins)
    return unlay(x)


def _lay(A, B, C, d, tile, transposed):
    """The kernels' inputs: each block's entries flat, the batch padded to whole tiles, and
    transposed to `(entries, NP, devices)` or not; the padded batch, the output shape for a
    number of entries, and the way back for the solution."""
    n, NP, _, k = d.shape
    pad = (-n) % tile
    # The blocks once, in the solver's layout: without the barrier XLA fuses their whole
    # assembly into each of the four transposes (or paddings) below, and computes it four times.
    barrier = getattr(jax.lax, "optimization_barrier", lambda x: x)  # not in JAX 0.4.31
    A, B, C, d = barrier((A, B, C, d))

    def lay(M, cols):  # n × NP × 3 × cols → (n + pad) × NP × 3·cols, or its transpose
        M = M.reshape(n, NP, 3 * cols)
        if pad:
            M = jnp.concatenate([M, jnp.broadcast_to(M[-1:], (pad, NP, 3 * cols))])
        return M.transpose(2, 1, 0) if transposed else M

    N = n + pad

    def shape(rows):
        return (rows, NP, N) if transposed else (N, NP, rows)

    def unlay(x):
        x = x.transpose(2, 1, 0) if transposed else x
        return x[:n].reshape(n, NP, 3, k)

    return (lay(A, 3), lay(B, 3), lay(C, 3), lay(d, k)), N, shape, unlay


def mosaic_sweep(A, B, C, d, tile: int = MOSAIC_TILE, transposed: bool | None = None):
    """`pallas_sweep` in Pallas's Mosaic GPU backend (NVIDIA only): the same kernel, its refs in
    global memory, each program's devices a slice of them. `tile` a multiple of 128 (a
    warpgroup); `transposed` as `SWEEP_TRANSPOSED`, which is its default."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import mosaic_gpu as plgpu

    transposed = SWEEP_TRANSPOSED if transposed is None else transposed
    k = d.shape[-1]
    ins, N, shape, unlay = _lay(A, B, C, d, tile, transposed)

    def columns():
        return pl.ds(jax.lax.axis_index("tiles") * tile, tile), tile

    x, _ = plgpu.kernel(
        _kernel(d.shape[1], k, transposed, columns),
        out_type=(
            jax.ShapeDtypeStruct(shape(3 * k), d.dtype),
            jax.ShapeDtypeStruct(shape(9), d.dtype),
        ),
        grid=(N // tile,),
        grid_names=("tiles",),
    )(*ins)
    return unlay(x)


@jax.custom_batching.custom_vmap
def block_sweep(A, B, C, d):
    """The solver's block sweep: `solver._block_thomas` for one device; for a batch the kernel
    or the scan (`sweep_kind`)."""
    return _block_thomas(A, B, C, d)


@block_sweep.def_vmap
def _block_sweep_vmap(axis_size, in_batched, A, B, C, d):
    A, B, C, d = (x if batched else jnp.broadcast_to(x, (axis_size, *x.shape))
                  for x, batched in zip((A, B, C, d), in_batched, strict=True))  # fmt: skip
    kind = sweep_kind()
    if kind == "pallas":
        return pallas_sweep(A, B, C, d), True
    if kind == "mosaic":
        return mosaic_sweep(A, B, C, d), True
    return jax.vmap(_block_thomas)(A, B, C, d), True
