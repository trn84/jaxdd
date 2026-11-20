"""The carriers' statistics: the density of a band at the reduced quasi-Fermi level
`η = (E_F − E_c)/(k T)` (electrons; holes with `E_v − E_F`), `n = N_c·F(η)`.

- `"boltzmann"`: `F(η) = exp(η)`, SIMsalabim's and jaxdd's default.
- `"blakemore"`: `F(η) = 1/(exp(−η) + γ)`, `γ = 0.27` (Blakemore, Solid-State Electron. 25,
  1067 (1982)): within a few percent of Fermi–Dirac's `F_{1/2}` to `η ≈ 1.3`, and bounded by
  `N_c/γ`.
- `"fermi-dirac"`: the Fermi–Dirac integral of order 1/2, normalised to `exp(η)` far below the
  band, by TeSCA's approximation (as ChargeTransport.jl's `FermiDiracOneHalfTeSCA`).

The solver keeps `ln n` as the unknown; what a statistics changes is the electrochemical
potential, `η = ln(n/N_c) + w`, with the excess chemical potential `w(ln(n/N_c)) = F⁻¹(n/N_c) −
ln(n/N_c)`, 0 for Boltzmann's (`excess`). The flux at a constant mobility follows `η` (the
Scharfetter–Gummel flux with `w`'s difference beside the potential's, docs/model.md), the
contacts' densities are `N_c·F(η)`, and a carrier's equilibrium density product `n_i²·e^(−w_n
− w_p)` (the recombination's).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

STATISTICS = ("boltzmann", "blakemore", "fermi-dirac")

#: Blakemore's γ.
BLAKEMORE_GAMMA = 0.27

#: TeSCA's two pieces meet here.
_TESCA_BREAK = 1.6107


def _tesca(eta, xp=jnp):
    """TeSCA's `F_{1/2}(η)` (normalised: `exp(η)` far below the band), `xp` NumPy or JAX."""
    low = xp.log1p(xp.exp(xp.minimum(eta, _TESCA_BREAK)))
    low = (1 + 0.16 * low) * low
    hi_arg = xp.maximum(eta, _TESCA_BREAK) ** 0.75
    # log1p(exp(z)) without overflow: z + log1p(exp(−z))
    z = hi_arg + xp.log1p(xp.exp(-hi_arg))
    high = 0.3258 - (0.0321 - 0.7523 * z) * z
    return xp.where(eta < _TESCA_BREAK, low, high)


def _tesca_dlog(eta):
    """`d ln F/dη` of TeSCA's `F_{1/2}`, both pieces in closed form."""
    z_low = jnp.log1p(jnp.exp(jnp.minimum(eta, _TESCA_BREAK)))
    low = (1 + 0.32 * z_low) * jax.nn.sigmoid(eta) / ((1 + 0.16 * z_low) * z_low)
    t = jnp.maximum(eta, _TESCA_BREAK)
    hi_arg = t**0.75
    z = hi_arg + jnp.log1p(jnp.exp(-hi_arg))
    F = 0.3258 - (0.0321 - 0.7523 * z) * z
    high = (1.5046 * z - 0.0321) * jax.nn.sigmoid(hi_arg) * 0.75 * t**-0.25 / F
    return jnp.where(eta < _TESCA_BREAK, low, high)


#: Newton's steps on TeSCA's `ln F` (`_fd_inverse`): from its start to round-off in 4.
FD_INVERSE_STEPS = 5


def density(eta, kind: str, xp=jnp):
    """`F(η)`: a band's density over its `N_c` at the reduced quasi-Fermi level `η`."""
    if kind == "boltzmann":
        return xp.exp(eta)
    if kind == "blakemore":
        return 1.0 / (xp.exp(-eta) + BLAKEMORE_GAMMA)
    if kind == "fermi-dirac":
        # far below the band TeSCA's log1p(exp(η)) loses exp(η) to round-off: exp(η) there
        return xp.where(eta < -30.0, xp.exp(eta), _tesca(eta, xp))
    raise ValueError(f"statistics {kind!r}: one of {STATISTICS}")


def _fd_inverse(ln_u):
    """`F_{1/2}⁻¹(u)` (TeSCA's) at `ln u`: a start (Joyce–Dixon's series below the band, the
    degenerate limit above), then Newton on `ln F` in `η`, whose slope is between 0 and 1."""
    u = jnp.exp(ln_u)
    a1, a2, a3 = 3.53553e-1, -4.95009e-3, 1.48386e-4
    series = ln_u + u * (a1 + u * (a2 + u * a3))
    degenerate = (0.75 * np.sqrt(np.pi) * u) ** (2.0 / 3.0)
    eta = jnp.where(u < 4.0, series, degenerate)
    for _ in range(FD_INVERSE_STEPS):
        eta = eta - (jnp.log(_tesca(eta)) - ln_u) / _tesca_dlog(eta)
    return jnp.where(ln_u < -30.0, ln_u, eta)


@jax.custom_jvp
def _fd_excess(ln_u):
    return _fd_inverse(ln_u) - ln_u


@_fd_excess.defjvp
def _fd_excess_jvp(primals, tangents):
    (ln_u,), (t,) = primals, tangents
    eta = _fd_inverse(ln_u)
    # dη/d ln u = u/F'(η) = F/F' at the solution: 1/(d ln F/dη); far below the band 1
    dlnF = jnp.where(ln_u < -30.0, 1.0, _tesca_dlog(eta))
    return eta - ln_u, (1.0 / dlnF - 1.0) * t


def excess(ln_u, kind: str):
    """The excess chemical potential `w = F⁻¹(u) − ln u` at `ln u = ln(n/N_c)`: 0 for
    Boltzmann's, `−ln(1 − γ·u)` for Blakemore's (its share capped below `1/γ`, so that a Newton
    step past the bound leaves the logarithm defined), TeSCA's `F_{1/2}`'s by Newton."""
    if kind == "boltzmann":
        return jnp.zeros_like(ln_u)
    if kind == "blakemore":
        share = jnp.minimum(BLAKEMORE_GAMMA * jnp.exp(ln_u), 1.0 - 1e-12)
        return -jnp.log1p(-share)
    if kind == "fermi-dirac":
        return _fd_excess(ln_u)
    raise ValueError(f"statistics {kind!r}: one of {STATISTICS}")


def ln_inverse(ln_u, kind: str, xp=jnp):
    """`η = F⁻¹(u)` at `ln u`, in NumPy (`xp=np`) or JAX (differentiable): a band's reduced Fermi
    level at which it holds `u·N_c` (the `sfb` work functions)."""
    if kind == "boltzmann":
        return ln_u
    if kind == "blakemore":
        return ln_u - xp.log1p(-BLAKEMORE_GAMMA * xp.exp(ln_u))
    if xp is np:
        return np.asarray(jax.device_get(_fd_inverse(jnp.asarray(ln_u))))
    return _fd_excess(ln_u) + ln_u


def inverse(u, kind: str):
    """`η = F⁻¹(u)` in NumPy (a device's contacts and `sfb` work functions)."""
    u = np.asarray(u, dtype=float)
    if kind == "boltzmann":
        return np.log(u)
    if kind == "blakemore":
        if np.any(BLAKEMORE_GAMMA * u >= 1):
            raise ValueError(f"a density at or above Blakemore's bound N_c/{BLAKEMORE_GAMMA}")
        return np.log(u) - np.log1p(-BLAKEMORE_GAMMA * u)
    return np.asarray(jax.device_get(_fd_inverse(jnp.log(jnp.asarray(u)))))
