"""Photon recycling in an absorbing slab: where the light of radiative recombination goes.

Light from radiative recombination in a film whose index is high (a perovskite, n ≈ 2.6) leaves
it only within a narrow cone; the rest is reflected back, totally, and is absorbed again, a new
electron–hole pair (photon recycling). `slab_kernel` gives, for a slab between two media, the
share of the photons emitted at each point that is absorbed in each control volume and the
shares that leave it at either side, integrated over the directions (isotropic emission) and
the emission spectrum. `solve_transient(recycle=...)` takes the first as a generation, from
each step's state before it (docs/model.md, "Photon recycling").

The optics, per photon energy `E` and direction `μ = cos θ` in the slab (`κ = α/|μ|` along `z`):
the light runs back and forth between the slab's two faces and is absorbed on its way,
`e^{−κ·Δz}`; at each face it is reflected by Fresnel's `R(μ)` (unpolarised) into the medium
beyond, and totally where it would leave the escape cone of the medium it ends in (`n_exit`,
air): light that enters the glass at an angle the glass's back face reflects totally comes back
(laterally displaced, which a film infinite in its plane does not notice). The series of
round trips is summed in closed form. Interference is not (the light of the bands' edge is
incoherent over a film's thickness), and nothing is absorbed outside the slab.

The emission spectrum is van Roosbroeck and Shockley's, `∝ α(E)·E²·exp(−E/kT)`, from the
absorption `α(E)` (`urbach_alpha`, or any given on a grid of energies).
"""

from __future__ import annotations

import numpy as np

from jaxdd.constants import K_B, Q


def urbach_alpha(E, E_g=1.52, E_U=0.013, alpha_g=2.3e6):
    """An absorption edge (1/m at photon energies `E`, eV): Urbach's tail `α_g·e^{(E−E_g)/E_U}`
    below the gap `E_g`, `α_g·(1 + √((E − E_g)/E_U))` above it. The defaults: FAPbI₃ near its
    edge, 5e4 cm⁻¹ at 806 nm (Kober-Czerny et al.'s ellipsometry, 4.4–6.2e4)."""
    E = np.asarray(E, dtype=float)
    above = alpha_g * (1.0 + np.sqrt(np.maximum(E - E_g, 0.0) / E_U))
    return np.where(E - E_g < 0, alpha_g * np.exp((E - E_g) / E_U), above)


def vrs_weights(E, alpha, T=295.0):
    """Van Roosbroeck–Shockley: the share of the photons emitted at each energy of the grid `E`
    (eV), `∝ α(E)·E²·exp(−E/kT)·ΔE`, summed to 1."""
    E = np.asarray(E, dtype=float)
    kT = K_B * T / Q
    w = np.asarray(alpha) * E**2 * np.exp(-(E - E.min()) / kT) * np.gradient(E)
    return w / w.sum()


def fresnel(mu, n1, n2):
    """Unpolarised reflectance from a medium of index `n1` into one of `n2` at `cos θ = mu` in the
    first; 1 beyond the critical angle."""
    mu = np.asarray(mu, dtype=float)
    s2 = n1 / n2 * np.sqrt(np.maximum(1.0 - mu**2, 0.0))
    total = s2 >= 1.0
    c2 = np.sqrt(np.maximum(1.0 - s2**2, 0.0))
    rs = (n1 * mu - n2 * c2) / (n1 * mu + n2 * c2)
    rp = (n2 * mu - n1 * c2) / (n2 * mu + n1 * c2)
    return np.where(total, 1.0, 0.5 * (rs**2 + rp**2))


def face(mu, n_slab, n_out, n_exit):
    """A face's reflectance `R(μ)` and whether light through it leaves for good: Fresnel's into
    `n_out`, and 1 where it would not leave the escape cone of `n_exit` (it comes back)."""
    escapes = n_slab * np.sqrt(np.maximum(1.0 - np.asarray(mu) ** 2, 0.0)) < n_exit
    return np.where(escapes, fresnel(mu, n_slab, n_out), 1.0)


def _one_way(z_from, a, b, d, kappa, R_near, R_far):
    """Light that starts at `z_from` towards the face at `d` (cells `[a, b]` along the same
    axis): `(absorbed per cell, arriving at the near face, arriving at the far face)`, every
    round trip summed. `kappa`, `R_near` (the face at `d`), `R_far` (the face at 0) broadcast
    over the leading axes (energies × directions)."""
    k = kappa[..., None]
    up_a, up_b = np.maximum(a, z_from), np.maximum(b, z_from)
    direct = np.exp(-k * (up_a - z_from)) - np.exp(-k * (up_b - z_from))
    T1 = np.exp(-kappa * (d - z_from))  # arriving at the near face first
    full = np.exp(-kappa * d)
    S = 1.0 / (1.0 - R_near * R_far * full**2)
    down = R_near * T1 * S  # leaving the near face downwards, every round trip
    up = R_far * down * full  # leaving the far face upwards
    absorbed = (direct + down[..., None] * (np.exp(-k * (d - b)) - np.exp(-k * (d - a)))
                + up[..., None] * (np.exp(-k * a) - np.exp(-k * b)))  # fmt: skip
    return absorbed, T1 + up * full, down * full


def slab_kernel(z, edges, thickness, E=None, alpha=None, n_slab=2.6, n_top=1.0, n_bottom=1.5,
                n_exit=1.0, T=295.0, n_mu=96):  # fmt: skip
    """The photons emitted at the points `z` of a slab `[0, thickness]` (m, from its bottom) and
    where they go, over the cells `edges` (cell `i` from `edges[i, 0]` to `edges[i, 1]`):
    `(P, top, bottom)`, `P[i, j]` the share of point `j`'s photons absorbed in cell `i`, `top`
    and `bottom` the shares that leave through either face for good (into the escape cone of
    `n_exit`). `E` (eV) and `alpha` (1/m): the absorption on a grid of energies, by default
    `urbach_alpha` from 1.35 to 1.70 eV; the spectrum `vrs_weights` at `T`. The media beyond:
    `n_top` above, `n_bottom` below (glass), light leaving for good only within the cone of
    `n_exit` (air). `P`'s columns, `top` and `bottom` sum to 1 (to the quadrature's error)."""
    z = np.asarray(z, dtype=float)
    edges = np.clip(np.asarray(edges, dtype=float), 0.0, thickness)
    if E is None:
        E = np.linspace(1.35, 1.70, 141)
        alpha = urbach_alpha(E)
    E, alpha = np.asarray(E, dtype=float), np.maximum(np.asarray(alpha, dtype=float), 1e-3)
    w_E = vrs_weights(E, alpha, T)
    x, w_x = np.polynomial.legendre.leggauss(n_mu)
    mu, w_mu = 0.5 * (x + 1.0), 0.5 * w_x  # (0, 1]
    R_t = face(mu, n_slab, n_top, n_exit)[None, :]
    R_b = face(mu, n_slab, n_bottom, n_exit)[None, :]
    kappa = alpha[:, None] / mu[None, :]  # energies × directions
    weight = (w_E[:, None] * w_mu[None, :])[..., None]  # half of the directions each way
    a, b = edges[:, 0], edges[:, 1]
    P = np.zeros((len(a), len(z)))
    top, bottom = np.zeros(len(z)), np.zeros(len(z))
    for j, zj in enumerate(z):
        up, at_top, at_bottom = _one_way(zj, a, b, thickness, kappa, R_t, R_b)
        # downwards: the same in the mirrored axis
        dn, at_bottom2, at_top2 = _one_way(thickness - zj, thickness - b, thickness - a,
                                           thickness, kappa, R_b, R_t)  # fmt: skip
        P[:, j] = 0.5 * np.sum(weight * (up + dn), axis=(0, 1))
        top[j] = 0.5 * np.sum(weight[..., 0] * (1 - R_t) * (at_top + at_top2))
        bottom[j] = 0.5 * np.sum(weight[..., 0] * (1 - R_b) * (at_bottom + at_bottom2))
    return P, top, bottom


def device_kernel(dev, **optics):
    """`slab_kernel` for a device's absorber, its points where `k_direct > 0` (one layer, its
    cells from the midpoints between the points, clipped to the layer): the matrix over the
    device's interior points that `solve_transient(recycle=...)` takes, its rows and columns 0
    outside the absorber, and the shares that leave at the device's left and right."""
    x = np.asarray(dev.x)
    inner = x[1:-1]
    on = np.asarray(dev.k_direct)[1:-1] > 0
    idx = np.flatnonzero(on)
    if idx.size == 0:
        raise ValueError("no point with k_direct > 0: nothing emits")
    if np.any(np.diff(idx) != 1):
        raise ValueError("the points with k_direct > 0 are not one layer")
    full = idx + 1  # in the full grid
    lo = 0.5 * (x[full[0] - 1] + x[full[0]]) if full[0] > 1 else x[0]
    hi = 0.5 * (x[full[-1]] + x[full[-1] + 1]) if full[-1] < len(x) - 2 else x[-1]
    mids = 0.5 * (x[full[:-1]] + x[full[1:]])
    edges = np.stack([np.r_[lo, mids], np.r_[mids, hi]], -1) - lo
    P, top, bottom = slab_kernel(inner[idx] - lo, edges, hi - lo, **optics)
    n = len(inner)
    K = np.zeros((n, n))
    K[np.ix_(idx, idx)] = P
    return K, bottom, top  # the device's left is the slab's bottom
