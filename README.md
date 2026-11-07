# jaxdd

A one-dimensional drift-diffusion solver for perovskite solar cells, written in JAX to solve
hundreds of thousands of devices at once on a data-centre GPU (A100/H100). It follows the device
model of [SIMsalabim](https://github.com/kostergroup/SIMsalabim) and is validated against its
output. First use: the drift-diffusion tables of
[FauxPV](https://git.bugraturan.de/trn84/fauxpv) (`tools/dd_tables.py`).

**Status: planning.** Nothing is built yet; see [`PLAN.md`](PLAN.md).

## Licence

GNU LGPL version 3 ([`COPYING.LESSER`](COPYING.LESSER), supplementing the GNU GPL version 3 in
[`COPYING`](COPYING)), as SIMsalabim. What derives from SIMsalabim is listed in
[`NOTICE`](NOTICE).

## Citing

If you use jaxdd, cite SIMsalabim, whose model it implements:

> M. Koopmans, V. M. Le Corre, L. J. A. Koster, *SIMsalabim: An open-source drift-diffusion
> simulator for semiconductor devices*, J. Open Source Softw. 7, 3727 (2022),
> doi:[10.21105/joss.03727](https://doi.org/10.21105/joss.03727)
