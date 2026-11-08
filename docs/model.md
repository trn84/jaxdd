# The device model and its discretisation

jaxdd's equations are SIMsalabim 5.36's, point for point (`Units/DDRoutines.pas`). This page
states them as jaxdd implements them; the names in brackets are SIMsalabim's routines, for a
reader who wants to check one against the other. Units are SI, energies in eV, potentials in V.

## The grid (`Make_Grid`)

`NP + 2` points from the left electrode (`x[0] = 0`) to the right one (`x[NP+1]`); the `NP`
interior points carry the unknowns, the two ends are the contacts. Each layer gets points in
proportion to its thickness, at least five (`MinGridPointsPerLayer`). Within a layer the spacing
grows exponentially from both ends towards the middle, with the setup's `grad`; afterwards the
spacing of the three intervals around each interface is made equal. The spacing is kept relative
to the total length: `h[i] = (x[i+1] − x[i]) / L`.

`L` is the sum of the layer thicknesses. The equalisation at the interfaces leaves `Σh`
slightly off 1 (0.006 % on a 1/500/1 nm stack); SIMsalabim scales everything by `L` regardless,
and so does jaxdd. `device.make_grid` builds it; `build_device` then fills every array on it.

## Bands and generalised potentials

Per point: the conduction and valence band edges `E_c`, `E_v`, the effective density of states
`N_c` (SIMsalabim uses the same value for both bands), the permittivity `ε`, and
`n_i = N_c·exp(−(E_v − E_c)/(2 V_T))` with `V_T = kT/q`.

Band offsets and changes in the density of states enter the carrier transport through the
**generalised potentials**:

```
Vgn = V + (E_c − E_c[0]) + V_T·ln(N_c / N_c[0])
Vgp = V + (E_v − E_v[0]) − V_T·ln(N_c / N_c[0])
```

so that an electron sees `Vgn` as it would see `V` in a homogeneous device.

## Poisson's equation (`Solve_Poisson`)

At every interior point `i`, with SIMsalabim's three-point coefficients for a permittivity that
jumps between layers:

```
fac_l·V[i−1] + fac_m·V[i] + fac_u·V[i+1] = q·(n − p + N_A − N_D + n_anion − n_cation)
```

## Continuity: Scharfetter–Gummel (`Cont_Eq_Elec`, `Cont_Eq_Holes`)

In steady state `∇·J_n = q(R − G)` and likewise for holes. The flux between two neighbouring
points is Scharfetter–Gummel's, written with the Bernoulli function `B(x) = x/(eˣ − 1)` of the
difference of the generalised potentials:

```
J_n(i+½) ∝ μ_n(i+½)·V_T/h · [ B(ΔVgn/V_T)·n[i+1] − B(−ΔVgn/V_T)·n[i] ]
```

Summed over the two intervals of a point, this gives a three-point equation in `n[i−1]`, `n[i]`,
`n[i+1]` with the net generation `G − R` integrated over the point's share of the grid. The
mobilities live on the intervals (`μ[i]` between `x[i]` and `x[i+1]`).

`B` is 0/0 at zero. jaxdd evaluates it exactly, with a series below |x| = 1e-2, and in a form
whose derivative is finite everywhere (the "double where": the full expression is evaluated at a
safe argument where the series is used, so automatic differentiation never sees 0/0). SimSS uses
a coarser Taylor series up to |x| = 1.5; `bernoulli_name="simss"` reproduces it
([differences.md](differences.md) §2).

## Interfaces (`Calc_Elec_Mob`)

At an interface the interval between the last point of one layer and the first of the next gets
the mobility `L·h·ν_int / V_T`, with the interface velocity `nu_int_n` or `nu_int_p` of the layer
on the left. The Scharfetter–Gummel flux across that interval is then thermionic emission at
velocity `ν_int` over the band offset. A carrier facing a large offset (say 1 eV) is
effectively blocked.

## Recombination and generation

```
R = k_direct·(n·p − n_i²)
  + N_t·C_n·C_p·(n·p − n_t0·p_t0) / ( C_n·(n + n_t0) + C_p·(p + p_t0) )
```

the second term (SRH through one neutral trap level `E_t` of density `N_t`) only in layers
with traps, `n_t0 = N_c·exp((E_c − E_t)/V_T)`, `p_t0 = N_c·exp((E_t − E_v)/V_T)`. Neither acts on
the electrodes.

The generation profile is read from the file the setup names (`Init_Generation_Profile`): its
x-axis is stretched to the device, it is interpolated linearly onto the grid, and it is zero in
layers with `layerGen = 0`. The device's `G_frac` scales it; `G_frac = 0` is the dark.

## Contacts and the applied voltage

The contacts are ohmic. The work functions set the boundary values of the potential and the
densities there; the applied voltage is split evenly between the two ends:

```
V[0]    = (W_R − W_L)/2 − V_int/2
V[NP+1] = (W_L − W_R)/2 + V_int/2
n, p at the contacts: from N_c and the distance of the work function to E_c, E_v
```

Without series resistance, `V_int` is the applied voltage.

## Ions (`Init_Ionic_Region`, `Calc_Ion_Distribution_Steady_State`)

Anions and cations each have a density `N` and a mobility in a layer. Three cases:

- **Immobile** (`mu = 0`): a fixed charge `N` in the layer.
- **Free** (mobile, steady state): ions carry no current, so each species is distributed by
  Boltzmann in the potential, normalised to the number of ions in its layer:

  ```
  n_anion(x)  = C_a·exp(+V(x)/V_T)        with  ∫ n_anion dx  = N_anion·L_layer
  n_cation(x) = C_c·exp(−V(x)/V_T)        with  ∫ n_cation dx = N_cation·L_layer
  ```

  The integral is SimSS's trapezoidal rule over the layer. The normalisations `C_a`, `C_c` are
  two extra unknowns of the Newton system ([solver.md](solver.md)). SimSS builds the same
  profile as a product of Bernoulli ratios, which equals the exponential only with the exact
  `B` ([differences.md](differences.md) §3).
- **Frozen** (`preCond = 1`, `fixIons = 1`): the free profile at the pre-bias `Vpre`, then held
  fixed for the whole scan, in Poisson's equation too (SimSS with
  [`simss/fixed_ions.patch`](../simss/README.md); unpatched SimSS 5.36 lets Poisson move frozen
  ions).

A mobile region spans one layer; an ionic region across layers is refused.

## The current (`Calc_Curr_Int`, `Calc_All_Currents`)

The terminal current is not read off a single Scharfetter–Gummel flux. SIMsalabim's integral
form (De Mari's, `currDiffInt = 2`) integrates each carrier's continuity equation across the
device, which makes the current insensitive to local discretisation error:

```
J_n(x) = q·K_n + q·∫₀ˣ (R − G) dx'       (likewise for holes, with the opposite sign)
K_n from the contact densities and ∫ exp(−Vgn/V_T)/μ_n dx
```

Across an interface the current is taken from the point before it (SIMsalabim's handling of
interface traps, of which Stage A has none). The reported `J` is `J_n + J_p` averaged over the
device, weighted by the grid spacing. In a converged steady state it is constant across the
device; SimSS's `errJ` is its spread.

## Constants

SIMsalabim's, not CODATA's: `q = 1.6022e-19 C`, `k = 1.3807e-23 J/K`, `ε₀ = 8.8542e-12 F/m`. A
difference in the fifth digit would move `V_T` and every exponential with it.
