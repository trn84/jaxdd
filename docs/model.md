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
The points per layer are fixed by the thicknesses of the input; within them the spacing is
smooth in the thicknesses, so a device built in JAX (`batch.device_tree`) is differentiable in
them ([solver.md](solver.md#derivatives)).

A layer's share of the points, `FLOOR((NP − n_thin·5)·L/(Ltot − L_thin))`, is taken in exact
arithmetic from the input's decimal values (`device.layer_points`). Where it is a whole number,
SimSS's own rounding can give one point less; `build_device(setup, grid_split="simss")`
reproduces it ([differences.md](differences.md) §5).

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

## The field-dependent mobility (`Calc_Elec_Mob`, `Calc_Hole_Mob`)

In a layer with `mobnDep = 1` (`mobpDep` for holes) each interval's mobility follows its own
field, Poole–Frenkel's form:

```
μ_n(i+½) = μ_n · exp( γ_n · √|F| ),   F = (V[i+1] − V[i]) / (L·h[i])
```

with `gamma_n` (`gamma_p`) in (m/V)^½ and `μ_n` the zero-field mobility. The intervals across
an interface keep their transfer mobility `L·h·ν_int/V_T`. The mobility then depends on the
potential: Newton takes its derivative into the Jacobian ([solver.md](solver.md)), and the
current is integrated with each interval's mobility in the converged field. With Langevin
recombination in the same layer the rate follows the field as well (below).

## Recombination and generation

```
R = k_direct·(n·p − n_i²)
  + Σ_levels N_t·C_n·C_p·(n·p − n_t0·p_t0) / ( C_n·(n + n_t0) + C_p·(p + p_t0) )
```

the second term (SRH through the trap levels `E_t` of density `N_t`) only in layers with traps,
`n_t0 = N_c·exp((E_c − E_t)/V_T)`, `p_t0 = N_c·exp((E_t − E_v)/V_T)`. Neither acts on the
electrodes.

In a layer with `useLangevin = 1` the direct rate is Langevin's instead of `k_direct`
(`Calc_Langevin_Factor`):

```
k_direct = preLangevin · q · (μ_n + μ_p) / ε
```

with the layer's mobilities and permittivity at each point. SimSS takes the mobilities of the
interval after the point, which at the last point of each layer but the last is the interface's
transfer mobility `L·h·ν_int/V_T`; `build_device(..., langevin="simss")` does the same
([differences.md](differences.md) §8). With a field-dependent mobility in the layer the rate
takes each mobility in its interval's field: the interval after the point (SimSS's), or, at
the last point of a layer, the one before it, the layer's own (the default). The rate then
depends on the potential at the point and at a neighbour, and so does the recombination in
Newton's Jacobian.

### Bulk traps (`Init_Trapping`, `Calc_Trap_Filling_Charge`)

A layer has one level (`N_t_bulk` m⁻³ at `E_t_bulk`) or, with `bulkTrapFile`, the levels of that
file: a header (`E Ntrap`), then each level's energy (eV) and density (m⁻³), absolute (the file
overrides `E_t_bulk` and `N_t_bulk`). Every level lies strictly inside its layer's gap, and no
level of a file has density 0; SIMsalabim refuses either, and so does jaxdd. `intTrapFile` does
the same for a layer's interface to the next (densities in m⁻²), each level as in the section
below, their captures and charges summed.

`bulkTrapType` makes the layer's levels neutral (0), donors (1, positive when empty) or
acceptors (−1, negative when full). A charged level carries `(c − f)·N_t` into Poisson's
equation, `c` = 1 for a donor and 0 for an acceptor, with its occupancy by electrons

```
f = (C_n·n + C_p·p_t0) / (C_n·(n + n_t0) + C_p·(p + p_t0))
```

at the point's own densities; the SRH rate above is unchanged. The charge couples only a point's
own unknowns, so the Jacobian keeps its blocks.

### Interface traps (`Calc_Trap_Filling_Charge`, `Calc_Recombination_n/p`)

A layer's `N_t_int` (m⁻²) is a trap level at `E_t_int` on its interface to the next layer: one
level, neutral, donor or acceptor (`intTrapType` 0, 1, −1). The interface lies between the last
point of the layer, `i`, and the first of the next, `i + 1`; its traps are spread over a width
`w = ½·L·(½h[i+1] + h[i] + ½h[i−1])`. Their occupancy takes both points' densities, each side with
the `n_t0`, `p_t0` of its own layer:

```
f = (C_n·(n_i + n_{i+1}) + C_p·(p_t0,i + p_t0,i+1))
  / (C_n·(n_i + n_{i+1} + n_t0,i + n_t0,i+1) + C_p·(p_i + p_{i+1} + p_t0,i + p_t0,i+1))
```

and each point captures from its own densities, so the two carriers' rates differ point by point
(their sums over the interface agree):

```
R_n,k = N_t·C_n·(n_k·(1 − f) − n_t0,k·f) / w        k = i, i + 1
R_p,k = N_t·C_p·(p_k·f − p_t0,k·(1 − f)) / w
```

`R_n` enters the electrons' continuity equation, `R_p` the holes'. A charged level carries
`(c − f)·½·N_t / w` at each of the two points into Poisson's equation, `c` = 1 for a donor (empty
it is positive) and 0 for an acceptor. Both couple the two points of an interface, so the
Jacobian stays block tridiagonal. The level must lie inside both layers' gaps.

The generation profile (`Init_Generation_Profile`) comes in one of three ways, and the device's
`G_frac` scales it; `G_frac = 0` is the dark. It is zero on the electrodes.

- **From a file** (`genProfile` names it): its x-axis is stretched to the device, it is
  interpolated linearly onto the grid, and it is zero in layers with `layerGen = 0`.
- **Uniform** (`genProfile = none`): each layer's `G_ehp`, whatever its `layerGen` (SIMsalabim's).
- **By the transfer-matrix method** (`genProfile = calc`, `optics.py`, SIMsalabim's
  `TransferMatrix`): light from a substrate (`nkSubstrate`, its front reflection counted
  incoherently) through a transparent electrode (`L_TCO`, `nkTCO`; none for `L_TCO = 0`), the
  layers (`nkLayer` each) and a back electrode (`L_BE`, `nkBE`). On wavelengths from `lambda_min`
  to `lambda_max` in steps of 1 nm, the optical field at each grid point follows from 2×2
  interface and layer matrices (Pettersson et al., J. Appl. Phys. 86, 487 (1999)), and
  `G(x) = Σ_λ Δλ·α·n·AM(λ)·|E(x, λ)|²·λ/(h·c)` with `α = 4π·k/λ` and the spectrum `AM`
  (`spectrum`); zero in layers with `layerGen = 0`. In JAX (`batch.device_tree`) it is
  differentiable in the layers' thicknesses, which move the optics as well as the grid.

## Contacts and the applied voltage

The work functions set the boundary values of the potential and the equilibrium densities
there; the applied voltage is split evenly between the two ends:

```
V[0]    = (W_R − W_L)/2 + leftElec·V_int/2
V[NP+1] = (W_L − W_R)/2 − leftElec·V_int/2
n_eq, p_eq at the contacts: from N_c and the distance of the work function to E_c, E_v
```

A work function lies within its layer's gap (SIMsalabim refuses one outside it, and so does
jaxdd), anywhere in it: away from the band edges the contact is a Schottky barrier of
`W − E_c` for electrons and `E_v − W` for holes. `W_L` or `W_R` = `sfb` (semi-flat band) is the
adjacent layer's equilibrium Fermi level from its net doping, `E_v − V_T·ln(N_c/(N_A − N_D))`
p-doped, `E_c + V_T·ln(N_c/(N_D − N_A))` n-doped, mid-gap undoped (`ObtainWorkFunction`).

With an infinite surface recombination velocity (`S_n_L`, `S_p_L`, `S_n_R`, `S_p_R` negative,
SIMsalabim's default) the densities at the contacts are `n_eq`, `p_eq`. With a finite one
(m/s, 0 blocking), the carrier's flux into the contact is `S·(n − n_eq)`:

```
J·n̂ = −q·S_n·(n − n_eq)     electrons
J·n̂ =  q·S_p·(p − p_eq)     holes             n̂ the outward normal
```

which SIMsalabim's manual applies only while the carrier is extracted there (`n > n_eq`), the
contact's density `n_eq` while it is injected (`Cont_Eq_Elec`, `Cont_Eq_Holes`). That is jaxdd's
default (`build_device(..., surface="extraction")`); `surface="robin"` applies the condition in
both directions, thermionic emission over the barrier at the velocity `S`. SimSS 5.36 has the
sign of `S` wrong at two of the four ends ([differences.md](differences.md) §7); jaxdd's
condition is the same at all four, mirror images of each other.

`leftElec` is −1 when the left electrode is the cathode and 1 when it is the anode (a p-i-n cell
in its own orientation, lit through its anode); with the anode on the left every current's sign is
turned (SIMsalabim's), so that a photocurrent is negative either way.

The solver works at the internal voltage `V_int`. A series and a shunt resistance (`R_series`,
`R_shunt`, Ω m²) are SimSS's external circuit around the solved curve, not part of the device's
equations:

```
J_ext = J + V_int / R_shunt          (no shunt for R_shunt ≤ 0)
V_ext = V_int + J_ext · R_series
```

The shunt sits at the internal voltage, as in SimSS 5.36. `solve_batch` returns `J_ext` at the
voltages it was given, which are internal; `batch.external_voltages` gives `V_ext`. Without series
resistance the two voltages are one.

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

Each carrier's `R` includes its interface-trap capture. Across an interface the current is the
one before it plus the current through the interface traps, `J[i] = J[i−1] ± ½·q·L·h[i]·(R[i] +
R[i+1])` (SIMsalabim's; the points after it keep their values). The reported `J` is `J_n + J_p` averaged over the
device, weighted by the grid spacing. In a converged steady state it is constant across the
device; SimSS's `errJ` is its spread.

## Constants

SIMsalabim's, not CODATA's: `q = 1.6022e-19 C`, `k = 1.3807e-23 J/K`, `ε₀ = 8.8542e-12 F/m`. A
difference in the fifth digit would move `V_T` and every exponential with it.
