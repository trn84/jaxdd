# The device model and its discretisation

jaxdd's equations are SIMsalabim 5.36's (`Units/DDRoutines.pas`), on SIMsalabim's grid point for
point, by default on jaxdd's graded grid (below). This page
states them as jaxdd implements them; the names in brackets are SIMsalabim's routines, for a
reader who wants to check one against the other. Units are SI, energies in eV, potentials in V.

## The grid

`NP + 2` points from the left electrode (`x[0] = 0`) to the right one (`x[NP+1]`); the `NP`
interior points carry the unknowns, the two ends are the contacts. Two grids: jaxdd's graded
one, the default, and SIMsalabim's (`grid="simss"`), SimSS's own.

**SIMsalabim's (`Make_Grid`).** Each layer gets points in
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

**The graded grid** (`build_device(setup)`, `grid="graded"`, the default). SIMsalabim's grid gives a layer
whose share is below five points those five at any `NP`, and its spacing at a layer's end
follows `grad`, not the device: on devices with a 1 nm transport layer, ions accumulated at the
interfaces or thin doped layers it converges slowly or not at all (docs/validation.md). The
graded grid places the points by the density `1/H + β/(d + d₀)` in every layer (`d` the distance
to its nearer end, `H` the total thickness over `NP`): its uniform share and about `β = 6` points
per e-fold of the distance to each end, from `d₀ = 3 pm` (`V_T/E` at the fields of accumulated
ions) on. Each layer gets its share of the density's integral, `L/H + 2β·ln(1 + L/d₀)`, so a
1 nm layer has its ends resolved as a 500 nm one has; within it the points equidistribute the
integral. The equalisation at the interfaces and the scaling by the thicknesses are SIMsalabim's,
so it is differentiable in the thicknesses as that one is, and its points per layer are fixed
by the input's thicknesses as SIMsalabim's are. Where a curve is compared with SimSS's point by
point, SIMsalabim's grid is the one to take ([differences.md](differences.md) §12).

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

### Field-dependent generation (`Calc_Dissociation`, Braun)

In a layer with `fieldDepG = 1` light makes bound pairs at a distance `r` that dissociate with
Braun's probability (C. Braun, J. Chem. Phys. 80, 4157 (1984)):

```
P(r, F) = k_d / (k_d + k_f),   k_d = 3·k/(4π·r³) · exp(−E_B/V_T) · J(b)
E_B = q/(4π·ε·r),   b = q·|F|/(8π·ε·V_T²),   J(b) = 2·I₁(√(8b))/√(8b)
```

with `k` the point's recombination strength (its `k_direct`, or Langevin's), `k_f` the pairs'
decay rate and `F` the field at the point, `(V[i+1] − V[i−1])/(x[i+1] − x[i−1])`. `P` is
averaged over the distance's distribution about the length `a` (`thermLengDist`: 1 a delta at
`a`, 2 a Gaussian, 3 an exponential, 4 r²·exponential, 5 r⁴·Gaussian), over `r` from 0.01 `a`
to 20 `a` as SIMsalabim. At the point, the generation becomes `(P0 + (1 − P0)·P)·G` and the
direct recombination `(1 − P)·k·(n·p − n_i²)`: a pair that recombines may dissociate again.
`P0` is the share of pairs that are free at once.

jaxdd averages by Gauss–Legendre in `ln r` (16 panels of 32 nodes), to 1e-14 against an
adaptive quadrature up to 1e8 V/m; `E_B/V_T` reaches 1000 at the small distances, so `P` is taken
in logarithms (`solver._dissociation`). SimSS integrates by Romberg to 1 % and cuts `J` after
`b¹¹`; `build_device(..., braun="simss")` does the same ([differences.md](differences.md) §9).
Generation and recombination then depend on the potential at the point's two neighbours.

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
  n_anion(x)  = C_a·exp(+V(x)/V_T)        with  ∫ n_anion dx  = Σ N_anion·L_layer
  n_cation(x) = C_c·exp(−V(x)/V_T)        with  ∫ n_cation dx = Σ N_cation·L_layer
  ```

  over the species' ionic region (`Init_Ionic_Region`): its layer, or a run of adjacent layers
  that each let it enter (`ionsMayEnter`), one profile across their interfaces, the sum over the
  region's layers. The integral is SimSS's trapezoidal rule over the region. A species may have
  several regions (in both transport layers, say, the absorber closed to it), each with its own
  normalisation; with one region each these are `C_a`, `C_c`, extra unknowns of the Newton system
  ([solver.md](solver.md)). SimSS builds the same
  profile as a product of Bernoulli ratios, which equals the exponential only with the exact
  `B` ([differences.md](differences.md) §3).
- **Frozen** (`preCond = 1`, `fixIons = 1`): the free profile at the pre-bias `Vpre`, then held
  fixed for the whole scan, in Poisson's equation too (SimSS with
  [`simss/fixed_ions.patch`](../simss/README.md); unpatched SimSS 5.36 lets Poisson move frozen
  ions).
- **Held from the start** (`fixIons = 1`, `preCond = 0`): where SimSS starts them
  (`Init_Pot_Dens_Ions_Traps`), each layer's own `N` at each of its points, and held there for
  the whole scan: the same as immobile ions (the patched SimSS; unpatched, Poisson moves them,
  40 % of the current's scale on `tests/data/held_thin_light`).

Two of SimSS's regions of several layers are refused: those it would solve against the input
([differences.md](differences.md) §10).

## The current (`Calc_Curr_Int`, `Calc_Curr_Diff`, `Calc_All_Currents`)

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

With `currDiffInt = 1` (`Calc_Curr_Diff`) each interval's current is instead its own
Scharfetter–Gummel flux, the one the continuity equation balances,

```
J_n[i] = −q·μ_n[i]·V_T·(n[i+1]·B((Vgn[i+1] − Vgn[i])/V_T) − n[i]·B((Vgn[i] − Vgn[i+1])/V_T)) / (L·h[i])
```

(holes with the signs turned), the same correction across interfaces and the same average. The
two forms differ by the discretisation's and the convergence's error; at SimSS's tolerances of
1e-9 the differential one agrees with SimSS to 2e-11 of the current's scale.

## Transients (ZimT's `Main_Solver` with `dti > 0`)

ZimT 5.36 solves a list of time steps `(t, Vext, G_frac)` (its `tVGFile`, read by `read_tvg`),
each from the state of the step before, by implicit Euler with `dti = 1/Δt` (Selberherr
6-4.32). In each continuity equation's control volume, the same `fac` the generation has,

```
… + fac·(G − R) − fac·dti·(n − n_prev) = 0          (likewise for holes)
```

and `dti = 0` is the steady state, so the first step (`t = 0`, which ZimT requires) and any
return to it are steady states. jaxdd adds these terms to the residual and its Jacobian
(`solver._time_terms`) and solves each step by Newton as it solves a voltage
(`jaxdd.transient`). The current takes each density's change into the integral form's
integrand, `U = R − G + dti·(n − n_prev)`, and adds the displacement current on each interval,

```
J_D[i] = ε[i]·dti·((V[i+1] − V[i]) − (V_prev[i+1] − V_prev[i])) / (L·h[i])
```

to the average; the differential form is unchanged (ZimT's `Calc_Curr_Diff` has no time term).

In time the integral form sums each point's `U` over the point's control volume,

```
J_n[i] = J_n[i−1] − q·L·½·(h[i−1] + h[i])·U_n[i]      (holes with the sign turned)
```

the volume in which its continuity equation and Poisson's hold (`batch._sum_widths`). The two
carriers' `U` differ in time by the charge's change, which Poisson's equation makes the
displacement current's change from one interval to the next, so `J_n + J_p + J_D` is the same
on every interval, and the integral form gives the current the differential form gives (to
5e-11 of the scale at 6400 points) without reading it off single fluxes. ZimT sums over the
interval after each point, `h[i]`, and counts the left contact's change in half an interval;
its current converges at first order in the grid where the spacing changes
(`solve_transient(..., zimt_current=True)`, [differences.md](differences.md) §13). In the
steady state both carriers' `U` are the same, their sums cancel in the total, and the width
does not matter: there it is SIMsalabim's.

**Traps** add no unknowns: each level's occupation is state carried from step to step. Over a
step it follows `df/dt = b − a·f` with `a = C_n·(n + n_t0) + C_p·(p + p_t0)` and
`b = C_n·n + C_p·p_t0` from the densities at the step's end (for an interface level both
points' densities and both sides' `n_t0`, `p_t0`), solved exactly from the occupation before
(ZimT's `Calc_f_tb`, `Calc_Trap_Filling_Charge`):

```
f  = b/a + (f_prev − b/a)·e^(−a·Δt)                     the charge (Poisson) takes this
f̄  = b/a + (f_prev − b/a)·(1 − e^(−a·Δt))/(a·Δt)        the capture takes this, its mean
R_n = C_n·N_t·(n − (n + n_t0)·f̄)     R_p = C_p·N_t·((p + p_t0)·f̄ − p_t0)
```

(`solver._trap_step`; ZimT's `Calc_Recombination_n/p` integrate the ODE's capture over the
step). With `dti = 0` both are `b/a` and the rates the steady state's SRH; over a step the two
carriers recombine at different rates while the traps fill or empty, so each continuity
equation and each carrier's current take their own.

**Mobile ions** become unknowns of their own: five per point, `(V, ln n, ln p, ln a, ln c)`,
5×5 blocks (`solver._assemble_ions`). In each of its regions an ion's row is ZimT's
(`Solve_Neg_Ions`, `Solve_Pos_Ions`): Scharfetter–Gummel fluxes in `V` over the region's
intervals, none through its ends or through a contact (a contact's density follows its
neighbour's by that zero flux), each interval's mobility its left point's, and implicit Euler in
the same control volume the carriers have, also at the region's end points:

```
anions:  h_{i−1}·F_r − h_i·F_l + fac·dti·(e^{l_k}·a_prev − a) = 0,   F = Vt·μ·(B(x)·a₁ − B(−x)·a₀)
```

(cations as holes). ZimT rescales the ions to their total after every solve
(`Rescale_Ion_Density`, the trapezoid rule over the region); its converged state is therefore
the discrete equation's with the density before the step scaled by a factor per region,
`e^{l_k}`, which is a slot (one unknown per region, its constraint the total; Stage B's). With
`dti = 0` the row is Boltzmann's and `l_k` its normalisation, as in the steady state. Outside
its regions an ion's unknown is held. The ions' currents (each region by the integral form with
the densities' change as its integrand, or by its fluxes with `currDiffInt = 1`, the current
across a layer interface the one before it) join the carriers' and the displacement current in
the average. The 5×5 blocks are solved by the sweep kernels as the 3×3 are (or the scan), each
block inverted by its partition (`solver._inverse5`: the carriers' 3×3 by its adjugate, the
ions' 2×2 Schur complement by its own); a batch takes 5×5 where any device has mobile ions,
the others' ion rows held.

**The internal voltage.** ZimT's `Vext` is the external voltage; given, and without a series
resistance, `Vint = Vext`. At the open circuit (`Vext = oc`, at `t = 0` too) `Vint` is the
voltage at which `Jext = Jint + Vint/R_shunt` is 0, with a series resistance the one at which
`Vint + Jext·R_series = Vext`. ZimT brackets it and runs Brent on whole solves to `tolVint`
(`Residual_Current_Voltage`, `Find_Vint_Brent`). jaxdd makes it one more unknown of the step's
Newton solve (`batch._system_vint`): its column, the rows' and the ion constraints' derivatives
in it (through the contacts' potentials), and its row (`batch._vint_row`: `Jext` over the
device's current scale, or the voltage balance in `V_T`), dense in the grid like the ions'
constraints, one more border solved by the same Schur complement. Its step is held to the
potential's cap (10 `V_T`), and the current's settling is no stop there, since the voltage's
last digits are the result. Where one device of a batch needs it every device takes the row,
the others' `Vint = Vext`. The first step at an open circuit starts from 0 V, as ZimT
brackets from it (13 Newton steps to `Voc` for `thin_l1_light`), each step after it from the
last voltage extrapolated as the state is; with a series resistance from `Vext` less the last
current's drop. `solve_transient` returns `Vint` and `Vext = Vint + Jext·R_series` per step.

**Gradients through time.** A transient is differentiable in its parameters
(`batch.device_tree`, as the steady state's for its fits): each step's derivatives are those
of its converged solution by the implicit function (`batch._converged_jvp`, with the internal
voltage's border where it is an unknown), and they reach the next step through the state it
starts from (the densities and potential before it, the traps' occupations, the ions), so
forward mode (`jax.jvp`, `jax.jacfwd`) carries the tangents from step to step: no reverse pass
through time, which would hold every state. Under tracing `solve_transient` takes `ions` and
`vint` as given and returns JAX arrays. `validation/fit_transient.py` fits a transient from
its own curve.

**Stopping and failures**, as ZimT's main loop has them, per device (`TVG` carries the setup's
`failureMode`, `convVar` and tolerances). A step not converged: `failureMode` 0 stops the run,
1 keeps the step, 2 skips it, keeping it only in the steady state (`t = 0`), but stops where the
last step kept is the reference. Each step's `Δt` is from the last step kept, so after a
skipped step the next one is longer. `Track` 1 makes the last state kept the reference
(if its step converged); 2 and 3 compare it with the reference (`States_Are_Equal`: the same
`G_frac`, the potentials within `tolPois` in the Euclidean norm over the grid, then by
`convVar` the densities each within `tolDens` relatively, the current within `tolCurr` or the
smaller `errJ`, either or both), 3 then makes it the reference; five equal states in a row
(`MinCountStatic`) stop the run after that step. The batch runs in lockstep, so a device's
steps after its stop are solved but not kept: `TransientResult.accepted` marks the steps ZimT
writes, `stop` why a device stopped. Two of ZimT's own ways are kept: its `tVGFile` forgets a
`Track = 1` on its first line (a 2 or 3 after it is refused, as ZimT refuses it, though only
when it reaches that line), and `convVar` 2–4 also loosen ZimT's own convergence (its solve
stops once the current is within `tolCurr`), which jaxdd's does not follow: its states are
converged as always. Where the states' difference crosses `tolPois` from one step to the next,
ZimT's own error, which is about `tolPois`, decides the step it stops at: jaxdd stops within a
step of it.

**Output**: the rest of ZimT's `tJFile` (`Write_To_tJV_File`), per step and device, named as
ZimT names them by `tj_columns`: `errJ` (the current's rms deviation over the grid from its
average), per layer the QFLS averaged on the grid and the photocurrent and each carrier's
direct and bulk recombination as currents, at each interface the interface traps'
recombination of each carrier, the minority carriers' currents at the contacts, the shunt's,
the ions' and the displacement current, and with `specialOutput = spv` each layer's voltage
drop. The currents are `batch._current_parts`, what the average `Jint` is made of. ZimT's
`varFile` (the profiles at each step) is not produced. The QFLS is `V_T·ln(n·p/n_i²)`. It
takes the logarithm of densities far below any that moves a current, below Newton's floor of
convergence (`n_floor`, 1e-12 of the largest `N_c`): the minority carriers in the dark and in a
transport layer. Newton steps those in the density, not its logarithm, so that they converge
with the rest although they do not count for convergence (docs/solver.md, "When is a voltage
converged?"); the QFLS is then the converged solution's at every step (docs/validation.md).

## Constants

SIMsalabim's, not CODATA's: `q = 1.6022e-19 C`, `k = 1.3807e-23 J/K`, `ε₀ = 8.8542e-12 F/m`. A
difference in the fifth digit would move `V_T` and every exponential with it.
