# The device model and its discretisation

jaxdd's equations are SIMsalabim 5.36's (`Units/DDRoutines.pas`), on SIMsalabim's grid point for
point, by default on jaxdd's graded grid (below), as jaxdd implements them; the names in brackets
are SIMsalabim's routines. Units are SI, energies in eV, potentials in V.

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

With another statistics than Boltzmann's (`statistics=`, below) the graded grid's `d₀` at the
two contacts is the contact layer's Debye length at its `N_c`, `√(ε·V_T/(q·N_c))`, not 3 pm:
there an accumulation is bounded by the band's density of states and varies on no shorter a
length, and finer intervals only multiply the round-off at a blocking contact
([solver.md](solver.md), "A floor above 1e-6"). The interfaces keep 3 pm (the ions' fields).

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

## The carriers' statistics (jaxdd's, Boltzmann's by default)

SIMsalabim's carriers follow Boltzmann's statistics, `n = N_c·exp(η)` with the reduced
quasi-Fermi level `η = (E_Fn − E_c)/kT` (holes likewise), which puts no bound on the density:
an accumulation `e^(V/V_T)` above equilibrium passes `N_c` a few tenths of a volt from it
(docs/solver.md, "A floor above 1e-6"). `build_device(..., statistics=)` takes another
(`jaxdd.statistics`), `n = N_c·F(η)`:

| `statistics` | `F(η)` | |
|---|---|---|
| `"boltzmann"` (default) | `exp(η)` | SIMsalabim's, to the bit |
| `"blakemore"` | `1/(exp(−η) + 0.27)` | Blakemore 1982: within a few percent of `F_{1/2}` to `η` ≈ 1.3, bounded by `N_c/0.27` |
| `"fermi-dirac"` | `F_{1/2}(η)`, TeSCA's approximation | within 1 % of the integral (as ChargeTransport.jl's `FermiDiracOneHalfTeSCA`) |

The unknowns stay `ln n` and `ln p`. What changes is the electrochemical potential: `η = ln(n/N_c)
+ w` with the excess chemical potential `w = F⁻¹(n/N_c) − ln(n/N_c)` (`statistics.excess`; 0 for
Boltzmann's, `−ln(1 − 0.27·n/N_c)` for Blakemore's, Newton on TeSCA's `F` for `F_{1/2}`, with its
own derivative `1/(d ln F/dη) − 1`). Everywhere a density meets its quasi-Fermi level:

- **The flux** at a constant mobility follows `η`: the Scharfetter–Gummel flux with `w`'s
  difference over the interval beside the potential's, `x − Δw_n` for electrons and `x + Δw_p`
  for holes (as the ions bounded by their sites, below); its derivatives in `ln n` gain
  `∓(∂F/∂x)·dw/d ln n` (`_assemble`, written out and checked against autodiff,
  `tests/test_statistics.py`). Across an interface the same, each side's `w` at its own `N_c`.
- **The contacts**: the equilibrium densities `N_c·F(η)` at the work function, and `sfb` the level
  at which the majority band holds the net doping, `E_c − V_T·F⁻¹(N_D/N_c)` (`E_v +
  V_T·F⁻¹(N_A/N_c)`). With a finite `S` the condition's `x` holds the contact's own `w`: no longer
  linear in its density, it is solved by Newton on `ln n` from Boltzmann's density
  (`CONTACT_NEWTON_STEPS`, quadratic).
- **The recombination** compares with the equilibrium at the point's own quasi-Fermi levels:
  `n·p − n_i²·e^(−w_n − w_p)` (`= n·p·(1 − e^(−QFLS/kT))`, ChargeTransport.jl's form), and each
  trap level's `n_t0·e^(−w_n)`, `p_t0·e^(−w_p)` in SRH, its occupancy and its charge (`_degenerate`),
  so that every rate vanishes at a flat Fermi level.
- **The current** (both forms) takes the generalised potentials with `∓V_T·w`, and the QFLS
  `V_T·(ln(n·p/n_i²) + w_n + w_p)`.
- **In time** (`solve_transient`) nothing else: implicit Euler's `n − n_prev` is in the densities
  themselves, the traps' occupations over a step (`_trap_step`) take the degenerate `n_t0`,
  `p_t0` as the steady state does, the recycled photons the degenerate `n_i²`, and ZimT's QFLS
  column the excess potentials.
- **Interface traps** compare each point's capture with its own equilibrium, `n_t0·e^(−w_n)`
  and `p_t0·e^(−w_p)` at the edge's left and right point (`_interface_degenerate`), and **a free
  surface's** mid-gap level likewise `n_i·e^(−w_n)`, `n_i·e^(−w_p)` against
  `n·p − n_i²·e^(−w_n − w_p)` (`_surface_srh`). So in the dark at 0 V a degenerate layer's
  interface traps and surfaces recombine nothing (`tests/test_statistics.py`; with Boltzmann's
  `n_t0` there the holes' quasi-Fermi level would bend by 1e-4 V and more).

Another statistics costs compile time. Each evaluation of `F_{1/2}⁻¹` is five Newton steps on
TeSCA's `F`, so the assembly takes the excess potentials once per state (`stat_cache`,
`_cached`), and a contact's Newton on its density (`_robin`) is a loop with its body traced once,
its derivatives the iteration's (through the converged density alone, the implicit function,
Newton on the device diverges where the contact has not converged in its 12 steps). One assembly
of the degenerate HTL's stack then compiles in 2.2 s against Boltzmann's 0.8 s, a whole steady
state in 74 s against 19 s. Each evaluation's own Newton steps stay unrolled: as a loop they
compile in 40 % less time but run 15 % slower.

The implied `V_oc` and the PLQY keep their Boltzmann's definitions (`k·n_i²` the radiative
equilibrium). Against ChargeTransport.jl with degenerate transport layers, in the steady state and in time:
[validation.md](validation.md).

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
on the left. The Scharfetter–Gummel flux across that interval is then, per area,
`ν_int·[B(x)·n₁ − B(−x)·n₀]`, with `x` the jump of the generalised potential across it over `V_T`
(the band offset, the potential's drop over the interval and the `N_c` ratio). That is
thermionic emission over the offset, `v·(n_h − n_l·e^(−|x|))` from the side of the higher band
(density `n_h`) to that of the lower (`n_l`), at the effective velocity
`v = ν_int·B(−|x|) = ν_int·|x|/(1 − e^(−|x|))`: `ν_int` for a small offset, about `ν_int·ΔE/kT`
for a large one (39 `ν_int` at 1 eV and 300 K). The flux vanishes at equal quasi-Fermi levels
whatever `ν_int` is. A carrier facing a large offset (say 1 eV) is effectively blocked.

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

### Auger recombination (jaxdd's, off by default)

SIMsalabim has none. jaxdd adds, in a layer whose file names them, the Auger coefficients
`C_n_Auger` and `C_p_Auger` (m⁶/s; 0 where not named):

```
R_Auger = (C_n_Auger·n + C_p_Auger·p)·(n·p − n_i²)
```

the electrons' process (two electrons and a hole, eeh) and the holes' (ehh), each balanced by its
impact ionisation, so that it vanishes at a flat Fermi level; with another statistics `n_i²` is
the equilibrium's at the point's own quasi-Fermi levels, as for the direct rate. It is
non-radiative: the PLQY and implied `V_oc` count `k_direct`'s photons only, and Auger takes
from them through the densities. It adds to both carriers' rates, in the steady state and in
time, and Newton's Jacobian takes its derivatives as the other rates' (`jax.jvp` of
`_bulk_rates`); a transient's columns report it as `JAuger` per layer (jaxdd's), apart from the
bulk's. A device with no coefficient above 0 carries no Auger term at all (`Device.auger_n`
None), so SIMsalabim's inputs solve to the bit as before. Tested on a uniform film: the steady
state and each step of implicit Euler in closed form, to 1e-9 (`tests/test_free_surface.py`).

### Bulk traps (`Init_Trapping`, `Calc_Trap_Filling_Charge`)

A layer has one level (`N_t_bulk` m⁻³ at `E_t_bulk`) or, with `bulkTrapFile`, the levels of that
file: a header (`E Ntrap`), then each level's energy (eV) and density (m⁻³), absolute (the file
overrides `E_t_bulk` and `N_t_bulk`). Every level lies strictly inside its layer's gap, and no
level of a file has density 0; SIMsalabim refuses either, and so does jaxdd. `intTrapFile` does
the same for a layer's interface to the next (densities in m⁻²), each level as in the section
below, their captures and charges summed.

**Each level's own capture coefficients** (jaxdd's, not SIMsalabim's). In SIMsalabim a file's
levels share their layer's `C_n_bulk`, `C_p_bulk` (`C_n_int`, `C_p_int`). Then a shallow electron
trap, which takes electrons and gives them back but hardly holes, cannot sit beside a deep
recombination level: with the hole capture of the deep one it recombines itself. A trap file
whose header reads `E Ntrap C_n C_p` gives each level its own (m³/s, ≥ 0); a level with
`C_p` = 0 traps electrons and does not recombine. The other layers' levels keep their
layer's. A file without the columns is SIMsalabim's, and its device is the same to the bit; SimSS
cannot read one with them. The parameters (`device_parameters`) then hold `bulk_trap_Cn`,
`bulk_trap_Cp` (`int_trap_Cn`, `int_trap_Cp`), layers × levels, differentiable as the rest.

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

  jaxdd's own, 0 by default: a layer's `dE_nk` (eV; in its file or the parameters' dict,
  `device.OPTICS_PARAMETERS`) shifts its n,k data in photon energy, positive towards higher
  energy (a wider gap): `n(λ) = n_data(λ')` with `λ' = λ/(1 − dE_nk·λ/(h·c/q))`, the wavelength
  of the photon energy `h·c/λ − dE_nk`. The substrate, the TCO and the back electrode keep
  theirs. The layer's table is kept as read and interpolated at `λ'` as the data are on the
  wavelengths: linearly, and beyond either end of the data along its first or last interval. A
  shift reads the data from `λ'` of `lambda_min` to `λ'` of `lambda_max` (827 nm for 800 nm and
  0.05 eV; below `lambda_min` for a negative one), which they should cover: beyond either end
  the line through the end points can make `n` or `k` unphysical (`k` negative). `dE_nk` must
  stay below the photon energy of `lambda_max`. A zero shift gives `λ' = λ` exactly and the
  profile without a shift, SIMsalabim's, to the bit, and so are its derivatives in the
  thicknesses (to round-off where the shifts are traced under `jax.jit`). In JAX the
  profile is differentiable in the shifts, so that a band gap that moves can move the
  absorption edge with it within a batch; where a `λ'` crosses a data point the derivative
  jumps, and on a data point it is the one for an increasing shift: with data on the 1 nm
  wavelengths, as SIMsalabim's are, that is every `λ'` at a zero shift. The photon-recycling
  kernel (`jaxdd.recycling`) does not follow the shift: its `α(E)` is the caller's.

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

### Photon recycling (jaxdd's, in time)

SIMsalabim counts direct recombination as lost. In a film whose index is high (a perovskite,
`n ≈ 2.6`) most of its light does not leave: only within a narrow cone (23° into air), the rest
is reflected back, totally, and absorbed again as a new electron–hole pair. In 600 nm of FAPbI₃
about 97 % of the photons of its band edge are reabsorbed, so recycling slows the decay of the
carriers and spreads them through the film.

`jaxdd.recycling.slab_kernel` gives, for a slab between two media, the share `P[i, j]` of the
photons emitted at point `j` that is absorbed in the control volume of point `i`, and the
shares that leave at either face. Per photon energy `E` and direction `μ = cos θ` (isotropic
emission, Gauss–Legendre in `μ`) the light is absorbed as `e^{−α(E)·Δz/|μ|}` on its way and
reflected at each face by Fresnel's `R(μ)` (unpolarised) into the medium beyond, totally
beyond the escape cone of the medium it ends in (`n_exit`, air): light entering the glass at
an angle that the glass's back face reflects totally comes back. The round trips between the
faces are summed in closed form. The spectrum is van Roosbroeck and Shockley's,
`∝ α(E)·E²·e^{−E/kT}`, from `α(E)` (`urbach_alpha`, an Urbach tail below the gap, or any on a
grid of energies). Interference is neglected (the band edge's light is incoherent over a
film's thickness), and nothing absorbs outside the slab. The shares sum to 1 per emitting point
to the quadrature's error (1e-15). `device_kernel(dev)` gives the matrix over a device's
interior points, its absorber the points with `k_direct > 0`.

`solve_transient(batch, tvg, recycle=K)` adds to each time step's generation the photons the
state before the step emits and the matrix reabsorbs,

```
G_rec[i] = Σ_j K[i, j] · k_direct[j]·(n_j·p_j − n_i,j²) · vol_j / vol_i,
```

explicit in time (first order, like implicit Euler's steps; a step much shorter than the
radiative lifetime). A steady state (`t = 0`, a return to it) has none: in the dark
`n·p = n_i²` emits nothing net, and under light the recycled generation would need its own
solution. Without `recycle` a transient is the same to the bit.

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

## Free surfaces (jaxdd's, `W_L` or `W_R` = `free`)

SIMsalabim fixes the potential at both ends: every device lies between two electrodes. A film
measured by photoluminescence often has one or none: an absorber on glass, glass/absorber/transport
layer, a contact with a half stack on it. jaxdd lets either end, or both, be a **free surface**:
`W_L = free` or `W_R = free` in the setup. SimSS refuses such an input, so it is no reference here
(tests/test_free_surface.py: analytic cases and the device's invariances).

At a free end nothing flows: no electrons, no holes, no displacement field beyond it (the film is
neutral as a whole, and outside it there is no charge). In the discretisation the end's interval
carries no transport, its mobilities 0 and Poisson's stencil without the boundary's potential,
and the end point's control volume reaches to the surface, `h_0 + h_1/2` (or `h_NP + h_{NP−1}/2`)
instead of half of each interval:

```
point 1, left end free:  ε₁·(V₂ − V₁)/(L·h₁) = q·(n₁ − p₁ + N_A − N_D)·vol₁,   vol₁ = L·(h₀ + h₁/2)
                         F_n(edge 0) = F_p(edge 0) = 0
```

The boundary point then holds its neighbour's state (the solution is flat across the dead
interval), which is what `solve_state` returns there. In a device with a free end Poisson's
equation is in flux form at every point (each interval's `ε` its left point's), and its mobile
ions are counted over the control volumes: both keep the film's charge exactly, where
SIMsalabim's forms do not ([solver.md](solver.md#free-surfaces)).

**Surface recombination.** The end's `S_n_L`, `S_p_L` (or `S_n_R`, `S_p_R`) are the surface's SRH
velocities (m/s, ≥ 0; a negative, SIMsalabim's "infinite", is refused at a free end), a level at
mid-gap:

```
R_s = S_n·S_p·(n·p − n_i²) / (S_n·(n + n_i) + S_p·(p + n_i))      m⁻² s⁻¹
```

It enters the end point's continuity equations as a recombination over its control volume,
`R_s/vol`, the same for both carriers.

**The gauge.** With one free end the other is a contact: the potential is fixed there, and the
applied voltage only shifts the potential as a whole (no current flows in the steady state, so
`J = 0`). With both ends free the potential is fixed nowhere: `V + c` solves the equations
wherever `V` does, and the film's charge is fixed only by Poisson's equation with no field at
either end. The solver fixes `V = 0` at the last point ([solver.md](solver.md#free-surfaces)).
The applied voltage and a free end's work function (taken as `sfb`'s) change nothing but that
gauge.

**What photoluminescence sees** (`solve_state`, `StateResult`): the splitting
`QFLS(x) = V_T·ln(n·p/n_i²)` at each point; the implied open-circuit voltage
`V_T·ln(∫k·n·p dx / ∫k·n_i² dx)`, the splitting that direct (radiative) recombination's emission
sees; and the share of the generated pairs that recombine directly, `∫k·(n·p − n_i²) dx / ∫G dx`,
the PLQY where all direct recombination is radiative and every photon leaves the film (neither
reabsorption nor outcoupling is modelled in the steady state; photon recycling enters a
transient, below).

A batch takes free ends beside contacted devices: `solve_states` gives the same per device,
differentiable in `device_tree`'s parameters; `solve_batch`'s current is 0 for them.

**Mobile ions** reach a free end as they reach a contact: none leaves its region, and in the
steady state they follow the potential. Over the end's dead interval the potential is flat, so
the ions' count (`_ion_constraints`, trapezoidal) takes the end point's potential at the boundary
point too (`_free_potential`), and the ions there are the end point's, as the carriers are.
A pre-bias (`preCond`, `fixIons`) changes nothing at a free end, where no voltage moves the
state. `solve_state` and `solve_states` return the ions (`anion`, `cation`, mobile and fixed).

**The rates in the field** (Braun's generation, Langevin's rate in a field-dependent mobility)
read the potential of the point's neighbours. At a free end the boundary point is its neighbour's
state (`solver._free_ends`), so the end point's field is `(V₂ − V₁)/(x₂ − x₀)`: none over the dead
interval, as there is none beyond the surface, averaged with the field after it. The implied
`V_oc` and the PLQY take the direct rate in the field where it follows the field.

**In time** (`solve_transient`) a free end is what it is in the steady state, at every step:
no flux over it, no field beyond it. The total current (the carriers', the ions' and the
displacement current) is then 0 everywhere. With one contact, its voltage only moves the
potential's gauge, and an open circuit is taken as the voltage 0. The surface's states follow
at once (no occupation of their own over a step). A transient photoluminescence decay is a
`tVGFile` with the light on at `t = 0` and off after it; the splitting is the `QFLS` column of
each layer (`tj_columns`).

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

### Ions bounded by their sites (jaxdd's, off by default)

Boltzmann's profile lets free ions pile up without limit: frozen at full accumulation, 1e18
cm⁻³ of them reach 1.5e21 cm⁻³ in nanometre layers at the interfaces. Real ions sit on lattice
sites, at most one each. A layer's `N_max_anion` and `N_max_cation` (m⁻³, in its parameter
file or the parameters' dict, `device.ION_SITE_PARAMETERS`; SIMsalabim has neither) bound each
mobile species by its density of sites, the occupancy of a site (the Fermi–Dirac integral of
order −1, Bikerman's modified Poisson–Boltzmann):

```
n_anion(x)  = N_max / (1 + exp(−(ln C_a + V(x)/V_T − ln N_max)))  =  b / (1 + b/N_max),
b = C_a·exp(+V(x)/V_T)        (cations with −V)
```

`b` is Boltzmann's density with the same normalisation, which `C_a` again sets to the region's
number of ions; far below `N_max` the two are equal, and with `N_max` infinite (the default)
the profile is Boltzmann's to the bit (`solver.occupancy`). The free steady state takes it, and
so the frozen profile after a pre-bias. A layer's mean density must lie below its sites
(refused otherwise). Whether a device has sites at all is part of its structure
(`Structure.ion_sites`, any finite `N_max` in the input): the parameters' `N_max` then move
them, differentiably; set in the parameters of an input without any, they do nothing.

What `N_max` is: a model parameter between the mean density and the density of the sites. The
halide sites are `3/a³` (three per unit cell), 1.21e22 cm⁻³ for MAPbI₃ (a = 6.28 Å; Abdel,
Courtier, Farrell, Opt. Quantum Electron. 55, 884 (2023), after Eames et al., Nat. Commun. 6,
7497 (2015)) and 1.17–1.26e22 cm⁻³ for the FA/Cs perovskites' pseudo-cubic `a` of about
6.2–6.36 Å; the A-site cations' `1/a³` about 4e21 cm⁻³. Drift-diffusion codes take it far
lower, as a parameter that limits the accumulation: IonMonger's template 2e21 cm⁻³ (`Plim`,
off by default, against a mean of 1.6e19; Clarke et al., J. Comput. Electron. 22, 364 (2023)),
ChargeTransport.jl's parameter files 1e21 and 1e20 cm⁻³ (62 and 100 times their mean). Which
defect each of SimSS's species stands for is the input's to say: the halide vacancies, positive,
are its cations; its anions, negative, cation vacancies or halide interstitials, each with its
own sites.

**In time** (`solve_transient`) a bounded species flows down its electrochemical potential at a
constant mobility, `j = −μ·n·∇(ln n + w ± V/V_T)·V_T` with the excess chemical potential
`w = −ln(1 − n/N_max)`: nonlinear diffusion, whose zero flux is the steady state's occupancy
(`solver._excess`). Discretised, the Scharfetter–Gummel flux with `w`'s difference over the
interval added to the potential's, `x + sign·Δw` (ChargeTransport.jl's excess chemical
potential scheme; with `w` = 0, no sites, the flux is Boltzmann's to the bit); the same `w` in
the ion's steady row (`ln n + w` Boltzmann's), at a contact's point (zero flux over its
interval: `ln n = −ln(e^(−η) + 1/N_max)`, `η` the neighbour's `ln n + w` and the step, exact for
the exact `B`) and in the ions' current (its Slotboom variable `n·e^(±V/V_T + w)`).

`solve_transient(..., ion_flux="modified-drift")` takes the other dynamics: the drift modified
by the free sites' share at a constant diffusivity, `j = −D·(∇n ± n·(1 − n/N_max)·∇V/V_T)`
(Abdel, Courtier, Farrell 2023). It is the flux above with the mobility times
`1 − n/N_max = e^(−w)`, so its zero flux, the steady state, is the same, and its transients are
slower where the ions crowd their sites. Discretised as Bessemoulin-Chatard's scheme with a
modified drift (ChargeTransport.jl's `DiffusionEnhancedModifiedDrift`): the Scharfetter–Gummel
flux at the constant mobility with the potential's argument over `g = Δη/Δln n`, `η = ln n + w`
(`solver._drift_factor`; below `Δln n` = 1e-5 its value at the middle, `e^w`), whose zero flux
is `Δη = ±x`, the steady state's; the contact's point as above (the same `Δη` for the exact
`B`), the ions' current with the mobility `μ·e^(−w)` in the integral form and `x/g` in
`Calc_Curr_Diff`'s. Without sites `g` = 1 and both fluxes are Boltzmann's to the bit. The
transients of both agree with ChargeTransport.jl's ([validation.md](validation.md)); they differ
from each other by up to the current itself. Newton
starts from the solution with Boltzmann's ions ([solver.md](solver.md)) and converges at their
steps on the hard devices with the sites of the lattice or the codes
([validation.md](validation.md)); where the ions fill much of their layer (sites a few times
their mean density) a voltage may not converge, and is reported so.

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
`e^{l_k}`, which is a slot (one unknown per region, its constraint the total, as in the steady state). With
`dti = 0` the row is Boltzmann's (bounded by their sites the occupancy's, above) and `l_k` its
normalisation, as in the steady state. Outside
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
`varFile` is not written; `solve_transient(..., profiles=True)` returns its core instead, each
step's potential and densities on every point (`TransientResult.profiles`, `V`, `n`, `p`, the
ions `anion` and `cation`, 0 outside their regions; a free end's boundary point its neighbour's
state, as the solver reads it), for band diagrams in time, with the traps' occupations
(`f_bulk`, `f_interface`) and each step's `dti`. The QFLS is `V_T·ln(n·p/n_i²)`. It
takes the logarithm of densities far below any that moves a current, below Newton's floor of
convergence (`n_floor`, 1e-12 of the largest `N_c`): the minority carriers in the dark and in a
transport layer. Newton steps those in the density, not its logarithm, so that they converge
with the rest although they do not count for convergence (docs/solver.md, "When is a voltage
converged?"); the QFLS is then the converged solution's at every step (docs/validation.md).

### Going on from a state (jaxdd's)

ZimT starts every run from a steady state. A caller that owns the time loop — one that ages a
device step by step and measures it in between — needs the next transient to start where the
last one left the device: slow ions off their steady state, traps part filled. A step's state is
what the next step reads of it: the potential, the densities and the ions on every point, the
traps' occupations, the internal voltage, and the step's `dti`, which the contacts' ions take
(`_ion_full`). `TransientResult.final` is the last state kept, `TransientResult.state(k)` step
`k`'s (from the profiles), both a `TransientState`; `solve_transient(..., initial=state)` starts
every device from it, each step's `Δt` from its `t`. The interior points are the unknowns'
start, the contacts follow from the device, the ions' slots start at a scale of 1 as after a
steady state. A transient split at any step and continued so gives the unsplit one's currents
to 1e-11 of their scale and its densities within the Newton tolerance
(`tests/test_transient_state.py`): only the first step's first guess differs, which is not
extrapolated from the step before. ZimT's automatic stop counts from the start again.

## Impedance (jaxdd's, `solve_impedance`)

ZimT measures an impedance in time, a transient per frequency. jaxdd solves it in the frequency
domain (`jaxdd.impedance`): at the steady state `y₀` at a voltage (`solve_states`), a small
voltage `δV·e^(iωt)` drives `(J + iω·M)·δy = −(∂F/∂V)·δV`, with `J` the steady residual's
Jacobian and `M` the time terms that implicit Euler adds over a step (`−fac·n` in each continuity
row's own `ln n`). That is one complex block-tridiagonal solve per frequency, by the steady
state's sweep.

- **Mobile ions** are unknowns of their own, as in a transient (`_assemble_ions` at `dti` = 1, its
  time terms `mass=True`). Each slot's number is a constraint, trapezoidal over its region as ZimT's
  `Rescale_Ion_Density` keeps it. That constraint also matters numerically: without it, only the
  time term `iω·n` fixes the ions' electrochemical potential in a region, and the system's
  condition grows as `1/(ωτ)`. With sites, the flux is the one `ion_flux` names.
- **The bulk traps' occupations** have their own time constant, `df/dt = b − a·f`. They are
  eliminated point by point, `δf = (∂b − f·∂a)·δy/(iω + a)`, which replaces the steady Jacobian's
  instantaneous occupation in the rates and in the traps' charge.
- **The interface traps' occupations** likewise, one per edge with traps, from both of its
  points' densities (`_interface_beyond`): through their four rates and their charge they
  couple the edge's two points, in the blocks `A`, `B` and `C` (`_interface_blocks`), and they
  enter the current through each point's interface rate (`tr["dRi"]`).
- **The current** is a transient's, `batch._current`: the integral form over each point's control
  volume, the ions' current and the displacement current, averaged over the intervals. It is
  affine in `dti`, so `δJ = J_y·δy + iω·(J_y(1) − J_y(0))·δy`, plus the traps' rates beyond their
  steady response (`tr["dU"]`). A single interval's Scharfetter–Gummel flux would not do: in a
  doped layer it is the difference of majority fluxes of 1e9–1e12 A/m², which loses the
  current to round-off.

`Y = δJ/δV_int` in SIMsalabim's sign, and the external `Z = 1/(Y + G_shunt) + R_series`.

With `dt=` the response is what `solve_transient` sees through implicit Euler at steps of `dt`:
`(1 − e^(−iωΔt))/Δt` in place of `iω`, and the traps over each step as `_trap_step` solves them
(the occupation at the step's end in the charge, its mean in the rates). This is what ZimT's
impedance from a sine would measure at that step, except for the sine's amplitude and settling,
and it is how the solve is tested against the transients ([validation.md](validation.md)).

Refused: a free end (there is no circuit through it) and ions frozen at a pre-bias.

## Temperature laws (jaxdd's, `jaxdd.temperature`)

SIMsalabim's `T` enters a device only through `V_T = kT/q`: `N_c`, the band edges, the
mobilities, the rate and capture coefficients, the surface recombination velocities, the ions'
mobilities and `ν_int` are the input's at any `T`. `at_temperature(params, T, laws, gap)` maps
`device_parameters`' dict from its own `T`, the reference `T0`, to another `T`, in front of
`device_tree`; the solver is unchanged. It is a pure function of the dict, in NumPy or
`jax.numpy`: differentiable in `T` and in every coefficient of its laws, batched by `vmap` over
`T`, and it composes with `least_squares_many` and the `axes` tables as any function that builds
the dict does. The forms are jaxdd's, the coefficients the caller's; nothing changes with `T`
unless a law names it.

**A parameter's law** (`Law(m, E)`, each a number or one per layer):

```
P(T) = P(T0)·(T/T0)^m·exp(−(E/k)·(1/T − 1/T0))
```

on any parameter of the dict but `T` and the energies: `N_c`, `mu_n`, `mu_p`, `gamma_n`,
`gamma_p`, `k_direct`, the capture coefficients (a law on `C_n_bulk` acts on a trap file's
levels' own `bulk_trap_Cn` too, a layer's coefficients for all of its levels, unless one is given
for those), `C_n_Auger`, `C_p_Auger`, `S_n_L`, …, `mu_anion`, `mu_cation`, `nu_int_n`,
`nu_int_p`. A law may instead be the caller's function `law(value, T, T0)`: two power-law
regimes, a phase transition, a disorder model's `exp(−(2σ/3kT)²)`. Passed on to a trap file's
levels, the function gets their coefficients levels × layers, so that a factor per layer
broadcasts over the layers, and its result is transposed back. A law keeps its value's shape:
one per layer on a single value (`S_n_L`, say) is refused. `textbook_laws` gives the textbook
ones: `N_c` a parabolic band's `T^{3/2}`, `ν_int` a thermal velocity's `T^{1/2}`, and each ion
species' mobility from an Arrhenius diffusion with the caller's activation energy (0 by
default, a constant `D`), `D ∝ exp(−E/kT)`, `μ = D/V_T ∝ T^{−1}·exp(−E/kT)` (computed as `μ`'s
factor, not through `D`).
None is applied unless passed. `k_direct` is not tied to the absorption edge: a measured
`k₂(T)` follows no general law, and its law is the caller's. In a layer with `useLangevin` the
rate is the mobilities' (with their laws), and `k_direct`'s law does nothing there.

**The band gap** (`Gap(slope, share)`, per layer): `E_g(T) = E_g(T0) + slope·(T − T0)`, or the
caller's `Eg(T)` (Varshni's, say), of which `Eg(T) − Eg(T0)` counts. Its change `ΔE_g` moves
`E_v` by `share·ΔE_g` and `E_c` by `−(1 − share)·ΔE_g` (energies below the vacuum level: a gap
that grows takes `E_v` deeper and `E_c` towards the vacuum); `share` 0.5 by default. `n_i`, the
traps' `n_t0`, `p_t0` and an `sfb` contact's work function follow in the device. The other
absolute energies follow what they are told (`FOLLOW`), their layer's bands moving:

| | `"fixed"` | `"E_c"` (`"E_v"`) | `"midgap"` |
|---|---|---|---|
| trap levels (`traps=`, one for all or one per layer) | the default: their energy stays | their distance to that band stays | their distance to the gap's middle stays |
| work functions (`contacts=`, one or (left, right)) | the default: the electrode's stays | the distance to the adjacent layer's band stays | the distance to its gap's middle stays |

A layer's trap levels are its `E_t_bulk` or `bulk_trap_E`, and its interface level to the next
(`E_t_int`, `int_trap_E`), which belongs to the layer on its left as in `device_parameters`.
Fixed by default: a level's energy below the vacuum is what is known of a deep defect, which
neither band carries; a shallow level follows its band. An 'sfb' or 'free' end's `W` is not
read; with the `Structure` given (`structure=`) it is not moved either. With `edge=True` the gap's
change is added to each layer's `dE_nk` (the shift of its n,k in photon energy, which the
generation computed by the transfer-matrix method takes), where the dict has it: the absorption
follows the gap.

**The gap check.** `check_gaps(structure, params)` checks a mapped dict's energies: each layer's
gap open (`E_v > E_c`), and as `device_parameters` checks the input, each work function (not
'sfb' or 'free') within its layer's gap, each trap level of a layer with traps strictly inside
its gap, an interface's inside both layers' (a trap file's levels of density 0 aside);
`ValueError` naming what is outside. On
concrete values, outside any trace: a dict, or a stack of them (several temperatures by `vmap`),
every one checked. Inside a trace nothing checks, as nothing checks a dict changed by hand.

**What stays.** A law scales what the input has; it cannot switch on a mechanism the input
lacks, which the `Structure` holds: traps on a layer without them, Auger in a device without a
coefficient, a finite `S` where it is infinite (a negative `S` stays as it is). The `Structure`'s
own uses of `T` are the input's: the graded grid's `d₀` at the contacts under Fermi–Dirac's or
Blakemore's statistics (the Debye length at the input's `T` and `N_c`), and `read_tvg`'s bound on a
voltage over `V_T`. Photon recycling's kernel (`recycling.device_kernel`) is the caller's to build
at each temperature, with its `T` and its absorption edge.

**Trap re-emission counted once.** A level's emission is detailed balance's, in `n_t0 =
N_c·exp((E_c − E_t)/V_T)` and `p_t0`: exponential in `T` through `V_T` already, and following
`N_c`'s law and the bands. A capture coefficient's law is the capture's alone (a thermal
velocity's `T^{1/2}`, a multiphonon barrier's activation), not an emission's Arrhenius on top.

**At `T0` to the bit.** Every law returns its value unchanged at `T = T0`: the factor is
`1^m·exp(−0)`, exactly 1, the gap's change `slope·0`, exactly 0. So a mapped dict gives the
unmapped device leaf for leaf, in NumPy and in JAX, eager and compiled with the dict and `T`
arguments of the compiled function. Folded in as a constant, a `T` or a whole dict is rounded
by XLA's constant folding, a few ulp from the eager values, mapped or not, as any dict whose `T`
is set by hand. A caller's law should do the same: `v·(g(T)/g(T0))` does for any `g`,
`v·g(T)/g(T0)` does not (it rounds twice). Tested (`tests/test_temperature.py`): at `T0` with
every law on, to the bit, on devices with ions, bulk and interface traps, trap files, finite
surfaces and Langevin's rate; each law against its closed form, a caller's per layer on a trap
file's levels too; the gradients in the powers, the activation energies and the gap's slope and share,
30 K from `T0`, through the steady state and through a scan with mobile ions, against central
finite differences; the gap check against levels and a work function moved out.

## Constants

SIMsalabim's, not CODATA's: `q = 1.6022e-19 C`, `k = 1.3807e-23 J/K`, `ε₀ = 8.8542e-12 F/m`. A
difference in the fifth digit would move `V_T` and every exponential with it.
