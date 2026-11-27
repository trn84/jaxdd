# Plan

What is planned for jaxdd and what may come. What exists and how it works is in the handbook,
[docs/](docs/README.md). Each item says what is to be done, why, and when it is done; finished
work leaves this file, and what it taught goes into the handbook. Measurements belong in the
handbook or the commit message, not here.

## 1 Next

### 1.1 Temperature laws as a parameter layer

jaxdd, like SIMsalabim, lets the device's `T` enter only through $V_T = kT/q$; $N_c$, the band
edges, the mobilities, the rate coefficients, the surface recombination, the ions' mobility and
$\nu_\mathrm{int}$ are constants. A series of devices at several temperatures (accelerated
ageing, say) needs them to follow $T$. A module `jaxdd.temperature` with a pure function that
maps `device_parameters`' dict from its own `T` (the reference $T_0$) to another $T$, in front
of `device_tree`, so that it composes with `jax.grad`, `vmap`, `least_squares_many` and the
`axes` tables like any function that builds the parameter dict; no change to the solver. The
forms are jaxdd's, the coefficients the caller's:

- the band gap linear in $T$, its change split between $E_c$ and $E_v$ by a parameter, per layer;
- every density-of-states, mobility or rate parameter (`N_c`, `mu_n`, `mu_p`, `gamma_*`,
  `k_direct`, the capture coefficients with the trap files' levels, `C_*_Auger`, `S_*`,
  `mu_anion`, `mu_cation`, `nu_int_*`) as $(T/T_0)^m\,e^{-(E/k)(1/T - 1/T_0)}$, with $N_c$'s
  $m = 3/2$, $\nu_\mathrm{int}$'s $1/2$ and the ions' $-1$ (an Arrhenius $D$, $\mu = D/V_T$) as
  the defaults; $k_\mathrm{direct}$ is not tied to the absorption edge (measured $k_2(T)$
  follows no general law);
- any law also as the caller's own function of $T$ (a phase transition, Varshni's gap, two
  power-law regimes, a disorder model's $e^{-(2\sigma/3kT)^2}$).

To decide and document with it: what trap levels and fixed work functions (absolute energies)
follow when the bands move; a check of a mapped dict against the gaps, on concrete values outside
any trace, as `device_parameters` checks the input; that a law scales what the input has and
cannot switch on a mechanism it lacks (a finite `S`, traps), and does nothing to
$k_\mathrm{direct}$ in a Langevin layer; how not to count trap re-emission twice ($n_1$, $p_1$
carry it already); that the `Structure` stays the input's (the graded grid's contact spacing
under Fermi-Dirac or Blakemore statistics, `read_tvg`'s voltage bound) and the recycling kernel
is built by the caller at each temperature.

**Done when** at $T = T_0$ the mapped dict gives a device tree equal leaf for leaf
(`array_equal`) to the unmapped one on the test devices (ions, traps, interfaces, surfaces);
each law matches its closed form; gradients in the exponents and energies, away from $T_0$,
match finite differences; the gap check refuses a level or work function moved out of its gap;
docs/model.md describes the laws, and docs/tutorial.md shows a JV scan series over $T$ on a
device with mobile ions (one scan rate puts the ions in different regimes at different $T$).

### 1.2 An absorption edge that follows the band gap

With `genProfile = calc` the layers' n,k are NumPy constants in the `Structure`: an edge that
moves with $T$ needs one `Structure` per temperature, cannot be batched over $T$, and has no
gradient in the gap's slope. A per-layer shift of n,k in photon energy, traced, inside
`device_arrays`: `optical_data` keeps n,k on a grid wide enough for the largest shift, and the
interpolation and the rest of optics.py run in `xp`.

**Done when** a zero shift gives the same bits as now; a shifted profile matches the one from n,k
files shifted beforehand within the interpolation's error; its gradient in the shift, between
data nodes, matches finite differences; the SimSS cases with computed generation still pass.

## 2 Waiting

### 2.1 M8: the MI50 cards (waiting for a run on the host)

Not the focus: jaxdd is made as fast as it can be on NVIDIA first. An AMD track with its own
optimisations follows later, beside it, starting from the scan, which runs everywhere. Prepared:
JAX 0.4.31 in `rocm/jax-community:rocm6.0.3-jax0.4.31-py3.11.10` runs `jax.jit` on gfx906, and
jaxdd passes its tests on JAX 0.4.31 ([docs/hardware.md](docs/hardware.md)).

1. On the MI50 host: `rocm/mi50.sh`, then send `rocm/mi50_probe.log`.
2. The batch a 16 GB card holds; the four cards as four processes (`HIP_VISIBLE_DEVICES`).
3. A large sweep split across the four cards.

If the image fails: the ROCm 6.1.3 and 6.2.3 images with the same JAX; then jaxlib built from
source for gfx906 against a ROCm that still ships its libraries (6.2/6.3); last, the block sweep
as a hand-written HIP kernel beside JAX.

**Done when** the benchmark's results are reproduced on the cards within the validation
criteria, at a throughput worth the cards (four MI50 are, for this bandwidth-bound FP64 solver,
about two A100s).

## 3 Later

### 3.1 Ageing simulators: what may still be needed

The four items an ageing caller needed (a transient from a given state, the ions in the
profiles, Auger, bit-identity on the CPU) are done (CHANGELOG.md). Not taken up until a caller
shows it needs them:

- **A temperature changing in time within a transient** (a scan that warms the device; jaxdd's
  `T` is one per device and call). A caller can split the transient at temperature nodes and go
  on from each segment's state with the next temperature's device (1.1; with computed
  generation, 1.2).
- **Two ion species of one sign** (jaxdd has one anion and one cation species), e.g. a second,
  slow species beside a fast one.

### 3.2 Tunnelling at contacts and interfaces

Thermionic transfer alone (jaxdd's and SIMsalabim's) may give a contact behind a thin barrier,
such as a self-assembled monolayer, too strong a temperature dependence: through a barrier 1-2 nm
thick and half an eV or more high, tunnelling can carry more current than emission over it. Not
taken up until a temperature series shows that the interfaces with 1.1's laws do not describe it.
In order of effort:

- **Two channels through a barrier that is not on the grid** (a monolayer or dipole as a property
  of the contact or interface, not a layer): $S$ or $\nu_\mathrm{int}$ as
  $v_\mathrm{TE}(T)\,e^{-\Phi/kT} + v_\mathrm{tun}$, a law in 1.1's layer, no solver change. The
  grid's own offset keeps its Boltzmann factor. At a contact it needs a finite `S` in the input
  and, to act in both directions, `surface="robin"` (`"extraction"` limits extraction only).
  **Done when** its limits ($v_\mathrm{tun} = 0$; $\Phi \to \infty$) are the single channels and
  docs/model.md says when it applies.
- **Thermionic-field emission through a barrier on the grid** (Yang, East, Haddad, Solid-State
  Electron. 36, 321 (1993)): $\nu_\mathrm{int}(1 + \delta)$, $\delta$ a WKB integral over the
  neighbouring layer's band, which depends on the solution at many points and so is coupled into
  Newton. To decide first: $\delta$ on jaxdd's interface flux or Yang's jump condition in its
  place (they differ by about $\Delta E/kT$ at large offsets). **Done when** $\delta = 0$ passes
  the SimSS interface cases unchanged, the Jacobian matches autodiff, Yang's heterojunction is
  reproduced from its published parameters, and the structure of Verschraegen and Burgelman (Thin
  Solid Films 515, 6276 (2007)) within the difference their refinement makes.

## 4 Not planned

- **A fused CUDA kernel**: decided against for now; why, and when it comes back, is in
  [docs/hardware.md](docs/hardware.md#why-there-is-no-fused-kernel).
- **Two dimensions**: a project of its own ([docs/overview.md](docs/overview.md#what-the-speed-is-for)).
