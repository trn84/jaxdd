# Reference curves

`thin_l1_light/`: a SimSS 5.36 run (SIMsalabim, LGPL-3.0, doi:10.21105/joss.03727) — its input
files and its `JV.dat`, unchanged. The device is a perovskite cell with 1 nm undoped transport
layers, no ions, under light, its absorber's diffusion length equal to its thickness at short
circuit. SimSS's output is not covered by its licence; cite SIMsalabim where it is used.

`frozen_1e17_x1_thin_light/`: the same device with 1e17 cm⁻³ of mobile ions, settled at the
pre-bias `V_pre = 0` under light and frozen for the scan (`preCond`, `fixIons`), run by SimSS
5.36 with [`simss/fixed_ions.patch`](../../simss/fixed_ions.patch) (Poisson leaves fixed ions
alone).

`steady_1e17_thin_light/`: the same ions free at every voltage, the unpatched SimSS 5.36.

`frozen_thick_htl_dark/`: a different stack, in the dark: a 1000 nm absorber whose diffusion
length is half its thickness, 2.5e14 cm⁻³ of mobile ions settled at the pre-bias `V_pre = 1.05`
V (about flat band) and frozen; a 2 nm undoped ETL and a 2000 nm HTL p-doped at 2.5e14 cm⁻³.
In the dark, the electrons in that HTL and the holes across much of the absorber lie far below
jaxdd's convergence floor, which once made the batched scan's extrapolated start fail from
short circuit on (docs/solver.md). SimSS
5.36 with the patch above; its first two voltages are SimSS stopping early
(docs/differences.md §1).

`iface_charged_light/`: `thin_l1_light` with interface traps (Stage B), a donor level of 3e14
m⁻² at the left interface and an acceptor level of 3e14 m⁻² at the right one (`E_t_int` 4.7 eV,
`C_n_int = C_p_int` = 2e-14 m³/s); SimSS 5.36. `iface_ions_light/`: the same traps in
`steady_1e17_thin_light`. `pin_iface_light/`: a p-i-n stack mirrored to put the cathode left (20 nm
of C60-like ETL, 500 nm absorber with free ions, 10 nm of TaTm-like HTL, the parameters of the
0.2 Cl sample of Heester et al., APL Energy 3, 036106 (2025), Table S2, with a uniform
generation), 4.6e14 and 5e13 m⁻² of donor traps at its interfaces; SimSS 5.36 at
`tolPois = tolDens = 1e-9`, which converges up to 0.975 V and no further.

`bulk_acceptor_light/`: `thin_l1_light` with its absorber's traps (3.9e21 m⁻³ at 4.8 eV)
charged, as acceptors (`bulkTrapType = -1`; Stage B). `trapfiles_light/`: `thin_l1_light` with
three donor levels in the absorber from `bulk_traps.txt` (1e21, 2e21 and 1e21 m⁻³ at 4.5, 4.8
and 5.1 eV), two acceptor levels at the left interface from `int_traps.txt` (1.5e14 m⁻² each at
4.6 and 5.0 eV) and one donor level of 3e14 m⁻² at 4.7 eV at the right interface. Both SimSS
5.36 at `tolPois = tolDens = 1e-9`.

`surface_light/`: `thin_l1_light` with finite surface recombination for the majority carriers,
20 m/s for the electrons at the left contact and 50 m/s for the holes at the right one (`S_n_L`,
`S_p_R`, the two ends SimSS 5.36 has right, docs/differences.md §7); SimSS 5.36 at
`tolPois = tolDens = 1e-9`.

`langevin_light/`: `thin_l1_light` with Langevin recombination in all three layers
(`useLangevin = 1`, `preLangevin` 1e-3, 1e-2, 1e-3); SimSS 5.36 at `tolPois = tolDens = 1e-9`.

`fieldmob_light/`: `thin_l1_light` with the field-dependent mobility (`mobnDep`, `mobpDep` = 1):
`gamma_n`, `gamma_p` 1e-4 and 0 (m/V)^½ in the first layer, 3e-4 and 2e-4 in the absorber, the
electrons' mobility constant and `gamma_p` 1e-4 in the last; SimSS 5.36 at
`tolPois = tolDens = 1e-9`. `langevin_field_light/`: `fieldmob_light` with `langevin_light`'s
Langevin recombination in all three layers; SimSS 5.36 at `tolPois = tolDens = 1e-9`.

`resist_light/`: `thin_l1_light` with a series resistance of 2e-4 Ω m² and a shunt of 0.3 Ω m²
(Stage B); SimSS 5.36.

`calc_light/` and `pin_calc_light/`: `thin_l1_light` and `pin_iface_light` with the generation
profile by the transfer-matrix method (`genProfile = calc`, 350–800 nm): glass, 110 or 150 nm of
ITO, the layers (PCBM/MAPI/PTAA and C60/MAPI/TaTm n,k data) and 100 nm of gold or silver. The n,k
files and AM1.5G are SIMsalabim's (`Data/`, their sources in its `Biblio_info_to_cite.txt`).
`G_simss.dat` is SimSS's profile on its grid (from its variable file); SimSS 5.36.
`none_light/`: `thin_l1_light` with a uniform generation of 1.5e28 m⁻³ s⁻¹ in the absorber
(`genProfile = none`); SimSS 5.36.

`pin_anode_left_light/`: the stack of `pin_iface_light` in its own orientation, the anode on the
left (`leftElec = 1`, glass / 150 nm ITO / TaTm / perovskite / C60 / 100 nm Ag), lit through the
anode with the transfer-matrix profile (the perovskite's n,k data SIMsalabim's `nk_peroTripleCat`,
a stand-in), 5e13 and 4.6e14 m⁻² of donor traps at its interfaces, free ions and a shunt of
1.32 Ω m²; SimSS 5.36 at `tolPois = tolDens = 1e-9`.
