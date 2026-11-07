# Reference curves

`thin_l1_light/`: a SimSS 5.36 run (SIMsalabim, LGPL-3.0, doi:10.21105/joss.03727) — its input
files and its `JV.dat`, unchanged. The device is a point of FauxPV's drift-diffusion table
(`L_diff/d` 1 at short circuit, 1 nm undoped transport layers, no ions, under light), written
by FauxPV's `tools/dd_tables.py`. SimSS's output is not covered by its licence; cite SIMsalabim
where it is used.

`frozen_1e17_x1_thin_light/`: the same device with 1e17 cm⁻³ of mobile ions, settled at the
pre-bias `V_pre = 0` under light and frozen for the scan (`preCond`, `fixIons`), run by SimSS
5.36 with FauxPV's `tools/simss/fixed_ions.patch` (Poisson leaves fixed ions alone).

`steady_1e17_thin_light/`: the same ions free at every voltage, the unpatched SimSS 5.36.
