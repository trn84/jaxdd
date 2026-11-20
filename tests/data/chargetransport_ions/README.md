# ChargeTransport.jl's reference for ions bounded by their sites

`tests/test_chargetransport.py` compares jaxdd with these. The cell is ChargeTransport.jl's
Ex103 (TiO2 | MAPI | spiro-OMeTAD) with mobile vacancies bounded by the Fermi–Dirac integral of
order −1 (docs/validation.md, "Ions bounded by their sites, against ChargeTransport.jl").

- `simulation_setup.txt`, `L1.txt`–`L3.txt`: the cell in SIMsalabim's input, as jaxdd builds it.
  The test sets the vacancies' density (`N_cation`, with as many immobile `N_anion`) and their
  sites (`N_max_cation`), and `G_frac`.
- `ions1e24_sites1e25/`, `ions4e24_sites2e25/`: ChargeTransport.jl 1.10.0's states, written by

  ```
  cd validation/chargetransport
  CA=1e24 NA=1e25 julia --project=. psc_ions.jl ../../tests/data/chargetransport_ions/ions1e24_sites1e25 \
      eq dark:-0.3 dark:0.6 dark:1.0 light:0.0 light:0.9
  ```

  (likewise with `CA=4e24 NA=2e25`). Each `<case>.csv` holds region, x, ψ, n, p and the
  vacancies on each region's nodes (an interface's node in both), its first line the terminal
  current; `params.txt` the vacancies' energy found and the grid.
