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
- `ions1.6e25_sites1e27/`: Ex103's own cell (`CA=1.6e25 NA=1e27`), the same cases.
- `*/tstep_dark_V_*` (1e24: 1.0 V; 4e24: 0.6 and 1.0 V; 1.6e25: 1.0 V): the voltage's step in time (`tstep:dark:V`,
  the vacancies at 1e-12 m²/Vs): `_J.csv` the current at each of ChargeTransport's times,
  `_at1`–`_at3` the states nearest 1 µs, 1 ms and 1 s (`_t.txt` their times).
