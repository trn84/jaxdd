# Measured TRPL decays: Kober-Czerny et al. (2025)

Three files of the dataset of M. Kober-Czerny, A. Dasgupta, S. Seo, F. M. Rombach, D. P. McMeekin,
H. Jin and H. J. Snaith, "Determining Parameters of Metal-Halide Perovskites Using
Photoluminescence with Bayesian Inference", PRX Energy 4, 013001 (2025),
doi:10.1103/PRXEnergy.4.013001; the dataset doi:10.5281/zenodo.14231027 (Zenodo, `TRPL_Data.zip`),
licence [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Time-correlated single-photon counting at 398 nm, 2 ns bins, of a FAPbI3 film of about 600 nm on
glass and of two half-stacks without electrodes:

| File | What |
|---|---|
| `Seo_FAPI_nuts_data_norm.dat` | the film on glass: four decays, normalised to their first bin, `Time (ns)` then 4.8e11 and 4.8e10 photons/cm² lit from the substrate (`S1`) and from the film (`S2`) |
| `Seo_SnO2_Glass_0_processed_ode.txt` | glass/SnO2/FAPbI3: lit through the glass at 4.8e11 (`OD0.0`) and 1.5e11 photons/cm² (`OD0.5`), and from the film at 4.8e11, normalised, from −40 ns; then the authors' fits |
| `Seo_Spiro_Perov.dat` | glass/FAPbI3/Spiro-OMeTAD: one raw decay (counts per bin after the instrument's header), lit from the film side at 4.8e11 photons/cm², 200 kHz |

Changed here: the line ends (CRLF to LF), nothing else. `validation/trpl_*.py` read them
([docs/validation.md](../../../docs/validation.md#trpl-without-electrodes)).
