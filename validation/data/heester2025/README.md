# Measured J-V curves: Heester et al. (2025)

The fifteen J-V curves of S. Heester, F. Ventosinos, L. Gil-Escrig, M. Sessolo, H. J. Bolink and
L. J. A. Koster, "Drift-diffusion modeling of perovskite solar cells: A systematic approach",
APL Energy 3, 036106 (2025), doi:10.1063/5.0268013; the dataset doi:10.34894/EDG5SI
(DataverseNL), licence [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Three samples of a p-i-n cell (ITO / 10 nm TaTm / 500 nm FA0.8Cs0.2Pb(I:Br:Cl)3 / 20 nm C60 /
BCP / Ag) with 0.1, 0.2 and 0.6 parts of PbCl2 (`01Cl`, `02Cl`, `06Cl`), each in a forward scan at
1, 0.83, 0.53, 0.38 and 0.10 sun as the file names say: `Vext` (V) and `Jext` (A/m², SimSS's
sign). Changed here: the line ends (CRLF to LF), nothing else. `validation/fit_measured.py` fits
them.
