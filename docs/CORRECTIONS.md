# Corrections to the earlier notebooks and conversation summaries

The project began as Colab notebooks (`M3_Mineral_Mapping_*.ipynb`, `Step1/Step2*.py`)
and two conversation summaries (`M3_Mineral_Mapping_Conversation*.pdf`). This file lists
every statement or code behaviour from those sources that turned out to be wrong, what is
actually true, and how `lunacorder` handles it. Items marked **verified** were checked
against the real PDS headers of scene `m3g20090607t025544_v01` (kept in
`tests/data/real_headers/` and tested in CI).

## Data format

| Earlier claim or code | What is true | Handling |
|---|---|---|
| L2 reflectance is int16, divide by 30000 (`step0_read_m3.py`, FINAL notebook text) | **Verified:** float32 (`data type = 4`) | data type read from header; ÷30000 only for integer data |
| Fill value −32768 | **Verified:** `data ignore value = -999.0` | read from header |
| LOC band 0 = latitude | **Verified:** `band names = {Longitude, Latitude, Radius}` | selected by band name |
| LOC is float32 | **Verified:** float64 (`data type = 5`) | data type read from header |
| OBS band 0 = incidence, 1 = emission, 2 = phase (FINAL notebook) | **Verified:** 1 = To-Sun Zenith (incidence), 3 = To-M3 Zenith (emission), 4 = Phase. The wrong indices masked 100 % of pixels | selected by band name |
| Header named `*_loc.img.hdr` | PDS uses several patterns (`*_loc.hdr`, `*_loc_img.hdr`, upper case) | all tried, case-insensitive |

## Wavelengths and band response

| Earlier claim | What is true | Handling |
|---|---|---|
| 85 global bands end at 2267.74 nm; an "86th band" is a bug | **Verified:** the 85 bands run 460.99–2976.20 nm. The hard-coded list ending at 2268 nm put the library on the wrong grid, which caused the spectral mismatch seen in the validation plots | wavelengths always read from the scene header, and identification refuses a library whose wavelengths differ by > 2 nm |
| "83 usable bands, 540–2268 nm" | **Verified:** bbl removes bands 1–2. Excluding thermally affected bands beyond 2500 nm leaves **71 bands, 541–2497 nm** | `good_bands()` = bbl ∩ 540–2500 nm |
| IBD2 upper shoulder moved to 2248 nm "because M3 ends at 2268 nm" | premise false (see above) | IBD2000 continuum uses 1530–1590 and 2450–2500 nm |
| FWHM polynomial "fitted to Green et al. (2011) Table A1", 20.6–45.8 nm | **Verified contradiction:** the header lists 256 native channels of FWHM 12.2–12.8 nm; each global band is the average of 2 or 4 of them (`global channel number`). The polynomial does not describe this | band response built from the header's native channels |
| Single Gaussian with FWHM 20 / 40 nm | an approximation of the above | used only as a fallback when the header lacks the channel table |

## Methods

| Earlier behaviour | Problem | Handling |
|---|---|---|
| LSMA on continuum-removed spectra | linear mixing is only defined for reflectance | LSMA on reflectance with a shade endmember; fractions reported as apparent |
| Endmembers = average of several continuum-removed library spectra | averaging different compositions smears band positions | SAM/SID compare individual spectra; LSMA/CEM use the spectrum feature fitting selected most often |
| Endmember = library spectrum with the highest mean reflectance ("freshest") | brightness depends on grain size and measurement, not freshness | data-driven choice (above) |
| CEM scores min-max rescaled; top 1 % = detections | always flags 1 % of pixels, even when the target is absent | CEM on continuum-removed spectra; detection needs score ≥ 0.5 and robust z ≥ 3 (tested with an absent target) |
| SAM/SID full matrix for the whole scene | ~13 GB for 5.4 M pixels × 600 spectra | processed in chunks |
| SAM/SID/LSMA classes treated as detections | they always return some spectrum, even for featureless pixels | used only as cross-checks of feature fitting (consensus 0–3) |
| "BD1900 = H₂O/OH" | lunar OH/H₂O is identified at ~2.8–3.0 µm (Pieters et al., 2009); 1.9 µm on the Moon is dominated by pyroxene | not produced |
| "R730/R1580 separates olivine from pyroxene" | no support found for this ratio | not produced; IBD1000/IBD2000 and feature fitting are used |
| Photometric masks: emission > 30°, phase > 90° | reasonable options, but with the wrong OBS indices they masked everything | available as `max_emission` / `max_phase` (off by default); incidence > 85° masked by default |
| Savitzky–Golay (7 bands) before all analysis | on M3's 20/40 nm spacing, 7 bands spans 140–280 nm, which can flatten narrow features | applied only to SAM/SID input; feature fitting uses unsmoothed data |

## Found on the first real-data run (lines 3000–3149 of m3g20090607t025544_v01)

| Problem | Evidence | Handling |
|---|---|---|
| 613 of 2,196 library rows were USGS `errorbars_for_*` files | library listing | skipped by the reader; dropped automatically when older libraries are loaded |
| Along-track detector stripes produced false band depths | IBD maps striped; 30 % vs 9 % LCP detections in edge vs other columns | cross-track destriping (METHOD §1) |
| Weakly pyroxene-bearing soil labelled Mg-spinel | 84 pixels; their 1 µm band (0.013) was as deep as in LCP pixels | relative absent-band rule (`max_ratio` 0.25), spinel `min_fit` 0.90 |
| An undetected material's first reference became an LSMA/CEM endmember | a featureless lab glass "explained" every pixel | endmembers only for detected materials, plus a scene-median background endmember |

## References to re-check

Several reference entries in the conversation summaries pair an author and year with the
wrong title or subject. For example, the summaries give Besse et al. (2013, Icarus 222) a
Marius Hills title, and Mustard et al. (2011, JGR 116, E00G12) a "highland crust" title,
but as far as I know those papers are about M3 photometric correction and Aristarchus crater respectively.
Check every reference against the publisher's record before citing it; do not copy the
reference lists from those summaries.
