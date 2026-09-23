# Method

This document describes exactly what `lunacorder` computes, so it can be cited and
reproduced. Every run also writes a `*_run.json` recording the bands, thresholds and
reference spectra that were actually used.

## 1. Input data

M3 Level-2 global-mode reflectance (Green et al., 2011; Clark et al., 2011; Besse et
al., 2013), 85 bands from 461 to 2976 nm. Data type, fill value (`data ignore value`,
−999 in current PDS products), interleave and wavelengths are read from each ENVI header.

* **Bands used.** The header's bad-band list removes bands 1–2. Bands beyond 2500 nm are
  also excluded by default, because residual thermal emission remains after the L2 thermal
  correction (Clark et al., 2011; Li & Milliken, 2016). The default range is 540–2500 nm
  (71 bands).
* **Coordinates.** The LOC file band order is longitude, latitude, radius (M3 Archive SIS),
  stored as float64 (unlike RFL and OBS, which are float32). Bands are selected by their
  header `band names`, with the SIS order as the fallback.
* **Geometry.** The OBS file's to-sun zenith is the incidence angle. Pixels with incidence
  above 85° are masked by default (`--max-incidence`). L2 reflectance is already
  photometrically normalised to i = 30°, e = 0°, g = 30°, so no further photometric
  correction is applied.

### Cross-track destriping

M3 is a pushbroom imager: every image column comes from one detector element, and small
differences in spectral response between elements create along-track stripes. In the first
real-data test (scene m3g20090607t025544_v01, lines 3000–3149), IBD1000 column medians varied
by twice the pixel noise, and 30 % of pixels in the edge columns (270–303) were labelled low-Ca
pyroxene against 9 % elsewhere. Before identification, each pixel is divided by its own mean
reflectance over the good bands, so albedo is preserved. The median of these shapes along each
column is compared with the scene median, and the ratio gives a per-column, per-band gain. By
default the gains come from about 1,000 lines spread over the whole strip, so local geology
averages out. Dividing by these gains reduced the IBD1000 column jitter from 0.121 to 0.018,
well below the per-pixel noise (0.057). Destriping can be disabled (`--destripe none`) or
estimated from the mapped window only.

## 2. Reference library

Laboratory spectra are resampled to the scene's own band centres λᵢ:

R̂(λᵢ) = Σⱼ gᵢ(λⱼ) R(λⱼ) Δλⱼ / Σⱼ gᵢ(λⱼ) Δλⱼ,  with gᵢ(λ) = Σₖ exp[−(λ − λₖ)² / 2σₖ²] over the native channels k of band i, and σₖ = FWHMₖ / 2√(2 ln 2).

* **Band response.** Global-mode bands are averages of 2 or 4 native target-mode channels
  (Green et al., 2011). The L2 header lists every native channel's centre and FWHM
  (`target wavelengths`, `target fwhm`, ≈12.2–12.8 nm) and the global band it belongs to
  (`global channel number`). When those fields are present, each band's response is the
  sum of its channels' Gaussians. This is the exact, flat-topped response, not a single wide
  Gaussian. For scene m3g20090607t025544_v01 the member-channel centres reproduce the
  global band centres to within 0.005 nm. If the table is missing, a single Gaussian with
  FWHM equal to the local band spacing (20 or 40 nm) is used instead. On synthetic features
  the two differ by < 0.1 % of the continuum for broad (σ = 150 nm) bands and by up to ≈1 %
  for narrow (σ = 15–30 nm) bands.
* **Coverage.** A band is left empty unless ≥ 95 % of its response-function weight falls on
  valid lab samples. Spectra must cover ≥ 90 % of the analysis range to be kept.
* **Consistency check.** The library is always built from the scene header. Identification
  refuses to run if library and scene wavelengths differ by more than 2 nm.

## 3. Feature fitting (after Clark et al., 1990, 2003)

Each diagnostic feature *f* is defined by a left and a right continuum interval, both in nm.
Let L and R be the bands inside those intervals. The continuum is the straight line through
(mean λ_L, mean ρ_L) and (mean λ_R, mean ρ_R). The fitted window W runs from the first band
of L to the last band of R.

For both the observed spectrum *o* and the reference *r*, define y = o/c − 1 and x = r/c − 1.
Then:

* **Scale** (least squares through the continuum): b = Σ xy / Σ x²
* **Fit:** F = Pearson correlation of x and y over W. F is set to 0 when b ≤ 0.
* **Band depth:** D = b · D_ref, where D_ref = max(−x) is the reference's own band depth.
* **Uncertainty:** σ_D = D_ref · √[ Σ(y − bx)² / ((n − 1) Σ x²) ].

A material with several features combines them with weights wₖ = D_ref,k / Σ D_ref:

F = Σ wₖFₖ,  D = Σ wₖDₖ,  σ_D = √Σ (wₖσ_D,k)²,  **score = F × D**.

Each reference spectrum of a material is evaluated separately. The material takes the
best-scoring reference that passes all rules, and that reference is recorded in the
`best_reference` output.

### Detection rules

A material is detected when all of the following hold:

* F ≥ `min_fit`, D ≥ `min_depth` and D/σ_D ≥ `min_snr`;
* b > 0 for every feature, and each feature meets its own `min_fit` if one is set;
* every **absent** feature has a fitted depth ≤ `max_depth`, and, where `max_ratio` is set,
  ≤ `max_ratio` × the material's fitted depth. Mature lunar soil has shallow bands
  everywhere, so an absolute limit alone is too permissive. For Mg-spinel the 1 µm band must
  be < 25 % of the 2 µm band depth (Pieters et al., 2011 describe spinel as having essentially
  no 1 µm band). On the real test area, the absolute rule alone gave 84 "spinel" pixels whose
  spectra were weakly pyroxene-bearing soil; the ratio rule removes them.

The absent-feature depth is not the noisy single-band minimum. It is the least-squares
amplitude of a half-sine band template spanning that window's continuum. Fitting all bands
at once suppresses noise: on the synthetic benchmark, the single-band version wrongly
rejected 36–83 % of genuine olivine and spinel pixels at σ = 0.004.

Within each group, the detected material with the largest score wins. The **margin**
(best − second) / best flags ambiguous pixels.

## 4. Spectral parameters

Integrated band depths are computed against straight-line continua, similar in approach to
the M3 IBD products (Mustard et al., 2011):

| Parameter | Left continuum (nm) | Right continuum (nm) | Integrated range (nm) |
|---|---|---|---|
| IBD1000 | 730–750 | 1530–1590 | 790–1310 |
| IBD2000 | 1530–1590 | 2450–2500 | 1660–2500 |
| IBD1250 | 930–1010 | 1520–1660 | 1110–1390 |

## 5. Independent cross-checks (SAM + SID, LSMA, CEM)

Feature fitting is the primary identification. Three whole-spectrum methods from the
project's earlier notebooks run alongside it as independent evidence (`ensemble=True`):

* **SAM + SID.** Pixel and library spectra are continuum-removed with an upper convex hull
  (Clark & Roush, 1984), after optional Savitzky–Golay smoothing of the pixels (7 bands,
  order 2). Each pixel is compared with *every* library spectrum by spectral angle
  (≤ 0.10 rad; Kruse et al., 1993) and spectral information divergence (≤ 0.04; Chang,
  2000). A match counts only where both select the same spectrum, and the material that
  spectrum belongs to is reported. Spectra are never averaged into endmembers, because
  averaging continuum-removed spectra of different compositions smears band positions.
* **LSMA.** Fully constrained (non-negative, sum-to-one) unmixing of *reflectance*, with a
  shade endmember (Adams et al., 1986). The endmembers are the reference spectra that feature
  fitting chose most often. Fits with RMSE > 0.02 are rejected. Linear mixing is not valid in
  continuum-removed space, and lunar regolith mixes intimately, so fractions are apparent
  areal fractions, not modal abundances.
* **CEM** (Harsanyi, 1993) on continuum-removed band-depth spectra (CR − 1), so the score does
  not depend on brightness. It uses a Tikhonov-regularised correlation matrix of the scene. A
  detection needs a score ≥ 0.5 **and** ≥ 3 robust standard deviations (1.4826 × MAD) above the
  scene median. A fixed "top 1 %" rule is not used, because it always flags 1 % of pixels even
  when the target is absent.

SAM, SID and LSMA always assign some library spectrum, even to featureless pixels; they have
no "none" answer. They therefore cannot replace feature fitting. They only confirm it. For
each pixel detected by feature fitting, the **consensus** product (0–3) counts how many of
the three methods name the same material, and `*_crosscheck.csv` gives agreement per material.

## 6. Map projection

Products are resampled by nearest neighbour onto an equirectangular longitude/latitude grid
at the native pixel spacing (or `--resolution-m`). A cell stays empty if no pixel lies
within one cell width. The CRS is the IAU Moon 2015 sphere (R = 1737.4 km).

## 7. Known limitations

* The HCP 2 µm band often extends past 2.5 µm, so its right continuum lies inside the band
  and HCP depths are lower bounds.
* Plagioclase is detectable only as crystalline plagioclase with a 1.25 µm band and minor
  mafic content (PAN). The map shows where that band is present, not plagioclase abundance.
* Opaque minerals (e.g. ilmenite) have no diagnostic absorptions in this range.
* Space weathering weakens bands. Thresholds should be tuned per region and reported.

## References

_Volume and page numbers were compiled from memory. Check each one against the publisher's record before you cite it._

* Adams, J. B. (1974). JGR 79, 4829–4836.
* Besse, S. et al. (2013). Icarus 222, 229–242.
* Cheek, L. C. et al. (2013). JGR Planets 118, 1805–1820.
* Clark, R. N. & Roush, T. L. (1984). JGR 89, 6329–6340.
* Clark, R. N. et al. (1990). JGR 95, 12653–12680.
* Clark, R. N. et al. (2003). JGR 108(E12), 5131.
* Clark, R. N. et al. (2011). JGR 116, E00G16.
* Clark, R. N. et al. (2024). Planetary Science Journal 5, 276.
* Cloutis, E. A. & Gaffey, M. J. (1991). JGR 96, 22809–22826.
* Green, R. O. et al. (2011). JGR 116, E00G19.
* Horgan, B. H. N. et al. (2014). Icarus 234, 132–154.
* Klima, R. L. et al. (2011). JGR 116, E00G06.
* Kokaly, R. F. et al. (2017). USGS Data Series 1035.
* Li, S. & Milliken, R. E. (2016). JGR Planets 121, 2081–2107.
* Mustard, J. F. et al. (2011). JGR 116, E00G12.
* Ohtake, M. et al. (2009). Nature 461, 236–240.
* Pieters, C. M. et al. (2011). JGR 116, E00G08.
* Sunshine, J. M. & Pieters, C. M. (1998). JGR 103, 13675–13688.
