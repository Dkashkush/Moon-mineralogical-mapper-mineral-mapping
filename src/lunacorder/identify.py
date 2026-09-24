"""Tetracorder-style mineral identification (Clark et al., 1990, 2003, 2024).

For every material and every one of its diagnostic features:

1. Remove a straight-line continuum from the observed spectrum *and* from the
   reference spectrum, anchored on the same two continuum intervals.
2. Fit the continuum-removed reference to the observation by least squares,
   constrained to pass through the continuum:  ``(obs - 1) = b * (ref - 1)``.
   The scale ``b`` says how much of the reference band depth is present.
3. Report
   * **fit**   – correlation coefficient between the two feature shapes,
   * **depth** – fitted band depth, ``b * reference depth``,
   * **sigma** – 1-sigma uncertainty of that depth from the fit residuals
     (an addition to Tetracorder, which reports no uncertainty),
   * **fit x depth** – the score Tetracorder maps.

Multi-feature materials combine features weighted by the reference's band
depths (deeper, more diagnostic features count more). A material is detected
in a pixel only if it passes its fit, depth, SNR and "absent feature" rules; in
each group, the detected material with the highest fit x depth wins.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .continuum import local_continuum_removed
from .expert import ExpertSystem, MaterialDef
from .library import SpectralLibrary

# ---------------------------------------------------------------------------
# Resolving the expert system against a library and a sensor
# ---------------------------------------------------------------------------

@dataclass
class _Feature:
    name: str
    left: np.ndarray
    right: np.ndarray
    window: np.ndarray
    ref_cr_minus_1: np.ndarray  # continuum-removed reference minus 1 (<= 0 in the band)
    ref_depth: float
    min_fit: float


@dataclass
class _Absent:
    left: np.ndarray
    right: np.ndarray
    window: np.ndarray
    max_depth: float
    max_ratio: float | None = None


@dataclass
class _Reference:
    material: int
    library_index: int
    name: str
    features: list[_Feature]
    weights: np.ndarray


@dataclass
class ResolvedExpert:
    expert: ExpertSystem
    wavelengths: np.ndarray
    references: list[_Reference]
    absent: dict[int, list[_Absent]]
    skipped: dict[str, str] = field(default_factory=dict)

    def references_for(self, material: int) -> list[_Reference]:
        return [r for r in self.references if r.material == material]

    def summary(self) -> str:
        lines = [f"Expert system '{self.expert.name}' on {len(self.wavelengths)} bands"]
        for i, m in enumerate(self.expert.materials):
            refs = self.references_for(i)
            if refs:
                lines.append(f"  {m.name:<24} {len(refs):3d} reference spectra")
            else:
                lines.append(f"  {m.name:<24}   - skipped: {self.skipped.get(m.name, 'no references')}")
        return "\n".join(lines)


def _band_indices(wl: np.ndarray, usable: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.flatnonzero(usable & (wl >= lo) & (wl <= hi))


def resolve(expert: ExpertSystem, library: SpectralLibrary, wavelengths: np.ndarray,
            good_bands: np.ndarray | None = None, tolerance_nm: float = 2.0) -> ResolvedExpert:
    """Turn wavelength-based rules into band indices and pre-compute reference features.

    ``library`` must already be convolved to ``wavelengths`` (the scene's own
    band centres); a mismatch larger than ``tolerance_nm`` is an error, because
    that exact mistake (library on a different grid than the image) is what
    made earlier versions of this project mis-classify.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    if len(library.wavelengths) != len(wl) or np.max(np.abs(library.wavelengths - wl)) > tolerance_nm:
        raise ValueError(
            "Library wavelengths do not match the scene. Re-build the library with the "
            "scene's own band centres (see `lunacorder build-library --scene-header`)."
        )
    lo, hi = expert.wavelength_range_nm
    usable = (wl >= lo) & (wl <= hi)
    if good_bands is not None:
        usable &= np.asarray(good_bands, bool)

    references: list[_Reference] = []
    absent: dict[int, list[_Absent]] = {}
    skipped: dict[str, str] = {}
    for mi, mat in enumerate(expert.materials):
        absent[mi] = []
        for a in mat.absent:
            left = _band_indices(wl, usable, *a.left)
            right = _band_indices(wl, usable, *a.right)
            if len(left) == 0 or len(right) == 0:
                raise ValueError(f"{mat.name}: absent-feature '{a.name}' has no usable continuum bands")
            window = np.arange(left[0], right[-1] + 1)
            window = window[usable[window]]
            absent[mi].append(_Absent(left, right, window, a.max_depth, a.max_ratio))

        candidates = _match_references(library, mat)
        if not candidates:
            skipped[mat.name] = f"no library spectra match {mat.references}"
            continue
        reasons = []
        for li in candidates:
            ref = library.spectra[li].astype(np.float64)
            ref_usable = usable & np.isfinite(ref)
            feats = []
            for f in mat.features:
                left = _band_indices(wl, ref_usable, *f.left)
                right = _band_indices(wl, ref_usable, *f.right)
                if len(left) == 0 or len(right) == 0:
                    break
                window = np.arange(left[0], right[-1] + 1)
                if not np.all(ref_usable[window]):
                    window = window[ref_usable[window]]
                cr = local_continuum_removed(wl, ref[None, :], left, right, window)[0]
                depth = float(1.0 - np.nanmin(cr))
                if not np.isfinite(depth) or depth < expert.min_reference_depth:
                    break  # this reference does not clearly show the feature
                feats.append(_Feature(f.name, left, right, window, cr - 1.0, depth, f.min_fit))
            if len(feats) != len(mat.features):
                reasons.append(library.names[li])
                continue
            depths = np.array([f.ref_depth for f in feats])
            references.append(_Reference(mi, li, library.names[li], feats, depths / depths.sum()))
        if not any(r.material == mi for r in references):
            skipped[mat.name] = ("matching spectra lack the diagnostic feature(s) (depth < "
                                 f"{expert.min_reference_depth}) or wavelength coverage: "
                                 + ", ".join(reasons[:5]))
    for name, why in skipped.items():
        warnings.warn(f"{name} will not be mapped: {why}", stacklevel=2)
    return ResolvedExpert(expert, wl, references, absent, skipped)


def _match_references(library: SpectralLibrary, mat: MaterialDef) -> list[int]:
    hits: list[int] = []
    for pattern in mat.references:
        hits.extend(i for i in library.find(pattern) if i not in hits)
    for pattern in mat.exclude:
        bad = set(library.find(pattern))
        hits = [i for i in hits if i not in bad]
    return hits


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_feature(wavelengths: np.ndarray, spectra: np.ndarray, feat: _Feature):
    """Fit one reference feature to many spectra.

    Returns (fit, depth, sigma_depth), each shaped (N,). Pixels with missing
    data in the feature get fit = 0 and depth = 0.
    """
    obs = local_continuum_removed(wavelengths, spectra, feat.left, feat.right, feat.window)
    y = obs - 1.0  # (N, W)
    x = feat.ref_cr_minus_1  # (W,)
    n = len(x)
    sxx = float(x @ x)
    sxy = y @ x
    b = sxy / sxx
    # Residual sum of squares of the through-continuum fit
    rss = np.maximum(np.einsum("ij,ij->i", y, y) - sxy * b, 0.0)
    sigma_b = np.sqrt(rss / max(n - 1, 1) / sxx)
    # Shape similarity: Pearson correlation of the two continuum-removed features
    xm = x - x.mean()
    ym = y - y.mean(axis=1, keepdims=True)
    denom = np.sqrt(np.einsum("ij,ij->i", ym, ym) * float(xm @ xm))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ym @ xm) / denom
    fit = np.where(np.isfinite(r) & (b > 0), np.clip(r, 0.0, 1.0), 0.0)
    depth = np.where(np.isfinite(b) & (b > 0), b * feat.ref_depth, 0.0)
    sigma = np.where(np.isfinite(sigma_b), sigma_b * feat.ref_depth, np.inf)
    return fit, depth, sigma


def absent_depth(wavelengths: np.ndarray, spectra: np.ndarray, rule: _Absent) -> np.ndarray:
    """Least-squares depth of a generic absorption inside an 'absent' window.

    The continuum-removed spectrum is fitted with a half-sine band template
    that is 0 at the two continuum centres and 1 midway, i.e. a smooth band
    with no assumed mineral shape. Fitting all bands at once averages the
    noise down (a noisy single-band minimum would reject genuine detections).
    Returns 0 for no band, and inf for pixels with missing data.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    cr = local_continuum_removed(wl, spectra, rule.left, rule.right, rule.window)
    x0, x1 = wl[rule.left].mean(), wl[rule.right].mean()
    t = np.clip(np.sin(np.pi * (wl[rule.window] - x0) / (x1 - x0)), 0.0, None)
    t[(wl[rule.window] < x0) | (wl[rule.window] > x1)] = 0.0
    b = -((cr - 1.0) @ t) / float(t @ t)
    return np.where(np.isfinite(b), np.maximum(b, 0.0), np.inf)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class IdentificationResult:
    """Per-pixel products, all shaped like the input image (rows, cols)."""

    material_names: list[str]
    group_names: list[str]
    fit: np.ndarray  # (M, rows, cols) best-reference fit for each material
    depth: np.ndarray  # (M, rows, cols) fitted band depth
    sigma: np.ndarray  # (M, rows, cols) 1-sigma depth uncertainty
    fit_depth: np.ndarray  # (M, rows, cols) fit x depth where detected, else 0
    best_reference: np.ndarray  # (M, rows, cols) library index of best reference, -1 if none
    group_class: np.ndarray  # (G, rows, cols) winning material index + 1, 0 = nothing detected
    group_score: np.ndarray  # (G, rows, cols) winning fit x depth
    group_margin: np.ndarray  # (G, rows, cols) (best - runner-up) / best; low = ambiguous
    reference_names: list[str] = field(default_factory=list)

    @property
    def detected(self) -> np.ndarray:
        return self.fit_depth > 0

    def snr(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.sigma > 0, self.depth / self.sigma, 0.0)

    def counts(self) -> dict[str, int]:
        return {n: int(self.detected[i].sum()) for i, n in enumerate(self.material_names)}


def identify(cube: np.ndarray, resolved: ResolvedExpert, chunk_pixels: int = 100_000,
             progress: bool = False) -> IdentificationResult:
    """Run identification on a reflectance cube shaped (rows, cols, bands) or (N, bands)."""
    shape2d = cube.shape[:-1]
    flat = cube.reshape(-1, cube.shape[-1])
    n_pix = flat.shape[0]
    expert = resolved.expert
    n_mat = len(expert.materials)
    groups = list(expert.groups)

    fit = np.zeros((n_mat, n_pix), np.float32)
    depth = np.zeros((n_mat, n_pix), np.float32)
    sigma = np.full((n_mat, n_pix), np.inf, np.float32)
    fd = np.zeros((n_mat, n_pix), np.float32)
    best_ref = np.full((n_mat, n_pix), -1, np.int32)

    wl = resolved.wavelengths
    for start in range(0, n_pix, chunk_pixels):
        stop = min(start + chunk_pixels, n_pix)
        block = flat[start:stop].astype(np.float64)
        for mi, mat in enumerate(expert.materials):
            refs = resolved.references_for(mi)
            if not refs:
                continue
            # Absent-feature rules depend only on the pixel, not the reference
            allowed = np.ones(stop - start, bool)
            ratio_rules = []
            for rule in resolved.absent[mi]:
                a_depth = absent_depth(wl, block, rule)
                allowed &= a_depth <= rule.max_depth
                if rule.max_ratio is not None:
                    ratio_rules.append((a_depth, rule.max_ratio))

            best_score = np.full(stop - start, -1.0)
            for ref in refs:
                f_tot = np.zeros(stop - start)
                d_tot = np.zeros(stop - start)
                v_tot = np.zeros(stop - start)
                feature_ok = np.ones(stop - start, bool)
                for w, feat in zip(ref.weights, ref.features):
                    f, d, s = fit_feature(wl, block, feat)
                    f_tot += w * f
                    d_tot += w * d
                    v_tot += (w * s) ** 2
                    feature_ok &= (d > 0) & (f >= feat.min_fit)
                s_tot = np.sqrt(v_tot)
                score = f_tot * d_tot
                with np.errstate(invalid="ignore", divide="ignore"):
                    snr = np.where(s_tot > 0, d_tot / s_tot, np.inf)
                passes = (allowed & feature_ok & (f_tot >= mat.min_fit)
                          & (d_tot >= mat.min_depth) & (snr >= mat.min_snr))
                # Relative absence: the unwanted band must be weak compared with the diagnostic one
                for a_depth, max_ratio in ratio_rules:
                    passes &= a_depth <= max_ratio * d_tot
                # Rank references by score, preferring any that pass all rules
                ranked = np.where(passes, score + 10.0, score)
                better = ranked > best_score
                best_score = np.where(better, ranked, best_score)
                sl = slice(start, stop)
                fit[mi, sl] = np.where(better, f_tot, fit[mi, sl])
                depth[mi, sl] = np.where(better, d_tot, depth[mi, sl])
                sigma[mi, sl] = np.where(better, s_tot, sigma[mi, sl])
                fd[mi, sl] = np.where(better, np.where(passes, score, 0.0), fd[mi, sl])
                best_ref[mi, sl] = np.where(better, ref.library_index, best_ref[mi, sl])
        if progress:
            print(f"  identified {stop:,}/{n_pix:,} pixels", flush=True)

    # Group decisions: highest fit x depth among detected materials
    g_class = np.zeros((len(groups), n_pix), np.int16)
    g_score = np.zeros((len(groups), n_pix), np.float32)
    g_margin = np.zeros((len(groups), n_pix), np.float32)
    for gi, g in enumerate(groups):
        members = expert.materials_in(g)
        if not members:
            continue
        scores = fd[members]
        order = np.argsort(scores, axis=0)
        top = np.take_along_axis(scores, order[-1:], axis=0)[0]
        second = np.take_along_axis(scores, order[-2:-1], axis=0)[0] if len(members) > 1 else 0.0
        winner = np.asarray(members)[order[-1]]
        g_class[gi] = np.where(top > 0, winner + 1, 0)
        g_score[gi] = top
        with np.errstate(invalid="ignore", divide="ignore"):
            g_margin[gi] = np.where(top > 0, (top - second) / top, 0.0)

    def img(a):
        return a.reshape(a.shape[:-1] + shape2d)

    return IdentificationResult(
        material_names=[m.name for m in expert.materials],
        group_names=groups,
        fit=img(fit), depth=img(depth), sigma=img(sigma), fit_depth=img(fd),
        best_reference=img(best_ref),
        group_class=img(g_class), group_score=img(g_score), group_margin=img(g_margin),
    )
