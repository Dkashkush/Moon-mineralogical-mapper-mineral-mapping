"""Whole-spectrum classifiers used alongside feature fitting: SAM, SID, LSMA, CEM.

These are the methods from the project's earlier notebooks, re-implemented with the
problems found in them fixed:

* **Memory.** SAM/SID are vectorised but processed in pixel chunks. A full M3 strip
  (5.4 M pixels x ~600 library spectra) would need ~13 GB as one matrix.
* **Endmembers.** SAM/SID compare pixels with *individual* library spectra and then
  report the material each spectrum belongs to. Averaging continuum-removed spectra
  of different compositions smears band positions, so it is not done.
* **LSMA** unmixes *reflectance*, not continuum-removed spectra, because linear mixing is
  only defined for reflectance. A photometric "shade" endmember is included (Adams et
  al., 1986). Lunar regolith mixes intimately and non-linearly, so fractions are apparent
  areal fractions, not modal abundances.
* **CEM** runs on continuum-removed band-depth spectra (CR − 1), so it is insensitive
  to brightness; on raw reflectance a darker patch of the target scored only ≈0.4. Scores
  are not min-max rescaled and not thresholded at a fixed top 1 %, which flags 1 % of
  pixels even when the target is absent. Detections need both a score above
  ``min_score`` and a robust z-score above ``min_z``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls
from scipy.signal import savgol_filter

from .continuum import upper_hull

# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def smooth(spectra: np.ndarray, window: int = 7, order: int = 2) -> np.ndarray:
    """Savitzky-Golay smoothing along the band axis (NaN-safe: rows with NaN are left as is).

    Note: M3 global bands are spaced 20 nm below 1549 nm and 40 nm above, so a 7-band
    window spans 140 nm or 280 nm. Use it for SAM/SID only; feature fitting does not
    need it (its least-squares fit already averages noise).
    """
    flat = spectra.reshape(-1, spectra.shape[-1]).astype(np.float64)
    out = flat.copy()
    ok = np.all(np.isfinite(flat), axis=1)
    if ok.any():
        out[ok] = savgol_filter(flat[ok], window, order, axis=1)
    return out.reshape(spectra.shape)


def hull_removed(wavelengths: np.ndarray, spectra: np.ndarray, progress: bool = False) -> np.ndarray:
    """Upper-convex-hull continuum removal for a stack (N, B); rows with NaN become NaN."""
    flat = spectra.reshape(-1, spectra.shape[-1]).astype(np.float64)
    out = np.full_like(flat, np.nan)
    ok = np.all(np.isfinite(flat), axis=1)
    idx = np.flatnonzero(ok)
    for n, i in enumerate(idx):
        cont = upper_hull(wavelengths, flat[i])
        out[i] = flat[i] / cont
        if progress and n and n % 200_000 == 0:
            print(f"  continuum removal {n:,}/{len(idx):,}", flush=True)
    return out.reshape(spectra.shape)


# ---------------------------------------------------------------------------
# SAM + SID
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    sam_index: np.ndarray  # best library row by SAM, -1 if above threshold / invalid
    sam_angle: np.ndarray  # radians
    sid_index: np.ndarray
    sid_value: np.ndarray
    agree_index: np.ndarray  # library row where SAM and SID pick the same spectrum and both pass
    agree_material: np.ndarray  # material index (+1) of agreed row; 0 = none


def sam_sid(pixels: np.ndarray, library: np.ndarray, sam_max_rad: float = 0.10, sid_max: float = 0.04,
            row_material: np.ndarray | None = None, chunk: int = 50_000) -> MatchResult:
    """Spectral Angle Mapper (Kruse et al., 1993) and Spectral Information Divergence
    (Chang, 2000), vectorised in chunks.

    ``pixels`` (N, B) and ``library`` (L, B) must be on the same bands and already
    pre-processed the same way (e.g. both continuum-removed). NaN library bands are
    excluded from every comparison.
    """
    lib = np.asarray(library, np.float64)
    use = np.all(np.isfinite(lib), axis=0)
    lib = lib[:, use]
    lib_n = lib / np.linalg.norm(lib, axis=1, keepdims=True)
    q = np.abs(lib) + 1e-12
    q = q / q.sum(axis=1, keepdims=True)
    log_q = np.log(q)
    h_q = (q * log_q).sum(axis=1)

    n = pixels.shape[0]
    out = {k: np.full(n, -1, np.int32) for k in ("sam_index", "sid_index")}
    sam_angle = np.full(n, np.nan, np.float32)
    sid_value = np.full(n, np.nan, np.float32)
    for s in range(0, n, chunk):
        p = np.asarray(pixels[s:s + chunk], np.float64)[:, use]
        ok = np.all(np.isfinite(p), axis=1)
        if not ok.any():
            continue
        pv = p[ok]
        # SAM
        cos = (pv / np.linalg.norm(pv, axis=1, keepdims=True)) @ lib_n.T
        ang = np.arccos(np.clip(cos, -1.0, 1.0))
        best = ang.argmin(axis=1)
        a = ang[np.arange(len(best)), best]
        # SID = D(p||q) + D(q||p)
        pp = np.abs(pv) + 1e-12
        pp = pp / pp.sum(axis=1, keepdims=True)
        log_p = np.log(pp)
        d_pq = (pp * log_p).sum(axis=1, keepdims=True) - pp @ log_q.T
        d_qp = h_q[None, :] - log_p @ q.T
        sid = np.maximum(d_pq + d_qp, 0.0)  # >= 0 by definition; clamp rounding noise
        bsid = sid.argmin(axis=1)
        v = sid[np.arange(len(bsid)), bsid]

        rows = np.arange(s, min(s + chunk, n))[ok]
        sam_angle[rows] = a
        sid_value[rows] = v
        out["sam_index"][rows] = np.where(a <= sam_max_rad, best, -1)
        out["sid_index"][rows] = np.where(v <= sid_max, bsid, -1)

    agree = (out["sam_index"] >= 0) & (out["sam_index"] == out["sid_index"])
    agree_index = np.where(agree, out["sam_index"], -1)
    agree_material = np.zeros(n, np.int16)
    if row_material is not None:
        rm = np.asarray(row_material)
        agree_material[agree] = rm[agree_index[agree]] + 1
        agree_material[agree & (rm[np.maximum(agree_index, 0)] < 0)] = 0
    return MatchResult(out["sam_index"], sam_angle, out["sid_index"], sid_value, agree_index, agree_material)


# ---------------------------------------------------------------------------
# LSMA
# ---------------------------------------------------------------------------


def lsma(pixels: np.ndarray, endmembers: np.ndarray, shade: bool = True, weight: float = 10.0,
         progress: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Fully constrained linear unmixing (non-negative, sum-to-one) of reflectance spectra.

    The sum-to-one constraint is imposed by appending a heavily weighted row of ones
    to a non-negative least-squares problem (Heinz & Chang, 2001). With ``shade``, a
    zero-reflectance endmember absorbs overall brightness differences (Adams et al., 1986).

    Returns (fractions (N, E[+1]), rmse (N,)). The last fraction is shade when enabled.
    """
    em = np.asarray(endmembers, np.float64)
    use = np.all(np.isfinite(em), axis=0)
    em = em[:, use]
    if shade:
        em = np.vstack([em, np.zeros(em.shape[1])])
    n_em = em.shape[0]
    a = np.vstack([em.T, weight * np.ones((1, n_em))])
    n = pixels.shape[0]
    frac = np.full((n, n_em), np.nan, np.float32)
    rmse = np.full(n, np.nan, np.float32)
    for i in range(n):
        p = np.asarray(pixels[i], np.float64)[use]
        if not np.all(np.isfinite(p)):
            continue
        f, _ = nnls(a, np.append(p, weight))
        frac[i] = f
        rmse[i] = np.sqrt(np.mean((p - em.T @ f) ** 2))
        if progress and i and i % 200_000 == 0:
            print(f"  LSMA {i:,}/{n:,}", flush=True)
    return frac, rmse


# ---------------------------------------------------------------------------
# CEM
# ---------------------------------------------------------------------------


def cem(pixels: np.ndarray, targets: np.ndarray, reg: float = 1e-6, max_background: int = 100_000,
        seed: int = 0) -> np.ndarray:
    """Constrained Energy Minimisation (Harsanyi, 1993) for each target spectrum.

    Uses the sample correlation matrix R of valid pixels (a random subsample of at
    most ``max_background``), Tikhonov-regularised by ``reg * trace(R)/B``. The filter
    w = R⁻¹d / (dᵀR⁻¹d) gives score 1 for a pure target and ≈ 0 for background.
    Returns scores (N, T); invalid pixels are NaN.
    """
    t = np.asarray(targets, np.float64)
    use = np.all(np.isfinite(t), axis=0)
    t = t[:, use]
    x = np.asarray(pixels, np.float64)[:, use]
    ok = np.all(np.isfinite(x), axis=1)
    bg = x[ok]
    if len(bg) > max_background:
        bg = bg[np.random.default_rng(seed).choice(len(bg), max_background, replace=False)]
    r = bg.T @ bg / len(bg)
    r += np.eye(r.shape[0]) * reg * np.trace(r) / r.shape[0]
    r_inv_d = np.linalg.solve(r, t.T)  # (B, T)
    w = r_inv_d / np.einsum("bt,tb->t", r_inv_d, t)[None, :]
    scores = np.full((x.shape[0], t.shape[0]), np.nan, np.float32)
    scores[ok] = x[ok] @ w
    return scores


def robust_z(scores: np.ndarray) -> np.ndarray:
    """(score - median) / (1.4826 * MAD), column-wise, ignoring NaN."""
    med = np.nanmedian(scores, axis=0)
    mad = np.nanmedian(np.abs(scores - med), axis=0) * 1.4826
    return (scores - med) / np.where(mad > 0, mad, np.nan)


def cem_detections(scores: np.ndarray, min_score: float = 0.5, min_z: float = 3.0) -> np.ndarray:
    """Boolean detections: target-like signal strength *and* a statistical outlier."""
    with np.errstate(invalid="ignore"):
        return (scores >= min_score) & (robust_z(scores) >= min_z)
