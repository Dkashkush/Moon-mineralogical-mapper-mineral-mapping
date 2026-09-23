"""Continuum removal (Clark & Roush, 1984).

Two flavours are provided:

* :func:`local_continuum_removed` – a straight-line continuum between the means
  of two continuum intervals on either side of one absorption feature. This is
  what Tetracorder uses for feature fitting, and it is fully vectorised over
  pixels.
* :func:`hull_continuum_removed` – the classic upper convex hull over the whole
  spectrum, used for display and for library inspection.
"""

from __future__ import annotations

import numpy as np


def local_continuum_removed(wavelengths: np.ndarray, spectra: np.ndarray,
                            left: np.ndarray, right: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Divide spectra by a straight-line continuum anchored on two intervals.

    Parameters
    ----------
    wavelengths : (B,) band centres.
    spectra : (..., B) reflectance.
    left, right : integer band indices of the left and right continuum intervals.
    window : integer band indices to return (normally from the first left band
        to the last right band, as in Tetracorder).

    Returns
    -------
    (..., len(window)) continuum-removed reflectance.
    """
    wl = np.asarray(wavelengths, dtype=np.float64)
    x_l, x_r = wl[left].mean(), wl[right].mean()
    y_l = spectra[..., left].mean(axis=-1)
    y_r = spectra[..., right].mean(axis=-1)
    t = (wl[window] - x_l) / (x_r - x_l)
    continuum = y_l[..., None] + (y_r - y_l)[..., None] * t
    with np.errstate(invalid="ignore", divide="ignore"):
        out = spectra[..., window] / continuum
    out[~(continuum > 0)] = np.nan
    return out


def upper_hull(wavelengths: np.ndarray, spectrum: np.ndarray) -> np.ndarray:
    """Upper convex hull of one spectrum, evaluated at every band (monotone chain)."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    y = np.asarray(spectrum, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.full_like(y, np.nan)
    xs, ys = wl[ok], y[ok]
    hull: list[int] = []
    for i in range(len(xs)):
        # Pop while the last two hull points and the new point turn counter-clockwise
        # (i.e. the middle point lies on or below the chord)
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = (xs[b] - xs[a]) * (ys[i] - ys[a]) - (ys[b] - ys[a]) * (xs[i] - xs[a])
            if cross >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    return np.interp(wl, xs[hull], ys[hull])


def hull_continuum_removed(wavelengths: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """Upper-hull continuum removal for one spectrum (B,) or a stack (N, B)."""
    spectra = np.asarray(spectra, dtype=np.float64)
    flat = spectra.reshape(-1, spectra.shape[-1])
    out = np.empty_like(flat)
    for i, s in enumerate(flat):
        cont = upper_hull(wavelengths, s)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = s / cont
    return out.reshape(spectra.shape)
