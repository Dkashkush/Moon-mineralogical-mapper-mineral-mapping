"""Library-free spectral parameters that complement identification maps.

These follow the widely used M3 "IBD" colour-composite approach (e.g. Mustard
et al., 2011; Isaacson et al., 2011): integrated band depths near 1 and 2 um
plus 1.58 um reflectance (albedo). Each is computed against a straight-line
continuum between the stated shoulders, and the exact wavelengths used are
stored with the product so they can be reported in a methods section.
"""

from __future__ import annotations

import numpy as np

from .continuum import local_continuum_removed

# name: (left shoulder nm, right shoulder nm, integrate from nm, integrate to nm)
PARAMETERS = {
    "IBD1000": ((730, 750), (1530, 1590), 790, 1310),
    "IBD2000": ((1530, 1590), (2450, 2500), 1660, 2500),
    "IBD1250": ((930, 1010), (1520, 1660), 1110, 1390),
}


def integrated_band_depth(wavelengths: np.ndarray, cube: np.ndarray, left: tuple, right: tuple,
                          start: float, stop: float, good_bands: np.ndarray | None = None) -> np.ndarray:
    """Sum of (1 - R/Rc) over bands in [start, stop] with a linear continuum."""
    wl = np.asarray(wavelengths, float)
    ok = np.ones(len(wl), bool) if good_bands is None else np.asarray(good_bands, bool)
    li = np.flatnonzero(ok & (wl >= left[0]) & (wl <= left[1]))
    ri = np.flatnonzero(ok & (wl >= right[0]) & (wl <= right[1]))
    wi = np.flatnonzero(ok & (wl >= start) & (wl <= stop))
    if len(li) == 0 or len(ri) == 0 or len(wi) == 0:
        raise ValueError(f"No bands for IBD window {left}-{right}")
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    cr = local_continuum_removed(wl, flat, li, ri, wi)
    return np.sum(1.0 - cr, axis=1).reshape(cube.shape[:-1]).astype(np.float32)


def band_parameters(wavelengths: np.ndarray, cube: np.ndarray,
                    good_bands: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """IBD1000, IBD2000, IBD1250 and R1580 for a (rows, cols, bands) cube."""
    out = {name: integrated_band_depth(wavelengths, cube, *spec, good_bands=good_bands)
           for name, spec in PARAMETERS.items()}
    wl = np.asarray(wavelengths, float)
    out["R1580"] = cube[..., int(np.argmin(np.abs(wl - 1580)))].astype(np.float32)
    return out


def stretch(band: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Percentile stretch to 0-1 for display."""
    finite = band[np.isfinite(band)]
    if finite.size == 0:
        return np.zeros_like(band)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((band - lo) / max(hi - lo, 1e-12), 0, 1)


def ibd_composite(params: dict[str, np.ndarray]) -> np.ndarray:
    """Standard M3 RGB: R = IBD1000, G = IBD2000, B = R1580."""
    rgb = np.dstack([stretch(params["IBD1000"]), stretch(params["IBD2000"]), stretch(params["R1580"])])
    rgb[~np.isfinite(rgb)] = 0
    return rgb
