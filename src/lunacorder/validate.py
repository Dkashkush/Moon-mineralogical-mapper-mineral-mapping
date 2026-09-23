"""Validation tools: how do we know a mineral map is right?

* :func:`locate` / :func:`site_window` – find the part of a scene covering a site.
* :func:`false_alarm_rate` – run the detector on *featureless* versions of real pixels
  (same brightness, continuum and noise, but no absorption bands). Any detection is a
  false alarm, so this measures the false-positive rate of the current thresholds.
* :func:`threshold_sweep` – how the detected fraction changes with fit/depth thresholds.
* :func:`site_summary` – detections within a radius of a ground-truth site, compared
  with what the returned samples say should (and should not) be there.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .continuum import upper_hull
from .identify import IdentificationResult, ResolvedExpert, identify
from .sites import Site

MOON_RADIUS_KM = 1737.4


def distance_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance on the Moon (haversine)."""
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = p2 - p1
    dlon = np.deg2rad((np.asarray(lon2) - np.asarray(lon1) + 180.0) % 360.0 - 180.0)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * MOON_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def locate(scene, lat: float, lon: float, chunk: int = 2000) -> tuple[int, int, float]:
    """(row, col, distance_km) of the scene pixel closest to (lat, lon)."""
    best = (-1, -1, np.inf)
    for r0 in range(0, scene.loc.lines, chunk):
        lo, la = scene.read_lonlat(slice(r0, min(r0 + chunk, scene.loc.lines)))
        d = distance_km(la, lo, lat, lon)
        i = np.nanargmin(d)
        if d.flat[i] < best[2]:
            r, c = np.unravel_index(i, d.shape)
            best = (r0 + int(r), int(c), float(d.flat[i]))
    return best


def site_window(scene, site: Site, half_lines: int = 60) -> tuple[slice, dict]:
    """Row window centred on a site, plus a small description (raises if the site is off-scene)."""
    row, col, dist = locate(scene, site.lat, site.lon)
    lo, la = scene.read_lonlat(slice(row, row + 1))
    pixel_km = float(np.nanmedian(distance_km(la[0, :-1], lo[0, :-1], la[0, 1:], lo[0, 1:])))
    if dist > 3 * max(pixel_km, 0.1):
        raise ValueError(f"{site.name} is not covered by this scene (closest pixel {dist:.1f} km away)")
    rows = slice(max(row - half_lines, 0), min(row + half_lines, scene.rfl.lines))
    return rows, {"site": site.name, "row": row, "col": col, "distance_km": dist, "pixel_km": pixel_km}


def featureless_nulls(cube: np.ndarray, wavelengths: np.ndarray, good: np.ndarray, n: int = 20000,
                      seed: int = 0) -> np.ndarray:
    """Band-free versions of randomly chosen real pixels, with their own noise level.

    Each null is the pixel's upper-hull continuum (its brightness and slope, no bands)
    plus that pixel's high-frequency residual (pixel minus a 5-band Savitzky-Golay fit),
    shuffled across bands so it keeps the noise amplitude but cannot form a band.
    """
    rng = np.random.default_rng(seed)
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    ok = np.flatnonzero(np.all(np.isfinite(flat[:, good]), axis=1))
    pick = rng.choice(ok, size=min(n, ok.size), replace=False)
    wl = np.asarray(wavelengths, float)[good]
    out = np.full((pick.size, flat.shape[1]), np.nan)
    for k, i in enumerate(pick):
        s = flat[i, good]
        cont = upper_hull(wl, s)
        resid = s - savgol_filter(s, 5, 2)
        out[k, good] = cont + rng.permutation(resid)
    return out


def false_alarm_rate(cube: np.ndarray, resolved: ResolvedExpert, good: np.ndarray, n: int = 20000,
                     seed: int = 0) -> dict[str, float]:
    """Fraction of featureless null spectra detected as each material (should be ~0)."""
    nulls = featureless_nulls(cube, resolved.wavelengths, good, n, seed)
    res = identify(nulls[None], resolved)
    return {name: float(res.detected[i].mean()) for i, name in enumerate(res.material_names)}


def threshold_sweep(result: IdentificationResult, material: str,
                    depths=(0.02, 0.03, 0.04, 0.05), fits=(0.80, 0.85, 0.90, 0.95),
                    min_snr: float = 3.0) -> list[dict]:
    """Detected fraction of valid pixels for each (min_depth, min_fit) pair.

    Recomputed from the best-reference fit/depth/sigma, so absent-band rules are not
    re-applied; use it to see how sensitive the map is, not as a final count.
    """
    i = result.material_names.index(material)
    f, d = result.fit[i], result.depth[i]
    snr = result.snr()[i]
    valid = np.isfinite(result.sigma[i])
    rows = []
    for md in depths:
        for mf in fits:
            det = valid & (f >= mf) & (d >= md) & (snr >= min_snr)
            rows.append({"min_depth": md, "min_fit": mf, "percent": 100.0 * det.sum() / max(valid.sum(), 1)})
    return rows


def site_summary(result: IdentificationResult, lon: np.ndarray, lat: np.ndarray, site: Site,
                 radius_km: float = 3.0) -> dict:
    """Detections within ``radius_km`` of a site, checked against the sample expectations."""
    near = distance_km(lat, lon, site.lat, site.lon) <= radius_km
    n = int(near.sum())
    found = {name: 100.0 * float((result.detected[i] & near).sum()) / max(n, 1)
             for i, name in enumerate(result.material_names)}
    checks = []
    for m in site.expect:
        checks.append({"material": m, "expected": "present", "percent": found.get(m, float("nan")),
                       "pass": found.get(m, 0.0) > 0.0})
    for m in site.expect_absent:
        checks.append({"material": m, "expected": "absent", "percent": found.get(m, float("nan")),
                       "pass": found.get(m, 0.0) < 1.0})
    return {"site": site.name, "radius_km": radius_km, "pixels": n, "percent_detected": found,
            "checks": checks}
