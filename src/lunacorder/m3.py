"""Chandrayaan-1 Moon Mineralogy Mapper (M3) Level-2 scene access.

An M3 L2 scene on the PDS consists of three co-registered ENVI cubes:

* ``*_rfl.img`` – photometrically and thermally corrected reflectance (85 bands in
  global mode). Data type, fill value and wavelengths are read from the header.
* ``*_loc.img`` – per-pixel selenographic coordinates. Band order is
  **longitude, latitude, radius** (M3 Archive SIS); the header's ``band names``
  are checked so a swapped file is caught rather than silently mirrored.
* ``*_obs.img`` – observation geometry (10 bands: to-sun azimuth/zenith,
  to-M3 azimuth/zenith, phase, path lengths, facet slope/aspect, cos i).

Earlier notebooks for this project got the LOC order and the OBS band indices
wrong (masking 100 % of pixels), so both are resolved by band name here, with
the SIS order as a documented fallback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .envi import EnviImage

# Default spectral range used for identification. Bands 1-2 are flagged bad in
# the L2 bbl, and beyond ~2.5 um residual thermal emission remains in L2 data
# (Clark et al., 2011; Li & Milliken, 2016), so it is excluded by default.
DEFAULT_RANGE_NM = (540.0, 2500.0)

# M3 Archive SIS order for the OBS file, used only if band names are missing.
OBS_SIS_ORDER = {
    "sun_azimuth": 0,
    "incidence": 1,
    "sensor_azimuth": 2,
    "emission": 3,
    "phase": 4,
    "sun_path": 5,
    "sensor_path": 6,
    "facet_slope": 7,
    "facet_aspect": 8,
    "facet_cos_i": 9,
}


def m3_global_wavelengths() -> np.ndarray:
    """Approximate M3 global-mode band centres (nm), 85 bands.

    Only used for tests and when a header lacks a ``wavelength`` field; the
    scene header is always preferred.
    """
    vis = 460.99 + 39.93 * np.arange(7)  # 461-701 nm, 40 nm spacing
    nir1 = 730.48 + 19.96 * np.arange(42)  # 730-1549 nm, 20 nm spacing
    nir2 = 1578.9 + 39.92 * np.arange(36)  # 1579-2976 nm, 40 nm spacing
    return np.concatenate([vis, nir1, nir2])


def fwhm_from_spacing(wavelengths: np.ndarray) -> np.ndarray:
    """Estimate each band's FWHM as its local band spacing.

    M3 global mode is made by summing adjacent native channels, so the
    effective bandpass of a binned channel is approximately its sample
    spacing (20 nm or 40 nm). This reproduces the Green et al. (2011) binning
    without hard-coding band numbers, and works for any evenly-binned sensor.
    """
    wl = np.asarray(wavelengths, dtype=float)
    gaps = np.diff(wl)
    left = np.concatenate([[gaps[0]], gaps])
    right = np.concatenate([gaps, [gaps[-1]]])
    # Use the smaller neighbour gap at transitions between binning regimes
    return np.minimum(left, right)


def _band_index(names: list[str], keywords: list[str], fallback: int) -> int:
    lowered = [n.lower() for n in names]
    for i, name in enumerate(lowered):
        if all(k in name for k in keywords):
            return i
    return fallback


@dataclass
class M3Scene:
    """Paths and lazy readers for one M3 L2 scene."""

    rfl: EnviImage
    loc: EnviImage | None = None
    obs: EnviImage | None = None
    scene_id: str = ""

    @classmethod
    def from_folder(cls, folder: str | os.PathLike, scene_id: str) -> "M3Scene":
        """Open ``<scene_id>_rfl.img`` (+ ``_loc``/``_obs`` if present) in ``folder``.

        Filename matching is case-insensitive because PDS OBS files are often
        upper case (``M3G..._OBS.IMG``).
        """
        folder = Path(folder)
        files = {p.name.lower(): p for p in folder.iterdir()}

        def pick(kind: str) -> Path | None:
            return files.get(f"{scene_id}_{kind}.img".lower())

        rfl_path = pick("rfl")
        if rfl_path is None:
            raise FileNotFoundError(f"{scene_id}_rfl.img not found in {folder}")
        loc_path, obs_path = pick("loc"), pick("obs")
        return cls(
            rfl=EnviImage.open(rfl_path),
            loc=EnviImage.open(loc_path) if loc_path else None,
            obs=EnviImage.open(obs_path) if obs_path else None,
            scene_id=scene_id,
        )

    # -- spectral metadata ----------------------------------------------
    @property
    def wavelengths(self) -> np.ndarray:
        wl = self.rfl.wavelengths
        if wl is None:
            if self.rfl.bands != 85:
                raise ValueError("Reflectance header has no wavelengths and is not 85-band global mode")
            wl = m3_global_wavelengths()
        if wl.max() < 100:  # micrometres -> nanometres
            wl = wl * 1000.0
        return wl

    @property
    def fwhm(self) -> np.ndarray:
        fw = self.rfl.fwhm
        if fw is not None and len(fw) == self.rfl.bands:
            return fw * 1000.0 if fw.max() < 1 else fw
        return fwhm_from_spacing(self.wavelengths)

    def good_bands(self, wl_range: tuple[float, float] = DEFAULT_RANGE_NM) -> np.ndarray:
        """Boolean mask of bands inside ``wl_range`` and not flagged in the header bbl."""
        wl = self.wavelengths
        return self.rfl.bbl & (wl >= wl_range[0]) & (wl <= wl_range[1])

    # -- pixel data -----------------------------------------------------
    def read_reflectance(self, rows: slice = slice(None), cols: slice = slice(None)) -> np.ndarray:
        cube = self.rfl.read(rows, cols)
        # Reflectance cannot be <= 0; such values are fill or artefacts.
        cube[cube <= 0] = np.nan
        # Old integer PDS versions stored reflectance scaled by 30000.
        if np.issubdtype(self.rfl.dtype, np.integer):
            cube /= 30000.0
        return cube

    def read_lonlat(self, rows: slice = slice(None), cols: slice = slice(None),
                    lon_180: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Return (longitude, latitude) in degrees for the window."""
        if self.loc is None:
            raise FileNotFoundError("No LOC file for this scene")
        names = self.loc.band_names
        i_lon = _band_index(names, ["lon"], 0)
        i_lat = _band_index(names, ["lat"], 1)
        block = self.loc.read(rows, cols, mask_nodata=False)
        lon, lat = block[:, :, i_lon].astype(np.float64), block[:, :, i_lat].astype(np.float64)
        if np.nanmax(np.abs(lat)) > 90:  # a mislabelled file: swap rather than mirror the map
            lon, lat = lat, lon
        if lon_180:
            lon = np.where(lon > 180, lon - 360, lon)
        return lon, lat

    def read_geometry(self, rows: slice = slice(None), cols: slice = slice(None)) -> dict:
        """Return incidence, emission and phase angles (degrees) for the window."""
        if self.obs is None:
            raise FileNotFoundError("No OBS file for this scene")
        names = self.obs.band_names
        idx = {
            "incidence": _band_index(names, ["sun", "zen"], OBS_SIS_ORDER["incidence"]),
            "emission": _band_index(names, ["zen"], OBS_SIS_ORDER["emission"]),
            "phase": _band_index(names, ["phase"], OBS_SIS_ORDER["phase"]),
        }
        # "zen" alone also matches the to-sun zenith; prefer a sensor-zenith band
        for i, n in enumerate(names):
            n = n.lower()
            if "zen" in n and "sun" not in n:
                idx["emission"] = i
                break
        block = self.obs.read(rows, cols, mask_nodata=False)
        return {k: block[:, :, i] for k, i in idx.items()}
