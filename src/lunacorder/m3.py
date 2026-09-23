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
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .envi import EnviImage, write_subset

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


def global_band_members(header: dict):
    """Native channels summed into each global-mode band, from an M3 L2 header.

    L2 reflectance headers list every target-mode channel (``target
    wavelengths``, ``target fwhm``, ~12.3 nm) and the global band it was binned
    into (``global channel number``, 2-86 for bands 1-85). Returns a list of
    (centres, fwhms) per band, or None if the header lacks these fields.
    """
    tw = header.get("target wavelengths")
    tf = header.get("target fwhm")
    gc = header.get("global channel number")
    n_bands = int(header.get("bands", 0))
    if tw is None or tf is None or gc is None or not isinstance(gc, np.ndarray):
        return None
    tw, tf, gc = np.asarray(tw, float), np.asarray(tf, float), np.asarray(gc).astype(int)
    if not (len(tw) == len(tf) == len(gc)):
        return None
    first = gc.min()
    members = []
    for band in range(n_bands):
        sel = gc == first + band
        if not sel.any():
            return None
        members.append((tw[sel], tf[sel]))
    return members


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
        upper case (``M3G..._OBS.IMG``). LOC/OBS come from the L1B product, whose
        version number can differ from the L2 reflectance (``..._V03_LOC.IMG`` next
        to ``..._V01_RFL.IMG``), so for those any version of the same scene is
        accepted, highest first.
        """
        folder = Path(folder)
        files = {p.name.lower(): p for p in folder.iterdir()}
        stem = re.sub(r"_v\d+$", "", scene_id.lower())

        def pick(kind: str) -> Path | None:
            exact = files.get(f"{scene_id}_{kind}.img".lower())
            if exact is not None or kind == "rfl":
                return exact
            pattern = re.compile(rf"{re.escape(stem)}_v(\d+)_{kind}\.img")
            versions = [(int(m.group(1)), name) for name in files if (m := pattern.fullmatch(name))]
            return files[max(versions)[1]] if versions else None

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

    @property
    def band_members(self):
        """Exact composite response of each band (see :func:`global_band_members`), or None."""
        members = global_band_members(self.rfl.header)
        if members is None:
            return None
        scale = 1000.0 if max(c.max() for c, _ in members) < 100 else 1.0
        return [(c * scale, f * scale) for c, f in members]

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


def find_scene_ids(folder: str | os.PathLike) -> list[str]:
    """Scene IDs of every ``*_rfl.img`` in ``folder`` (any case), e.g. ``M3G20090607T025544_V01``."""
    return sorted(p.name[: -len("_rfl.img")] for p in Path(folder).iterdir()
                  if p.name.lower().endswith("_rfl.img"))


def subset_scene(scene: M3Scene, outdir: str | os.PathLike, rows: slice, cols: slice = slice(None)) -> Path:
    """Write a small, self-contained copy of part of a scene (rfl, loc, obs + headers).

    Handy for sharing a test area, or for working on a laptop: 150 full-width
    lines of an M3 global-mode scene are ~18 MB instead of ~2.2 GB.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for kind in ("rfl", "loc", "obs"):
        image = getattr(scene, kind)
        if image is not None:
            write_subset(image, outdir / f"{scene.scene_id}_{kind}.img", rows, cols)
    return outdir


def share_parts(scene: M3Scene, outdir: str | os.PathLike, rows: slice, lines_per_part: int = 20) -> list[Path]:
    """Write ``rows`` as several small self-contained subsets (``part_00``, ``part_01``, ...).

    Each 20-line part of a full-width global-mode scene is ~2 MB, small enough to share
    through connectors with tight file-size limits. Rejoin them with :func:`join_parts`.
    """
    outdir = Path(outdir)
    start, stop = rows.start or 0, rows.stop if rows.stop is not None else scene.rfl.lines
    parts = []
    for n, r0 in enumerate(range(start, stop, lines_per_part)):
        parts.append(subset_scene(scene, outdir / f"part_{n:02d}", slice(r0, min(r0 + lines_per_part, stop))))
    return parts


def join_parts(parts: list[str | os.PathLike], outdir: str | os.PathLike, scene_id: str) -> Path:
    """Concatenate subsets written by :func:`share_parts` (in order) back into one scene folder."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scenes = [M3Scene.from_folder(p, scene_id) for p in parts]
    for kind in ("rfl", "loc", "obs"):
        images = [getattr(sc, kind) for sc in scenes]
        if any(im is None for im in images):
            continue
        first = images[0]
        if first.interleave not in ("bil", "bip"):
            raise ValueError("join_parts supports line-interleaved (BIL/BIP) images")
        out_img = outdir / f"{scene_id}_{kind}.img"
        with open(out_img, "wb") as fh:
            for im in images:
                fh.write(im.path.read_bytes())
        total = sum(im.lines for im in images)
        from .envi import find_header

        text = find_header(first.path).read_text(errors="replace")
        text = re.sub(r"(?im)^(\s*lines\s*=\s*)\d+", rf"\g<1>{total}", text)
        out_img.with_suffix(".hdr").write_text(text)
    return outdir


def column_gains(cube: np.ndarray, good: np.ndarray | None = None) -> np.ndarray:
    """Per-column, per-band multiplicative spectral gains (samples, bands) for destriping.

    M3 pushbroom data show along-track stripes: each detector column has a slightly
    different spectral response, which creates false band depths in whole columns (seen in
    the first real-data test as stripes in every IBD map and 3x more detections in the edge
    columns). Each pixel is first divided by its own mean over the good bands, so real albedo
    differences are not removed; only each column's *spectral shape* is compared with the
    scene median shape. Use as many lines as possible (ideally the whole strip).
    """
    cube = np.asarray(cube, dtype=np.float64)
    good = np.ones(cube.shape[-1], bool) if good is None else np.asarray(good, bool)
    import warnings

    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN fill columns / bad bands
        shape = cube / np.nanmean(cube[..., good], axis=-1, keepdims=True)
        col = np.nanmedian(shape, axis=0)  # (samples, bands)
        ref = np.nanmedian(col, axis=0)  # (bands,)
    with np.errstate(invalid="ignore", divide="ignore"):
        gains = col / ref
    gains[~np.isfinite(gains)] = 1.0
    return gains.astype(np.float32)


def scene_column_gains(scene: "M3Scene", good: np.ndarray | None = None, target_lines: int = 1000,
                       cols: slice = slice(None)) -> np.ndarray:
    """Destriping gains from ~``target_lines`` lines spread evenly over the whole strip.

    Every line is used for short scenes; a full 17,868-line strip is sampled every
    ~18th line, which keeps memory low while averaging out local geology.
    """
    step = max(1, scene.rfl.lines // target_lines)
    sample = scene.read_reflectance(slice(None, None, step), cols)
    return column_gains(sample, good)


def destripe(cube: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Divide each column's spectra by its gains (from :func:`column_gains`)."""
    return (cube / gains[None, :, :]).astype(cube.dtype)
