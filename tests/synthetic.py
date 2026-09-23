"""Synthetic lab spectra and M3-format scenes for tests.

Absorptions are Gaussians in wavelength multiplied onto a red-sloped continuum,
placed at textbook band centres. They are not meant to be realistic minerals,
only to have known answers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lunacorder.envi import write_envi
from lunacorder.library import Spectrum
from lunacorder.m3 import m3_global_wavelengths

LAB_WL = np.arange(350.0, 2600.0, 1.0)

# (centre nm, sigma nm, depth)
BANDS = {
    "Olivine_SYN1_Fo90": [(850, 70, 0.12), (1050, 110, 0.40), (1250, 150, 0.25)],
    "Enstatite_SYN2": [(920, 90, 0.35), (1900, 200, 0.28)],
    "Augite_SYN3": [(1010, 100, 0.33), (2250, 220, 0.25)],
    "Anorthite_SYN4": [(1250, 150, 0.12)],
    "Spinel_SYN5": [(2000, 230, 0.40)],
}


def absorbed(wl, bands, base=0.25, slope=0.00005):
    cont = base + slope * (wl - 500.0)
    trans = np.ones_like(wl, dtype=float)
    for c, s, d in bands:
        trans *= 1.0 - d * np.exp(-0.5 * ((wl - c) / s) ** 2)
    return cont * trans


def lab_spectra() -> list[Spectrum]:
    return [Spectrum(name, LAB_WL, absorbed(LAB_WL, bands), "synthetic") for name, bands in BANDS.items()]


def write_splib_like(root: Path) -> Path:
    """A tiny folder laid out like ASCIIdata_splib07a."""
    root.mkdir(parents=True, exist_ok=True)
    wl_um = LAB_WL / 1000.0
    (root / "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt").write_text(
        "splib07a Wavelengths BECK\n" + "\n".join(f"{w:.6f}" for w in wl_um) + "\n")
    chap = root / "ChapterM_Minerals"
    chap.mkdir(exist_ok=True)
    for name, bands in BANDS.items():
        refl = absorbed(LAB_WL, bands)
        refl[:5] = -1.23e34  # deleted channels
        (chap / f"splib07a_{name}_BECKb_AREF.txt").write_text(
            f"splib07a Record=1: {name}\n" + "\n".join(f"{r:.7e}" for r in refl) + "\n")
    return root


def scene_truth(rows=60, cols=40):
    """Label image: 0 = featureless, 1..5 = materials in BANDS order, -1 = fill."""
    truth = np.zeros((rows, cols), int)
    names = list(BANDS)
    for k in range(len(names)):
        truth[k * 10:(k + 1) * 10, :] = k + 1
    truth[:, -4:] = -1
    return truth, names


def write_scene(folder: Path, scene_id="m3g20090607t025544_v01", rows=60, cols=40, noise=0.001, seed=0):
    """Write *_rfl, *_loc, *_obs in the same formats as the PDS L2 products (BIL, float32, -999)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    wl = m3_global_wavelengths()
    truth, names = scene_truth(rows, cols)
    cube = np.empty((rows, cols, len(wl)), np.float32)
    for r in range(rows):
        for c in range(cols):
            k = truth[r, c]
            bands = BANDS[names[k - 1]] if k > 0 else []
            base = 0.12 + 0.1 * rng.random()
            cube[r, c] = absorbed(wl, bands, base=base) + rng.normal(0, noise, len(wl))
    cube[truth == -1] = -999.0

    def bil(path, data, header_lines):
        np.ascontiguousarray(np.moveaxis(data.astype("<f4"), -1, 1)).tofile(path)
        rows_, cols_, nb = data.shape
        hdr = ["ENVI", f"samples = {cols_}", f"lines = {rows_}", f"bands = {nb}", "header offset = 0",
               "file type = ENVI Standard", "data type = 4", "interleave = bil", "byte order = 0"]
        return hdr + header_lines

    rfl_hdr = bil(folder / f"{scene_id}_rfl.img", cube, [
        "data ignore value = -999.0",
        "wavelength = {" + ",\n ".join(f"{w:.2f}" for w in wl) + "}",
        "bbl = {" + ", ".join("0" if i < 2 else "1" for i in range(len(wl))) + "}",
    ])
    (folder / f"{scene_id}_rfl.hdr").write_text("\n".join(rfl_hdr) + "\n")

    lat = np.linspace(20.0, 18.0, rows)[:, None] * np.ones((1, cols))
    lon = np.linspace(-20.0, -19.2, cols)[None, :] * np.ones((rows, 1)) + 360.0
    loc = np.dstack([lon, lat, np.full((rows, cols), 1737.4)])
    loc_hdr = bil(folder / f"{scene_id}_loc.img", loc, ["band names = {Longitude, Latitude, Radius}"])
    # PDS quirk: the LOC header is named *_loc_img.hdr
    (folder / f"{scene_id}_loc_img.hdr").write_text("\n".join(loc_hdr) + "\n")

    obs = np.zeros((rows, cols, 10), np.float32)
    obs[..., 1] = 40.0  # incidence
    obs[..., 3] = 5.0  # emission
    obs[..., 4] = 45.0  # phase
    obs[:3, :, 1] = 88.0  # a strip of grazing illumination
    obs_names = ["To-Sun AZM", "To-Sun Zenith", "To-Inst AZM", "To-Inst Zenith", "Phase",
                 "To-Sun Path Length", "To-Inst Path Length", "Facet Slope", "Facet Aspect", "Facet Cos i"]
    up = scene_id.upper()
    obs_hdr = bil(folder / f"{up}_OBS.IMG", obs, ["band names = {" + ", ".join(obs_names) + "}"])
    (folder / f"{up}_OBS.HDR").write_text("\n".join(obs_hdr) + "\n")
    return wl, truth, names


__all__ = ["write_envi", "lab_spectra", "write_splib_like", "write_scene", "scene_truth", "BANDS"]
