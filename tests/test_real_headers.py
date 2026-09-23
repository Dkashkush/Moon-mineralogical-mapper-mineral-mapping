"""Checks against the real PDS headers of scene m3g20090607t025544_v01.

The .img files are replaced by sparse placeholder files of the correct size,
so layout, data types and band naming are verified without the 2 GB download.
"""

import shutil
from pathlib import Path

import numpy as np
import pytest

from lunacorder.library import Spectrum, convolve
from lunacorder.m3 import M3Scene, fwhm_from_spacing, global_band_members, m3_global_wavelengths

HEADERS = Path(__file__).parent / "data" / "real_headers"
SCENE = "m3g20090607t025544_v01"


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real")
    sizes = {"rfl": 17868 * 304 * 85 * 4, "loc": 17868 * 304 * 3 * 8, "obs": 17868 * 304 * 10 * 4}
    for kind, size in sizes.items():
        shutil.copy(HEADERS / f"{SCENE}_{kind}.hdr", folder)
        with open(folder / f"{SCENE}_{kind}.img", "wb") as fh:  # sparse: uses no disk space
            fh.truncate(size)
    return M3Scene.from_folder(folder, SCENE)


def test_real_rfl_header(scene):
    assert (scene.rfl.lines, scene.rfl.samples, scene.rfl.bands) == (17868, 304, 85)
    assert scene.rfl.interleave == "bil"
    assert scene.rfl.dtype == np.dtype("<f4")
    assert scene.rfl.nodata == -999.0
    wl = scene.wavelengths
    assert wl[0] == pytest.approx(460.99) and wl[-1] == pytest.approx(2976.2)
    good = scene.good_bands()
    assert good.sum() == 71 and not good[:2].any()
    np.testing.assert_allclose(m3_global_wavelengths(), wl, atol=0.2)


def test_real_loc_is_float64_lon_lat(scene):
    assert scene.loc.dtype == np.dtype("<f8")  # LOC is double precision, unlike RFL/OBS
    assert scene.loc.band_names == ["Longitude", "Latitude", "Radius"]
    lon, _ = scene.read_lonlat(slice(0, 2), slice(0, 2))
    assert lon.shape == (2, 2)


def test_real_obs_band_names(scene):
    names = scene.obs.band_names
    assert names[1].startswith("To-Sun Zenith") and names[3].startswith("To-M3 Zenith")
    geom = scene.read_geometry(slice(0, 1), slice(0, 1))
    assert set(geom) == {"incidence", "emission", "phase"}


def test_real_band_binning_table(scene):
    members = global_band_members(scene.rfl.header)
    assert members is not None and len(members) == 85
    counts = [len(c) for c, _ in members]
    assert counts[:7] == [4] * 7 and counts[7:49] == [2] * 42 and counts[49:] == [4] * 36
    centres = np.array([c.mean() for c, _ in members])
    np.testing.assert_allclose(centres, scene.wavelengths, atol=0.01)
    assert all(np.all((f > 12) & (f < 13)) for _, f in members)


def test_exact_and_gaussian_response_agree_on_smooth_spectra(scene):
    """The composite response matters only for sharp features; smooth spectra barely change."""
    wl = scene.wavelengths
    lab = np.arange(350.0, 3000.0, 1.0)
    smooth = Spectrum("smooth", lab, 0.2 + 0.1 * np.exp(-0.5 * ((lab - 1000) / 150) ** 2))
    exact = convolve(smooth, wl, fwhm_from_spacing(wl), members=scene.band_members)
    approx = convolve(smooth, wl, fwhm_from_spacing(wl))
    ok = np.isfinite(exact) & np.isfinite(approx)
    assert np.max(np.abs(exact[ok] - approx[ok])) < 2e-3
