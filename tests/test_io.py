from pathlib import Path

import numpy as np
import pytest

from lunacorder.envi import EnviImage, find_header, parse_header
from lunacorder.library import SpectralLibrary, build_library, convolve, read_relab_tab, read_usgs_splib07
from lunacorder.m3 import M3Scene, fwhm_from_spacing, m3_global_wavelengths
from synthetic import BANDS, lab_spectra, write_scene, write_splib_like


def test_parse_multiline_header(tmp_path):
    hdr = tmp_path / "x.hdr"
    hdr.write_text("ENVI\nsamples = 3\nlines = 2\nbands = 4\ndata type = 4\ninterleave = bil\n"
                   "wavelength = {500.0, 600.0,\n 700.0, 800.0}\nband names = {A, B C, D, E}\n")
    h = parse_header(hdr)
    assert h["samples"] == "3"
    np.testing.assert_allclose(h["wavelength"], [500, 600, 700, 800])
    assert h["band names"] == ["A", "B C", "D", "E"]


def test_scene_reading_matches_what_was_written(tmp_path):
    wl, _, _ = write_scene(tmp_path)
    scene = M3Scene.from_folder(tmp_path, "m3g20090607t025544_v01")
    assert scene.obs is not None and scene.loc is not None  # upper-case OBS and *_loc_img.hdr found
    assert scene.rfl.interleave == "bil"
    np.testing.assert_allclose(scene.wavelengths, wl, atol=0.01)

    cube = scene.read_reflectance(slice(10, 20), slice(0, 40))
    assert cube.shape == (10, 40, 85)
    assert np.isnan(cube[:, -1]).all()  # -999 fill -> NaN
    assert np.nanmin(cube) > 0

    good = scene.good_bands()
    assert not good[:2].any()  # bbl bands 1-2
    assert scene.wavelengths[good].max() <= 2500


def test_loc_band_order_is_lon_lat(tmp_path):
    write_scene(tmp_path)
    scene = M3Scene.from_folder(tmp_path, "m3g20090607t025544_v01")
    lon, lat = scene.read_lonlat()
    assert 18.0 <= lat.min() and lat.max() <= 20.0
    assert -20.0 <= lon.min() and lon.max() <= -19.2  # 340 E converted to -20


def test_obs_geometry_by_band_name(tmp_path):
    write_scene(tmp_path)
    scene = M3Scene.from_folder(tmp_path, "m3g20090607t025544_v01")
    g = scene.read_geometry()
    assert g["incidence"][10, 0] == pytest.approx(40.0)
    assert g["emission"][10, 0] == pytest.approx(5.0)
    assert g["phase"][10, 0] == pytest.approx(45.0)


def test_find_header_variants(tmp_path):
    (tmp_path / "a_loc.img").write_bytes(b"")
    (tmp_path / "a_loc_img.hdr").write_text("ENVI\n")
    assert find_header(tmp_path / "a_loc.img").name == "a_loc_img.hdr"


def test_envi_size_mismatch_is_reported(tmp_path):
    (tmp_path / "bad.img").write_bytes(b"\x00" * 10)
    (tmp_path / "bad.hdr").write_text("ENVI\nsamples = 2\nlines = 2\nbands = 2\ndata type = 4\ninterleave = bsq\n")
    with pytest.raises(ValueError, match="bytes"):
        EnviImage.open(tmp_path / "bad.img").read()


def test_fwhm_from_spacing_reproduces_m3_binning():
    fw = fwhm_from_spacing(m3_global_wavelengths())
    assert fw[10] == pytest.approx(19.96, abs=0.1)
    assert fw[60] == pytest.approx(39.92, abs=0.1)


def test_convolution_preserves_flat_and_linear_spectra():
    wl = m3_global_wavelengths()
    fw = fwhm_from_spacing(wl)
    lab = np.arange(300.0, 3100.0, 1.0)
    from lunacorder.library import Spectrum

    flat = convolve(Spectrum("flat", lab, np.full_like(lab, 0.3)), wl, fw)
    np.testing.assert_allclose(flat, 0.3, atol=1e-9)
    line = convolve(Spectrum("line", lab, 0.1 + 1e-4 * lab), wl, fw)
    ok = np.isfinite(line)
    np.testing.assert_allclose(line[ok], 0.1 + 1e-4 * wl[ok], atol=1e-5)


def test_convolution_leaves_uncovered_bands_empty():
    wl = m3_global_wavelengths()
    from lunacorder.library import Spectrum

    asd_like = Spectrum("asd", np.arange(350.0, 2501.0), np.full(2151, 0.2))
    out = convolve(asd_like, wl, fwhm_from_spacing(wl))
    assert np.isnan(out[wl > 2500]).all()
    assert np.isfinite(out[(wl > 540) & (wl < 2450)]).all()


def test_read_usgs_splib07_layout(tmp_path):
    root = write_splib_like(tmp_path / "ASCIIdata_splib07a")
    spectra = read_usgs_splib07(root)
    assert sorted(s.name for s in spectra) == sorted(f"{n}_BECKb_AREF" for n in BANDS)
    s = spectra[0]
    assert s.wavelengths.min() == pytest.approx(350.0)  # micrometres converted to nm
    assert np.isnan(s.reflectance[:5]).all()  # -1.23e34 deleted channels
    only_olivine = read_usgs_splib07(root, include=["^Olivine"])
    assert len(only_olivine) == 1


def test_read_relab_tab_with_inconsistent_last_row(tmp_path):
    p = tmp_path / "c1lx01.tab"
    rows = "\n".join(f"{w} {0.2:.3f} 0.001" for w in range(400, 2600, 5))
    p.write_text("441\n" + rows + "\n2600 --\n")
    s = read_relab_tab(p)
    assert len(s.wavelengths) == len(s.reflectance) == 440


def test_library_roundtrip(tmp_path):
    wl = m3_global_wavelengths()
    lib = build_library(lab_spectra(), wl, fwhm_from_spacing(wl), required_range=(540, 2500))
    assert len(lib) == len(BANDS)
    lib.save(tmp_path / "lib.npz")
    back = SpectralLibrary.load(tmp_path / "lib.npz")
    assert back.names == lib.names
    np.testing.assert_array_equal(back.spectra, lib.spectra)
    lib.to_envi_sli(tmp_path / "lib.sli")
    assert (tmp_path / "lib.hdr").exists()


def test_subset_scene_roundtrip(tmp_path):
    from lunacorder.m3 import subset_scene

    write_scene(tmp_path / "full")
    scene = M3Scene.from_folder(tmp_path / "full", "m3g20090607t025544_v01")
    subset_scene(scene, tmp_path / "sub", slice(10, 25), slice(5, 30))
    sub = M3Scene.from_folder(tmp_path / "sub", "m3g20090607t025544_v01")
    assert (sub.rfl.lines, sub.rfl.samples) == (15, 25)
    np.testing.assert_array_equal(sub.read_reflectance(), scene.read_reflectance(slice(10, 25), slice(5, 30)))
    np.testing.assert_array_equal(sub.read_lonlat()[1], scene.read_lonlat(slice(10, 25), slice(5, 30))[1])
    np.testing.assert_allclose(sub.wavelengths, scene.wavelengths)
    assert sub.obs is not None


def test_splib_zip_is_extracted_to_cache_not_next_to_zip(tmp_path, monkeypatch):
    import shutil

    root = write_splib_like(tmp_path / "src" / "ASCIIdata_splib07a")
    zip_path = Path(shutil.make_archive(str(tmp_path / "drive" / "ASCIIdata_splib07a"), "zip", root))
    monkeypatch.setenv("LUNACORDER_CACHE", str(tmp_path / "cache"))
    spectra = read_usgs_splib07(zip_path)
    assert len(spectra) == len(BANDS)  # errorbar files are skipped
    assert sorted(p.name for p in zip_path.parent.iterdir()) == ["ASCIIdata_splib07a.zip"]


def test_library_without_errorbars():
    wl = m3_global_wavelengths()
    lib = build_library(lab_spectra(), wl, fwhm_from_spacing(wl), required_range=(540, 2500))
    lib.names[0] = "errorbars_for_splib07a_X_BECKb_AREF"
    assert len(lib.without_errorbars()) == len(lib) - 1
