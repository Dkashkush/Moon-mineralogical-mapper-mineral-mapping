import numpy as np
import pytest

from lunacorder import classic
from lunacorder.library import build_library, read_mineral_list, read_relab_folder
from lunacorder.m3 import fwhm_from_spacing, m3_global_wavelengths
from lunacorder.products import unwrap_longitude
from synthetic import BANDS, absorbed, lab_spectra

WL = m3_global_wavelengths()
GOOD = (WL >= 540) & (WL <= 2500)


@pytest.fixture(scope="module")
def lib():
    return build_library(lab_spectra(), WL, fwhm_from_spacing(WL), required_range=(540, 2500))


def test_sam_sid_pick_the_right_spectrum(lib):
    rng = np.random.default_rng(0)
    ref = lib.spectra[:, GOOD].astype(float)
    pix = ref * rng.uniform(0.5, 1.5, (len(ref), 1)) + rng.normal(0, 0.001, ref.shape)  # brightness varies
    cr_lib = classic.hull_removed(WL[GOOD], ref)
    cr_pix = classic.hull_removed(WL[GOOD], pix)
    m = classic.sam_sid(cr_pix, cr_lib, row_material=np.arange(len(ref)))
    np.testing.assert_array_equal(m.sam_index, np.arange(len(ref)))
    np.testing.assert_array_equal(m.agree_material, np.arange(len(ref)) + 1)


def test_sam_sid_chunking_is_exact(lib):
    ref = classic.hull_removed(WL[GOOD], lib.spectra[:, GOOD].astype(float))
    pix = np.repeat(ref, 7, axis=0)
    a = classic.sam_sid(pix, ref, chunk=3)
    b = classic.sam_sid(pix, ref, chunk=10_000)
    np.testing.assert_array_equal(a.sam_index, b.sam_index)
    np.testing.assert_allclose(a.sid_value, b.sid_value, rtol=1e-6, atol=1e-9)
    assert (a.sid_value >= 0).all()


def test_lsma_recovers_known_mixture(lib):
    em = lib.spectra[:3, GOOD].astype(float)
    true = np.array([0.5, 0.3, 0.2])
    pixel = (true @ em) * 0.8  # 20 % shade
    frac, rmse = classic.lsma(pixel[None], em, shade=True)
    np.testing.assert_allclose(frac[0, :3] / frac[0, :3].sum(), true, atol=0.02)
    assert frac[0, 3] == pytest.approx(0.2, abs=0.03)
    assert abs(frac[0].sum() - 1) < 0.01 and rmse[0] < 1e-3


def test_cem_flags_target_and_not_background(lib):
    rng = np.random.default_rng(3)
    wl = WL[GOOD]
    background = np.array([absorbed(wl, BANDS["Enstatite_SYN2"], base=rng.uniform(0.1, 0.25))
                           + rng.normal(0, 0.002, len(wl)) for _ in range(2000)])
    target = np.array([absorbed(wl, BANDS["Spinel_SYN5"], base=rng.uniform(0.1, 0.25))
                       + rng.normal(0, 0.002, len(wl)) for _ in range(20)])
    pix = classic.hull_removed(wl, np.vstack([background, target])) - 1
    d = classic.hull_removed(wl, lib.spectra[lib.find("^Spinel")[0], GOOD][None].astype(float)) - 1
    det = classic.cem_detections(classic.cem(pix, d))[:, 0]
    assert det[-20:].mean() > 0.9
    assert det[:-20].mean() < 0.01


def test_cem_absent_target_gives_few_detections(lib):
    """The old 'top 1 %' rule flags 1 % of pixels even when the target is absent."""
    rng = np.random.default_rng(4)
    wl = WL[GOOD]
    background = np.array([absorbed(wl, BANDS["Enstatite_SYN2"], base=rng.uniform(0.1, 0.25))
                           + rng.normal(0, 0.002, len(wl)) for _ in range(2000)])
    pix = classic.hull_removed(wl, background) - 1
    d = classic.hull_removed(wl, lib.spectra[lib.find("^Spinel")[0], GOOD][None].astype(float)) - 1
    assert classic.cem_detections(classic.cem(pix, d)).mean() < 0.001


def test_relab_names_from_xml(tmp_path):
    folder = tmp_path / "cartorder"
    folder.mkdir()
    rows = "\n".join(f"{w} 0.2 0.001" for w in range(400, 2600, 10))
    (folder / "c1lr01.tab").write_text("220\n" + rows + "\n")
    (folder / "c1lr01.xml").write_text(
        '<Product xmlns:r="urn:relab"><r:Specimen><r:specimen_name>Orange glass 74220</r:specimen_name>'
        "</r:Specimen><title>RELAB spectrum</title></Product>")
    (folder / "c1lr02.tab").write_text("220\n" + rows + "\n")  # no label
    names = sorted(s.name for s in read_relab_folder(folder))
    assert names == ["Orange glass 74220 [c1lr01]", "c1lr02"]


def test_mineral_list_patterns(tmp_path):
    p = tmp_path / "moon_minerals.txt"
    p.write_text("Olivine GDS70  # forsterite\n\nAugite\n")
    pats = read_mineral_list(p)
    import re

    assert re.search(pats[0], "Olivine_GDS70.a_Fo89_165u_BECKb_AREF", re.I)
    assert re.search(pats[1], "Augite_NMNH120049_BECKb_AREF", re.I)


def test_unwrap_longitude_across_dateline():
    lon = np.array([[179.5, -179.5], [179.8, -179.2]])
    out = unwrap_longitude(lon)
    assert out.min() > 179 and out.max() < 181
    same = np.array([[10.0, 11.0]])
    np.testing.assert_array_equal(unwrap_longitude(same), same)
