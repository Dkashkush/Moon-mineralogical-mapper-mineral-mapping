import numpy as np
import pytest

from lunacorder.expert import ExpertSystem
from lunacorder.identify import identify, resolve
from lunacorder.library import build_library
from lunacorder.m3 import M3Scene, fwhm_from_spacing, m3_global_wavelengths
from lunacorder.sites import APOLLO, Site
from lunacorder.validate import distance_km, false_alarm_rate, locate, site_summary, site_window, threshold_sweep
from synthetic import BANDS, absorbed, lab_spectra, write_scene

WL = m3_global_wavelengths()
GOOD = (WL >= 540) & (WL <= 2500)


@pytest.fixture(scope="module")
def resolved():
    lib = build_library(lab_spectra(), WL, fwhm_from_spacing(WL), required_range=(540, 2500))
    with pytest.warns(UserWarning):
        return resolve(ExpertSystem.builtin(), lib, WL, GOOD)


def test_distance_one_degree_of_latitude():
    assert distance_km(0, 0, 1, 0) == pytest.approx(2 * np.pi * 1737.4 / 360, rel=1e-6)
    assert distance_km(0, 179.9, 0, -179.9) == pytest.approx(distance_km(0, 0, 0, 0.2))


def test_apollo_sites_are_plausible():
    assert set(APOLLO) == {f"apollo{n}" for n in (11, 12, 14, 15, 16, 17)}
    for s in APOLLO.values():
        assert -10 < s.lat < 30 and -30 < s.lon < 35


def test_locate_and_site_window(tmp_path):
    write_scene(tmp_path)
    scene = M3Scene.from_folder(tmp_path, "m3g20090607t025544_v01")
    lon, lat = scene.read_lonlat()
    target = Site("test", float(lat[25, 10]), float(lon[25, 10]), "synthetic")
    row, col, d = locate(scene, target.lat, target.lon, chunk=7)
    assert (row, col) == (25, 10) and d < 1e-6
    rows, info = site_window(scene, target, half_lines=5)
    assert rows == slice(20, 30) and info["row"] == 25
    with pytest.raises(ValueError, match="not covered"):
        site_window(scene, Site("far", -60, 100, "x"))


def test_false_alarm_rate_is_low_for_featureless_nulls(resolved):
    rng = np.random.default_rng(0)
    pix = np.array([absorbed(WL, BANDS[k], base=rng.uniform(0.1, 0.2)) + rng.normal(0, 0.002, len(WL))
                    for k in BANDS for _ in range(40)])
    rates = false_alarm_rate(pix[None], resolved, GOOD, n=200)
    assert max(rates.values()) < 0.02


def test_threshold_sweep_and_site_summary(resolved):
    pix = np.stack([absorbed(WL, BANDS["Augite_SYN3"], base=0.15), absorbed(WL, [], base=0.15)])
    res = identify(pix[None], resolved)
    sweep = threshold_sweep(res, "High-Ca pyroxene")
    assert sweep[0]["percent"] == pytest.approx(50.0)
    lat = np.array([[20.19, 20.19]])
    lon = np.array([[30.77, 30.772]])
    summ = site_summary(res, lon, lat, APOLLO["apollo17"], radius_km=1)
    checks = {c["material"]: c for c in summ["checks"]}
    assert checks["High-Ca pyroxene"]["pass"] and checks["Mg-spinel"]["pass"]


def test_validate_site_end_to_end(tmp_path):
    import json

    from lunacorder import sites
    from lunacorder.pipeline import validate_site

    write_scene(tmp_path / "M3")
    scene = M3Scene.from_folder(tmp_path / "M3", "m3g20090607t025544_v01")
    lon, lat = scene.read_lonlat()
    # a fake site on the synthetic high-Ca pyroxene rows (20-29)
    sites.APOLLO["synthetic"] = Site("Synthetic HCP", float(lat[25, 10]), float(lon[25, 10]), "test",
                                     expect=("High-Ca pyroxene",), expect_absent=("Mg-spinel",))
    lib = build_library(lab_spectra(), WL, fwhm_from_spacing(WL), required_range=(540, 2500))
    with pytest.warns(UserWarning):
        report = validate_site(tmp_path / "M3", "m3g20090607t025544_v01", lib, "synthetic", tmp_path / "out",
                               half_lines=8, radius_km=0.5, figures=False, ensemble=False)
    assert all(c["pass"] for c in report["near_site"]["checks"])
    assert max(report["false_alarm_rate"].values()) < 0.02
    assert (tmp_path / "out" / "m3g20090607t025544_v01_synthetic_validation.md").exists()
    json.loads((tmp_path / "out" / "m3g20090607t025544_v01_synthetic_validation.json").read_text())
    del sites.APOLLO["synthetic"]
