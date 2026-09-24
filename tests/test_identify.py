import json

import numpy as np
import pytest

from lunacorder.expert import ExpertSystem
from lunacorder.identify import identify, resolve
from lunacorder.library import build_library
from lunacorder.m3 import fwhm_from_spacing, m3_global_wavelengths
from synthetic import BANDS, absorbed, lab_spectra, scene_truth, write_scene

WL = m3_global_wavelengths()
GOOD = (WL >= 540) & (WL <= 2500)


@pytest.fixture(scope="module")
def library():
    return build_library(lab_spectra(), WL, fwhm_from_spacing(WL), required_range=(540, 2500))


@pytest.fixture(scope="module")
def resolved(library):
    with pytest.warns(UserWarning, match="Fe-bearing glass"):
        return resolve(ExpertSystem.builtin(), library, WL, GOOD)


def test_builtin_expert_loads():
    ex = ExpertSystem.builtin()
    assert {m.name for m in ex.materials} >= {"Olivine", "Low-Ca pyroxene", "High-Ca pyroxene",
                                              "Plagioclase", "Mg-spinel"}


def test_resolve_skips_materials_without_references(resolved):
    assert "Fe-bearing glass" in resolved.skipped
    assert len(resolved.references) == len(BANDS)


def test_library_on_wrong_grid_is_rejected(library):
    with pytest.raises(ValueError, match="do not match"):
        resolve(ExpertSystem.builtin(), library, WL + 15.0, GOOD)


def test_fitted_depth_recovers_diluted_band(library, resolved):
    """A reference mixed 50:50 with a featureless spectrum should fit with ~half the band depth."""
    ref_i = library.find("^Enstatite")[0]
    ref = library.spectra[ref_i].astype(float)
    flat = absorbed(WL, [], base=0.25)
    mixed = 0.5 * ref + 0.5 * flat
    res = identify(np.stack([ref, mixed])[None], resolved)
    mi = [m.name for m in resolved.expert.materials].index("Low-Ca pyroxene")
    d_pure, d_mix = res.depth[mi, 0]
    assert res.fit[mi, 0, 0] > 0.99
    assert 0.4 < d_mix / d_pure < 0.6


def test_each_pure_material_is_identified(library, resolved):
    expected = {"Olivine_SYN1_Fo90": "Olivine", "Enstatite_SYN2": "Low-Ca pyroxene",
                "Augite_SYN3": "High-Ca pyroxene", "Anorthite_SYN4": "Plagioclase",
                "Spinel_SYN5": "Mg-spinel"}
    names = [m.name for m in resolved.expert.materials]
    cube = library.spectra[None].astype(float)
    res = identify(cube, resolved)
    for j, lib_name in enumerate(library.names):
        winner = res.group_class[0, 0, j]
        assert winner > 0, f"{lib_name} not detected"
        assert names[winner - 1] == expected[lib_name]


def test_featureless_and_missing_pixels_are_unclassified(resolved):
    flat = absorbed(WL, [], base=0.2)
    missing = np.full_like(flat, np.nan)
    res = identify(np.stack([flat, missing])[None], resolved)
    assert (res.group_class == 0).all()
    assert (res.fit_depth == 0).all()


def test_end_to_end_scene(tmp_path, library):
    from lunacorder.pipeline import map_scene

    folder = tmp_path / "M3_Project"
    _, truth, _ = write_scene(folder, noise=0.001)
    library.save(tmp_path / "lib.npz")
    with pytest.warns(UserWarning):
        result, extras = map_scene(folder, "m3g20090607t025544_v01", tmp_path / "lib.npz",
                                   tmp_path / "out", figures=True)

    material_names = result.material_names
    expected = {1: "Olivine", 2: "Low-Ca pyroxene", 3: "High-Ca pyroxene", 4: "Plagioclase", 5: "Mg-spinel"}
    cls = result.group_class[0]
    valid = extras["valid"]
    # The top rows have grazing incidence (88 deg) and must be masked
    assert not valid[:3].any()
    assert not valid[:, -4:].any()  # fill columns
    for k, name in expected.items():
        region = (truth == k) & valid
        predicted = np.array([material_names[c - 1] if c else "none" for c in cls[region]])
        accuracy = (predicted == name).mean()
        assert accuracy > 0.95, f"{name}: {accuracy:.2%}"
    region = (truth == 0) & valid
    if region.any():
        assert (cls[region] == 0).mean() > 0.95

    out = tmp_path / "out"
    for suffix in ["_fit_depth.img", "_groups.img", "_summary.csv", "_mineral_map.png",
                   "_spectra.png", "_groups_map.tif", "_run.json"]:
        assert (out / f"m3g20090607t025544_v01{suffix}").exists(), suffix
    meta = json.loads((out / "m3g20090607t025544_v01_run.json").read_text())
    assert meta["detections"]["Olivine"] > 0

    # Masked pixels are NODATA, not "analysed, nothing found" (class 0)
    from lunacorder.envi import EnviImage
    from lunacorder.products import NODATA

    groups = EnviImage.open(out / "m3g20090607t025544_v01_groups.img").read(mask_nodata=False)
    assert np.all(groups[~valid][:, 0] == NODATA)
    assert np.all(groups[valid][:, 0] != NODATA)

    # SAM+SID on band-depth spectra: featureless pixels get no match, absorbing pixels do
    samsid = extras["ensemble"].samsid_material
    assert (samsid[(truth == 0) & valid] == 0).mean() > 0.95
    assert (samsid[(truth == 2) & valid] == 2).mean() > 0.8  # enstatite -> Low-Ca pyroxene (index 1 + 1)


def test_reference_without_a_clear_band_is_not_used():
    from lunacorder.library import Spectrum

    lab = np.arange(350.0, 2600.0, 1.0)
    weak = Spectrum("Anorthite_WEAK", lab, absorbed(lab, [(1250, 150, 0.004)]), "synthetic")
    lib = build_library([*lab_spectra(), weak], WL, fwhm_from_spacing(WL), required_range=(540, 2500))
    with pytest.warns(UserWarning):
        res = resolve(ExpertSystem.builtin(), lib, WL, GOOD)
    plag = [m.name for m in res.expert.materials].index("Plagioclase")
    assert [r.name for r in res.references_for(plag)] == ["Anorthite_SYN4"]


def test_geotiff_is_georeferenced(tmp_path, library):
    rasterio = pytest.importorskip("rasterio")
    from lunacorder.pipeline import map_scene

    folder = tmp_path / "M3_Project"
    write_scene(folder)
    with pytest.warns(UserWarning):
        map_scene(folder, "m3g20090607t025544_v01", library, tmp_path / "out", figures=False)
    with rasterio.open(tmp_path / "out" / "m3g20090607t025544_v01_groups_map.tif") as ds:
        left, bottom, right, top = ds.bounds
        assert -20.1 < left < -19.9 and -19.3 < right < -19.1
        assert 17.9 < bottom < 18.1 and 19.9 < top < 20.1
        assert "1737400" in ds.crs.to_wkt()


def test_scene_truth_layout():
    truth, names = scene_truth()
    assert set(np.unique(truth)) == {-1, 0, 1, 2, 3, 4, 5}
    assert len(names) == 5


def test_absent_rule_is_robust_to_noise(resolved):
    """Regression: a noisy single-band minimum used to veto real olivine/spinel detections."""
    rng = np.random.default_rng(1)
    names = [m.name for m in resolved.expert.materials]
    for lib_name, material in [("Olivine_SYN1_Fo90", "Olivine"), ("Spinel_SYN5", "Mg-spinel")]:
        pixels = np.array([absorbed(WL, BANDS[lib_name], base=0.17) + rng.normal(0, 0.004, len(WL))
                           for _ in range(200)])
        res = identify(pixels[None], resolved)
        hit = res.group_class[0, 0] == names.index(material) + 1
        assert hit.mean() > 0.95, f"{material}: {hit.mean():.2%}"


def test_weak_pyroxene_is_not_called_spinel(resolved):
    """Regression from the first real M3 run: mature soil with a weak 1 um band and a 2 um band
    passed the absolute 1 um limit (0.03) and was labelled Mg-spinel."""
    names = [m.name for m in resolved.expert.materials]
    weak_px = absorbed(WL, [(930, 90, 0.05), (1980, 220, 0.06)], base=0.13)
    true_spinel = absorbed(WL, BANDS["Spinel_SYN5"], base=0.13)
    res = identify(np.stack([weak_px, true_spinel])[None], resolved)
    spinel = names.index("Mg-spinel") + 1
    assert res.group_class[0, 0, 0] != spinel
    assert res.group_class[0, 0, 1] == spinel
