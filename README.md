# lunacorder

**Tetracorder-style mineral identification for Chandrayaan-1 Moon Mineralogy Mapper (M3) data, in pure Python.**

`lunacorder` turns an M3 Level-2 reflectance scene into mineral maps the way USGS
[Tetracorder](https://github.com/PSI-edu/spectroscopy-tetracorder) does. It
fits the diagnostic absorption features of reference minerals to every pixel and maps
**fit**, **band depth** and **fit × depth** per mineral, plus a best-match map per
group. It adds two things Tetracorder lacks: **per-pixel depth uncertainty** and
**map-projected GeoTIFFs** that open directly in QGIS or ArcGIS.

| | Tetracorder | lunacorder |
|---|---|---|
| Language / install | Fortran + davinci, Linux only | `pip install`, runs anywhere incl. Google Colab |
| Rules (expert system) | command files | readable YAML ([`lunar_m3.yaml`](src/lunacorder/data/lunar_m3.yaml)) |
| Outputs | fit, depth, fit×depth, group maps | same, plus 1σ depth uncertainty, ambiguity margin, GeoTIFF |
| Library | specpr | USGS splib07a ASCII, RELAB `.tab`, any 2-column text |

## Status

Version 0.1, under active development. The engine, M3 I/O and product
writers are tested on synthetic M3-format data (see [Validation](#validation)).
Validation on real M3 scenes is the next milestone. Until that is complete,
treat the maps as candidate detections that need spectral checking.

## What it produces

For a scene `m3g20090607t025544_v01` the `map` command writes:

| File | Content |
|---|---|
| `*_groups.img` / `*_groups_map.tif` | best-matching material per pixel, its fit × depth, and a confidence margin |
| `*_fit_depth.img` / `*_fit_depth_map.tif` | Tetracorder fit × depth image for every material |
| `*_fit.img`, `*_depth.img`, `*_sigma.img` | shape fit, fitted band depth and 1σ depth uncertainty per material |
| `*_parameters.img` / `*_parameters_map.tif` | IBD1000, IBD2000, IBD1250, R1580 |
| `*_mineral_map.png` | quick-look class map |
| `*_spectra.png` | M3 spectra of the strongest detections vs. fitted reference, per feature |
| `*_ibd.png` | standard M3 IBD colour composite |
| `*_summary.csv`, `*_run.json` | detections per mineral; every setting, band and reference used (for your methods section) |

`.img` files stay in the sensor's image geometry. `*_map.tif` files are resampled to an
equirectangular grid in the Moon 2015 geographic CRS (R = 1737.4 km).

## Quick start

```bash
pip install "lunacorder[geo] @ git+https://github.com/Dkashkush/Moon-mineralogical-mapper-mineral-mapping"

# 1. Convolve lab spectra to *your scene's own* band centres
lunacorder build-library \
    --scene-header M3_Project/m3g20090607t025544_v01_rfl.hdr \
    --usgs ASCIIdata_splib07a.zip --relab Relab_lunar_mineral_spectra.zip \
    --out m3_library.npz

# (optional) cut a small, shareable test area: ~18 MB for 150 lines
lunacorder subset --folder M3_Project --scene m3g20090607t025544_v01 --rows 3000:3150 --out test_area/

# 2. Map a window of the scene (rows are along-track lines)
lunacorder map --folder M3_Project --scene m3g20090607t025544_v01 \
    --library m3_library.npz --rows 3000:4500 --out results/
```

To run on Google Colab, open [`notebooks/lunacorder_colab.ipynb`](notebooks/lunacorder_colab.ipynb).
Its first cell clones the code and imports it directly (about 10–20 s) rather than pip-installing it:

```python
!git clone -q --depth 1 https://github.com/Dkashkush/Moon-mineralogical-mapper-mineral-mapping /content/lunacorder
import sys; sys.path.insert(0, '/content/lunacorder/src')
!pip install -q rasterio   # only needed for GeoTIFF output
```

### Data you need

* **M3 L2 scene** from the [PDS Orbital Data Explorer](https://ode.rsl.wustl.edu/moon/):
  `*_rfl.img/.hdr` (reflectance), `*_loc.img/.hdr` (coordinates) and `*_obs.img/.hdr` (geometry).
* **Reference spectra:** [USGS splib07a](https://doi.org/10.5066/F7RR1WDJ) ASCII data, and
  optionally [RELAB](https://sites.brown.edu/relab/) lunar sample spectra.

## How it works

1. **Read** the scene through memory maps. Data type, fill value, interleave, wavelengths,
   bad-band list and LOC/OBS band order all come from the headers, never from assumptions.
2. **Convolve** every lab spectrum to the scene's bands using the exact global-mode
   response. The L2 header records which native ~12 nm channels were summed into each
   band, and the code uses that record.
3. **Identify.** For each material's diagnostic features:
   * remove a local continuum,
   * fit the reference feature by least squares,
   * compute fit (shape correlation), band depth and 1σ uncertainty,
   * check absent-feature rules (for example, olivine must *not* show a 2 µm band),
   * in each group, keep the passing material with the highest fit × depth.
4. **Write** image-geometry stacks, map-projected GeoTIFFs, figures and a JSON run log.

The full method, equations and references are in [`docs/METHOD.md`](docs/METHOD.md).

### Minerals in the built-in lunar expert system

Olivine · low-Ca pyroxene · high-Ca pyroxene · crystalline plagioclase (1.25 µm) ·
Mg-spinel · Fe-bearing volcanic glass (when a glass reference such as RELAB 74220 is in the
library). Ilmenite and other opaque phases have no diagnostic NIR bands and cannot be
identified by feature fitting.

## Validation

Every test runs in CI (`pytest`). Header parsing, data types, band naming and the
channel-binning table are checked against the **real PDS headers** of scene
`m3g20090607t025544_v01` (`tests/data/real_headers`). On a synthetic M3-format scene (85 global-mode bands, BIL
float32, −999 fill, PDS header quirks) built from known absorption bands:

| Gaussian noise (σ, reflectance ≈ 0.12–0.22) | Olivine | LCP | HCP | Plagioclase | Mg-spinel | False positives (featureless) |
|---|---|---|---|---|---|---|
| 0.002 | 100 % | 100 % | 100 % | 100 % | 100 % | 0 % |
| 0.004 | 100 % | 100 % | 100 % | 86 % | 100 % | 0 % |
| 0.008 | 99 % | 100 % | 98 % | 2 % | 96 % | 0 % |

Weak bands such as plagioclase's 1.25 µm feature drop out first as noise increases. This is
the expected behaviour: the tool reports nothing rather than guessing.

Planned validation on real data:
* compare with published M3 detections, e.g. Mg-spinel at Moscoviensis (Pieters et al.,
  2011), olivine around Copernicus and Aristarchus, and PAN in the Orientale rings;
* cross-check against IBD maps and against Tetracorder run on the same scene.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

## License

MIT
