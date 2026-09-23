"""Writing identification products: image stacks, map-projected GeoTIFFs, tables, figures.

Products mirror Tetracorder's outputs (per-material fit, depth and fit x depth
images plus per-group "best material" maps) and add uncertainty, confidence
margin and map-projected versions that open directly in QGIS/ArcGIS.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .envi import write_envi
from .identify import IdentificationResult

NODATA = -9999.0

# Moon 2015 (IAU) geographic CRS: sphere of radius 1737.4 km
MOON_GEOGCS_WKT = (
    'GEOGCS["Moon (2015) - Sphere / Ocentric",DATUM["Moon (2015) - Sphere",'
    'SPHEROID["Moon (2015) - Sphere",1737400,0]],PRIMEM["Reference Meridian",0],'
    'UNIT["degree",0.0174532925199433]]'
)


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).copy()
    a[~np.isfinite(a)] = NODATA
    return a


def write_image_products(result: IdentificationResult, outdir: str | Path, prefix: str,
                         params: dict[str, np.ndarray] | None = None) -> list[Path]:
    """Write ENVI stacks in the scene's own (unprojected) image geometry."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = result.material_names
    written = []
    for kind, stack in [("fit_depth", result.fit_depth), ("fit", result.fit),
                        ("depth", result.depth), ("sigma", result.sigma)]:
        written.append(write_envi(outdir / f"{prefix}_{kind}.img", _clean(np.moveaxis(stack, 0, -1)),
                                  band_names=names, nodata=NODATA))
    group_bands, group_names = [], []
    for gi, g in enumerate(result.group_names):
        group_bands += [result.group_class[gi].astype(np.float32), result.group_score[gi], result.group_margin[gi]]
        group_names += [f"{g} class", f"{g} fit x depth", f"{g} margin"]
    written.append(write_envi(outdir / f"{prefix}_groups.img", _clean(np.dstack(group_bands)),
                              band_names=group_names, nodata=NODATA,
                              extra={"class names": "{none, " + ", ".join(names) + "}"}))
    if params:
        written.append(write_envi(outdir / f"{prefix}_parameters.img",
                                  _clean(np.dstack(list(params.values()))),
                                  band_names=list(params), nodata=NODATA))
    return written


def write_summary(result: IdentificationResult, path: str | Path, valid: np.ndarray | None = None) -> Path:
    """CSV with detections per material and their mean fit, depth and SNR."""
    path = Path(path)
    n_valid = int(valid.sum()) if valid is not None else int(np.prod(result.fit.shape[1:]))
    snr = result.snr()
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["material", "pixels_detected", "percent_of_valid", "pixels_as_group_winner",
                    "mean_fit", "mean_depth", "median_snr"])
        for i, name in enumerate(result.material_names):
            det = result.detected[i]
            wins = int(sum((gc == i + 1).sum() for gc in result.group_class))
            n = int(det.sum())
            w.writerow([
                name, n, f"{100.0 * n / max(n_valid, 1):.3f}", wins,
                f"{result.fit[i][det].mean():.3f}" if n else "",
                f"{result.depth[i][det].mean():.4f}" if n else "",
                f"{np.median(snr[i][det]):.1f}" if n else "",
            ])
    return path


# ---------------------------------------------------------------------------
# Map projection (nearest-neighbour to an equirectangular grid)
# ---------------------------------------------------------------------------

def grid_to_equirectangular(lon: np.ndarray, lat: np.ndarray, bands: np.ndarray,
                            resolution_m: float | None = None, radius_m: float = 1737400.0):
    """Resample image-geometry bands onto a regular lon/lat grid.

    M3 L2 data are delivered in sensor geometry with per-pixel coordinates in
    the LOC file. Each output cell takes the nearest input pixel within one
    cell width; cells with no nearby pixel are NODATA.

    Returns (grid (bands, rows, cols), transform tuple, lon_res_deg, lat_res_deg).
    """
    from scipy.spatial import cKDTree

    bands = np.asarray(bands)
    if bands.ndim == 2:
        bands = bands[None]
    ok = np.isfinite(lon) & np.isfinite(lat)
    lat0 = np.deg2rad(np.nanmean(lat[ok]))
    m_per_deg = np.deg2rad(1.0) * radius_m
    if resolution_m is None:
        # Native spacing: median distance between along-track neighbours
        dy = np.abs(np.diff(lat, axis=0))
        dx = np.abs(np.diff(lon, axis=1)) * np.cos(lat0)
        resolution_m = float(np.nanmedian(np.concatenate([dy[np.isfinite(dy)], dx[np.isfinite(dx)]])) * m_per_deg)
    lat_res = resolution_m / m_per_deg
    lon_res = lat_res / max(np.cos(lat0), 1e-6)

    x, y = lon[ok], lat[ok]
    lon_min, lon_max, lat_min, lat_max = x.min(), x.max(), y.min(), y.max()
    ncols = int(np.ceil((lon_max - lon_min) / lon_res)) + 1
    nrows = int(np.ceil((lat_max - lat_min) / lat_res)) + 1
    gx = lon_min + (np.arange(ncols) + 0.5) * lon_res
    gy = lat_max - (np.arange(nrows) + 0.5) * lat_res

    tree = cKDTree(np.column_stack([x * np.cos(lat0), y]))
    out = np.full((bands.shape[0], nrows, ncols), NODATA, np.float32)
    flat_bands = bands.reshape(bands.shape[0], -1)[:, ok.ravel()]
    for r0 in range(0, nrows, 512):  # row blocks keep memory bounded
        rows = gy[r0:r0 + 512]
        qx, qy = np.meshgrid(gx * np.cos(lat0), rows)
        dist, idx = tree.query(np.column_stack([qx.ravel(), qy.ravel()]), distance_upper_bound=lat_res)
        hit = np.isfinite(dist)
        block = np.full((bands.shape[0], qx.size), NODATA, np.float32)
        block[:, hit] = flat_bands[:, idx[hit]]
        out[:, r0:r0 + len(rows)] = block.reshape(bands.shape[0], len(rows), ncols)
    transform = (lon_min, lon_res, 0.0, lat_max, 0.0, -lat_res)  # GDAL geotransform
    return out, transform, lon_res, lat_res


def write_geotiff(path: str | Path, grid: np.ndarray, transform: tuple, band_names: list[str]) -> Path:
    """Write a gridded stack as a GeoTIFF in the Moon 2015 geographic CRS (needs rasterio)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    path = Path(path)
    a, b, c, d, e, f = transform
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.shape[1], width=grid.shape[2], count=grid.shape[0],
        dtype="float32", crs=CRS.from_wkt(MOON_GEOGCS_WKT), transform=Affine(b, c, a, e, f, d),
        nodata=NODATA, compress="deflate",
    ) as dst:
        dst.write(grid.astype(np.float32))
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)
    return path


def write_map_products(result: IdentificationResult, lon: np.ndarray, lat: np.ndarray,
                       outdir: str | Path, prefix: str, params: dict[str, np.ndarray] | None = None,
                       resolution_m: float | None = None) -> list[Path]:
    """Map-project the key products to GeoTIFF."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stacks = {
        "groups": (np.concatenate([result.group_class.astype(np.float32), result.group_score,
                                   result.group_margin]),
                   [f"{g} class" for g in result.group_names]
                   + [f"{g} fit x depth" for g in result.group_names]
                   + [f"{g} margin" for g in result.group_names]),
        "fit_depth": (result.fit_depth, result.material_names),
        "sigma": (np.where(np.isfinite(result.sigma), result.sigma, np.nan), result.material_names),
    }
    if params:
        stacks["parameters"] = (np.stack(list(params.values())), list(params))
    written = []
    for kind, (stack, names) in stacks.items():
        grid, transform, *_ = grid_to_equirectangular(lon, lat, stack, resolution_m)
        written.append(write_geotiff(outdir / f"{prefix}_{kind}_map.tif", grid, transform, names))
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_mineral_map(result: IdentificationResult, colors: list[str], path: str | Path,
                     background: np.ndarray | None = None, group: int = 0, title: str = "",
                     extent=None):
    """Class map for one group, drawn over a grey albedo background, with a legend."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch

    cls = result.group_class[group]
    rgb = np.ones((*cls.shape, 3)) * 0.15
    if background is not None:
        from .parameters import stretch
        bg = stretch(background)
        rgb = np.dstack([bg, bg, bg]) * 0.8
    handles = []
    for i, (name, col) in enumerate(zip(result.material_names, colors)):
        mask = cls == i + 1
        if mask.any():
            rgb[mask] = to_rgb(col)
            handles.append(Patch(color=col, label=f"{name} ({mask.sum():,} px)"))
    aspect = "auto" if extent is not None else "equal"
    fig, ax = plt.subplots(figsize=(7, 9))
    ax.imshow(rgb, extent=extent, aspect=aspect, interpolation="nearest")
    if extent is not None:
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    ax.set_title(title or f"Best-matching material – {result.group_names[group]}")
    if handles:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_detection_spectra(result: IdentificationResult, cube: np.ndarray, wavelengths: np.ndarray,
                           library, resolved, path: str | Path, top_n: int = 25):
    """For each detected material: mean continuum-removed spectrum of its strongest
    pixels vs. the fitted reference, per diagnostic feature. The figure reviewers ask for."""
    import matplotlib.pyplot as plt

    from .continuum import local_continuum_removed

    detected = [i for i in range(len(result.material_names)) if result.detected[i].any()]
    if not detected:
        return None
    ncols = max(len(r.features) for mi in detected for r in resolved.references_for(mi))
    fig, axes = plt.subplots(len(detected), ncols, figsize=(4.2 * ncols, 3 * len(detected)), squeeze=False)
    flat_cube = cube.reshape(-1, cube.shape[-1]).astype(np.float64)
    for row, mi in enumerate(detected):
        fd = result.fit_depth[mi].ravel()
        top = np.argsort(fd)[::-1][:top_n]
        top = top[fd[top] > 0]
        best_lib = np.bincount(result.best_reference[mi].ravel()[top]).argmax()
        ref = next(r for r in resolved.references_for(mi) if r.library_index == best_lib)
        for col in range(ncols):
            ax = axes[row, col]
            if col >= len(ref.features):
                ax.axis("off")
                continue
            feat = ref.features[col]
            obs = local_continuum_removed(wavelengths, flat_cube[top], feat.left, feat.right, feat.window)
            mean = np.nanmean(obs, axis=0)
            y = mean - 1
            b = float(y @ feat.ref_cr_minus_1 / (feat.ref_cr_minus_1 @ feat.ref_cr_minus_1))
            wl = wavelengths[feat.window]
            ax.plot(wl, obs.T, color="0.8", lw=0.5)
            ax.plot(wl, mean, "k", lw=2, label=f"M3 mean of {len(top)} px")
            ax.plot(wl, 1 + b * feat.ref_cr_minus_1, "--", color="C3", lw=2, label="fitted reference")
            ax.set_title(f"{result.material_names[mi]} – {feat.name}", fontsize=9)
            ax.set_xlabel("Wavelength (nm)", fontsize=8)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("Continuum-removed", fontsize=8)
                ax.legend(fontsize=7, title=library.names[best_lib][:40], title_fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
