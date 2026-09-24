"""High-level workflows: build a scene-matched library, then map a scene."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import __version__, products
from .envi import EnviImage
from .expert import ExpertSystem
from .identify import identify, resolve
from .library import (
    SpectralLibrary,
    build_library,
    read_mineral_list,
    read_relab_folder,
    read_two_column,
    read_usgs_splib07,
)
from .m3 import DEFAULT_RANGE_NM, M3Scene, fwhm_from_spacing, global_band_members
from .parameters import band_parameters


def build_scene_library(scene_header: str | Path, out: str | Path, usgs: str | Path | None = None,
                        relab: str | Path | None = None, extra: list[str | Path] | None = None,
                        usgs_include: list[str] | None = None, mineral_list: str | Path | None = None,
                        required_range: tuple[float, float] = DEFAULT_RANGE_NM) -> SpectralLibrary:
    """Convolve reference spectra to the exact bands of an M3 scene and save them.

    ``scene_header`` is the scene's ``*_rfl.hdr``; its wavelengths (and FWHM, if
    present) define the target bands, so the library can never be on a
    different grid than the image.
    """
    img = EnviImage.open(scene_header, scene_header) if str(scene_header).lower().endswith(".hdr") \
        else EnviImage.open(scene_header)
    wl = img.wavelengths
    if wl is None:
        raise ValueError(f"{scene_header} has no wavelength field")
    if wl.max() < 100:
        wl = wl * 1000.0
    fwhm = img.fwhm if img.fwhm is not None and len(img.fwhm) == len(wl) else fwhm_from_spacing(wl)
    members = global_band_members(img.header)
    if members is not None:
        print(f"Using exact band responses from the header ({sum(len(c) for c, _ in members)} native channels)")
    else:
        print("Header has no channel-binning table; using Gaussian responses from band FWHM")

    if mineral_list:
        usgs_include = [*(usgs_include or []), *read_mineral_list(mineral_list)]
    spectra = []
    if usgs:
        spectra += read_usgs_splib07(usgs, include=usgs_include)
    if relab:
        spectra += read_relab_folder(relab)
    for path in extra or []:
        spectra.append(read_two_column(path))
    if not spectra:
        raise ValueError("No reference spectra were read; check the library paths")
    lib = build_library(spectra, wl, fwhm, required_range=required_range, members=members)
    lib.save(out)
    lib.to_envi_sli(Path(out).with_suffix(".sli"))
    print(f"Read {len(spectra)} spectra; kept {len(lib)} with coverage of {required_range} nm")
    return lib


def map_scene(folder: str | Path, scene_id: str, library: str | Path | SpectralLibrary,
              outdir: str | Path, expert: str | Path | ExpertSystem | None = None,
              rows: slice = slice(None), cols: slice = slice(None),
              max_incidence: float | None = 85.0, map_project: bool = True,
              figures: bool = True, resolution_m: float | None = None,
              max_emission: float | None = None, max_phase: float | None = None,
              ensemble: bool = True, ensemble_settings=None, destripe: str | None = "strip"):
    """Identify minerals in (a window of) an M3 scene and write all products.

    ``ensemble`` also runs SAM + SID, LSMA and CEM and cross-checks them against
    feature fitting (see :mod:`lunacorder.ensemble`).

    ``destripe`` removes along-track detector-column stripes: ``"strip"`` estimates the
    column gains from ~1,000 lines spread over the whole strip, ``"window"`` from the
    window being mapped, ``None`` disables it.
    """
    t0 = time.time()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scene = M3Scene.from_folder(folder, scene_id)
    lib = library if isinstance(library, SpectralLibrary) else SpectralLibrary.load(library)
    if expert is None:
        expert = ExpertSystem.builtin()
    elif not isinstance(expert, ExpertSystem):
        expert = ExpertSystem.from_yaml(expert)

    wl = scene.wavelengths
    good = scene.good_bands(expert.wavelength_range_nm)
    resolved = resolve(expert, lib, wl, good)
    print(resolved.summary())

    print(f"Reading reflectance rows {rows.start}:{rows.stop}, cols {cols.start}:{cols.stop} ...")
    cube = scene.read_reflectance(rows, cols)
    if destripe:
        from .m3 import column_gains, scene_column_gains
        from .m3 import destripe as apply_destripe

        gains = scene_column_gains(scene, good, cols=cols) if destripe == "strip" else column_gains(cube, good)
        cube = apply_destripe(cube, gains)
        print(f"Destriped ({destripe}): column gains {np.percentile(gains[:, good], 1):.3f}-"
              f"{np.percentile(gains[:, good], 99):.3f}")
    valid = np.all(np.isfinite(cube[..., good]), axis=-1)
    masks = {"incidence": max_incidence, "emission": max_emission, "phase": max_phase}
    if scene.obs is not None and any(v is not None for v in masks.values()):
        geom = scene.read_geometry(rows, cols)
        for key, limit in masks.items():
            if limit is not None:
                bad = ~(geom[key] <= limit)
                print(f"  {key} > {limit} deg: {int((bad & valid).sum()):,} pixels masked")
                valid &= ~bad
    cube[~valid] = np.nan
    print(f"Cube {cube.shape}, valid pixels: {valid.sum():,} ({100 * valid.mean():.1f}%)")

    result = identify(cube, resolved, progress=True)
    params = band_parameters(wl, cube, good)
    prefix = scene_id

    products.write_image_products(result, outdir, prefix, params, valid)
    products.write_summary(result, outdir / f"{prefix}_summary.csv", valid)
    ens = None
    if ensemble:
        from .ensemble import run_ensemble, write_crosscheck

        ens = run_ensemble(cube, wl, good, lib, resolved, result, ensemble_settings)
        products.write_ensemble_products(ens, outdir, prefix)
        write_crosscheck(result, ens, outdir / f"{prefix}_crosscheck.csv")
    lon = lat = None
    if scene.loc is not None:
        lon, lat = scene.read_lonlat(rows, cols)
        if map_project:
            try:
                products.write_map_products(result, lon, lat, outdir, prefix, params, resolution_m,
                                            consensus=None if ens is None else ens.consensus, valid=valid)
            except ImportError:
                print("rasterio not installed: skipping GeoTIFF output")
    if figures:
        colors = [m.color for m in expert.materials]
        products.plot_mineral_map(result, colors, outdir / f"{prefix}_mineral_map.png",
                                  background=params["R1580"], title=f"{scene_id} – {expert.name}")
        products.plot_detection_spectra(result, cube, wl, lib, resolved, outdir / f"{prefix}_spectra.png")
        _plot_parameters(params, outdir / f"{prefix}_ibd.png")
        if lon is not None:
            products.plot_projected_map(result, colors, lon, lat, outdir / f"{prefix}_mineral_map_projected.png",
                                        background=params["R1580"], title=f"{scene_id} – {expert.name}",
                                        consensus=None if ens is None else ens.consensus,
                                        resolution_m=resolution_m)

    meta = {
        "lunacorder_version": __version__,
        "scene": scene_id,
        "rows": [rows.start, rows.stop],
        "cols": [cols.start, cols.stop],
        "expert_system": expert.name,
        "bands_used_nm": [float(w) for w in wl[good]],
        "max_incidence_deg": max_incidence,
        "max_emission_deg": max_emission,
        "max_phase_deg": max_phase,
        "destripe": destripe,
        "valid_pixels": int(valid.sum()),
        "ensemble": None if ens is None else {
            "settings": vars(ens.settings),
            "endmembers": ens.endmember_names,
            "consensus_pixels": {str(k): int((ens.consensus == k).sum()) for k in range(4)},
        },
        "references_used": {m.name: [r.name for r in resolved.references_for(i)]
                            for i, m in enumerate(expert.materials)},
        "skipped_materials": resolved.skipped,
        "detections": result.counts(),
        "runtime_s": round(time.time() - t0, 1),
    }
    (outdir / f"{prefix}_run.json").write_text(json.dumps(meta, indent=2))
    from .methods_text import methods_paragraph

    (outdir / f"{prefix}_methods.md").write_text(methods_paragraph(meta, expert, lib))
    print(f"Done in {meta['runtime_s']} s. Products in {outdir}")
    return result, {"cube": cube, "valid": valid, "lon": lon, "lat": lat, "params": params,
                    "resolved": resolved, "library": lib, "ensemble": ens}


def _plot_parameters(params: dict, path: Path):
    import matplotlib.pyplot as plt

    from .parameters import ibd_composite, stretch

    fig, axes = plt.subplots(1, 4, figsize=(14, 6))
    axes[0].imshow(ibd_composite(params), interpolation="nearest")
    axes[0].set_title("R=IBD1000 G=IBD2000 B=R1580", fontsize=9)
    for ax, key in zip(axes[1:], ["IBD1000", "IBD2000", "IBD1250"]):
        ax.imshow(stretch(params[key]), cmap="magma", interpolation="nearest")
        ax.set_title(key, fontsize=9)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def validate_site(folder: str | Path, scene_id: str, library: str | Path | SpectralLibrary, site: str,
                  outdir: str | Path, half_lines: int = 60, radius_km: float = 3.0, **map_kwargs) -> dict:
    """Map the area around a ground-truth site and write a validation report.

    The report combines: detections near the site vs. what the returned samples say
    should and should not be there; the false-alarm rate on featureless nulls; and
    a threshold sweep for every material, so reviewers can see how robust the map is.
    """
    from .sites import APOLLO
    from .validate import false_alarm_rate, site_summary, site_window, threshold_sweep

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    target = APOLLO[site]
    scene = M3Scene.from_folder(folder, scene_id)
    rows, where = site_window(scene, target, half_lines)
    print(f"{target.name}: row {where['row']}, col {where['col']} "
          f"({where['distance_km']:.2f} km from the LM, pixel {where['pixel_km']:.2f} km)")
    result, extras = map_scene(folder, scene_id, library, outdir, rows=rows, **map_kwargs)
    resolved, good = extras["resolved"], scene.good_bands(extras["resolved"].expert.wavelength_range_nm)

    report = {
        "site": where,
        "rows": [rows.start, rows.stop],
        "near_site": site_summary(result, extras["lon"], extras["lat"], target, radius_km),
        "false_alarm_rate": false_alarm_rate(extras["cube"], resolved, good),
        "threshold_sweep": {m: threshold_sweep(result, m) for m in result.material_names
                            if resolved.references_for(result.material_names.index(m))},
        "sample_notes": target.notes,
    }
    (outdir / f"{scene_id}_{site}_validation.json").write_text(json.dumps(report, indent=2))
    (outdir / f"{scene_id}_{site}_validation.md").write_text(_validation_markdown(report, target))
    print((outdir / f"{scene_id}_{site}_validation.md").read_text())
    return report


def _validation_markdown(report: dict, site) -> str:
    near = report["near_site"]
    lines = [f"# Validation at {site.name} ({site.setting})", "",
             f"Closest pixel {report['site']['distance_km']:.2f} km from the LM; "
             f"{near['pixels']} pixels within {near['radius_km']} km.", "",
             "## Against the returned samples", "", "| Material | Expected | Detected within radius | Result |",
             "|---|---|---|---|"]
    for c in near["checks"]:
        lines.append(f"| {c['material']} | {c['expected']} | {c['percent']:.1f} % | "
                     f"{'PASS' if c['pass'] else 'FAIL'} |")
    lines += ["", f"Sample notes: {site.notes}", "", "## False-alarm rate (featureless nulls)", "",
              "| Material | False alarms |", "|---|---|"]
    lines += [f"| {m} | {100 * v:.2f} % |" for m, v in report["false_alarm_rate"].items()]
    lines += ["", "## Threshold sensitivity (percent of valid pixels detected)", ""]
    for m, rows in report["threshold_sweep"].items():
        fits = sorted({r["min_fit"] for r in rows})
        lines += [f"**{m}**", "", "| min depth \\ min fit | " + " | ".join(f"{f:.2f}" for f in fits) + " |",
                  "|---" * (len(fits) + 1) + "|"]
        for d in sorted({r["min_depth"] for r in rows}):
            vals = [next(r["percent"] for r in rows if r["min_depth"] == d and r["min_fit"] == f) for f in fits]
            lines.append(f"| {d:.2f} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
        lines.append("")
    return "\n".join(lines)
