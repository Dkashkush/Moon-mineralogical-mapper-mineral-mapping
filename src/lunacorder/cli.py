"""Command-line interface.

Examples
--------
Build a library matched to your scene's bands::

    lunacorder build-library --scene-header M3_Project/m3g20090607t025544_v01_rfl.hdr \\
        --usgs ASCIIdata_splib07a.zip --relab Relab_lunar_mineral_spectra.zip --out m3_library.npz

Map a window of the scene::

    lunacorder map --folder M3_Project --scene m3g20090607t025544_v01 \\
        --library m3_library.npz --rows 3000:4500 --out results/
"""

from __future__ import annotations

import argparse

from . import __version__


def _slice(text: str | None) -> slice:
    if not text:
        return slice(None)
    start, _, stop = text.partition(":")
    return slice(int(start) if start else None, int(stop) if stop else None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lunacorder", description=__doc__.split("\n")[0])
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build-library", help="convolve lab spectra to a scene's bands")
    b.add_argument("--scene-header", required=True, help="the scene's *_rfl.hdr")
    b.add_argument("--usgs", help="ASCIIdata_splib07a folder or zip")
    b.add_argument("--relab", help="folder or zip of RELAB .tab files")
    b.add_argument("--extra", nargs="*", default=[], help="two-column (wavelength, reflectance) files")
    b.add_argument("--mineral-list", help="text file, one USGS mineral name per line, to select spectra")
    b.add_argument("--out", required=True, help="output .npz (an ENVI .sli is written alongside)")

    m = sub.add_parser("map", help="identify minerals in an M3 scene")
    m.add_argument("--folder", required=True, help="folder with *_rfl/_loc/_obs .img + .hdr")
    m.add_argument("--scene", required=True, help="scene id, e.g. m3g20090607t025544_v01")
    m.add_argument("--library", required=True, help=".npz from build-library")
    m.add_argument("--expert", help="expert-system YAML (default: built-in lunar_m3)")
    m.add_argument("--rows", help="row window, e.g. 3000:4500")
    m.add_argument("--cols", help="column window, e.g. 0:304")
    m.add_argument("--max-incidence", type=float, default=85.0)
    m.add_argument("--max-emission", type=float, help="mask pixels above this emission angle (deg)")
    m.add_argument("--max-phase", type=float, help="mask pixels above this phase angle (deg)")
    m.add_argument("--no-ensemble", action="store_true", help="skip SAM+SID, LSMA and CEM cross-checks")
    m.add_argument("--resolution-m", type=float, help="map-projected cell size (default: native)")
    m.add_argument("--no-geotiff", action="store_true")
    m.add_argument("--no-figures", action="store_true")
    m.add_argument("--out", required=True)

    s = sub.add_parser("subset", help="copy a window of a scene (all three cubes) to a new folder")
    s.add_argument("--folder", required=True)
    s.add_argument("--scene", required=True)
    s.add_argument("--rows", required=True, help="e.g. 3000:3150")
    s.add_argument("--cols", help="e.g. 0:304 (default: all)")
    s.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    from . import pipeline

    if args.command == "subset":
        from .m3 import M3Scene, subset_scene

        scene = M3Scene.from_folder(args.folder, args.scene)
        print(subset_scene(scene, args.out, _slice(args.rows), _slice(args.cols)))
        return
    if args.command == "build-library":
        pipeline.build_scene_library(args.scene_header, args.out, usgs=args.usgs,
                                     relab=args.relab, extra=args.extra, mineral_list=args.mineral_list)
    else:
        pipeline.map_scene(args.folder, args.scene, args.library, args.out, expert=args.expert,
                           rows=_slice(args.rows), cols=_slice(args.cols),
                           max_incidence=args.max_incidence, map_project=not args.no_geotiff,
                           figures=not args.no_figures, resolution_m=args.resolution_m,
                           max_emission=args.max_emission, max_phase=args.max_phase,
                           ensemble=not args.no_ensemble)


if __name__ == "__main__":
    main()
