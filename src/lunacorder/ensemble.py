"""Run SAM + SID, LSMA and CEM next to feature fitting, and cross-check all of them.

Feature fitting (identify.py) is the primary product. The other methods are
independent lines of evidence, and a pixel is most trustworthy where they agree.
The ``consensus`` count (0-3) records, for the feature-fitting winner, how many of
SAM+SID, LSMA and CEM agree with it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import classic
from .identify import IdentificationResult, ResolvedExpert
from .library import SpectralLibrary


@dataclass
class EnsembleSettings:
    sam_max_rad: float = 0.10
    sid_max: float = 0.04
    smooth_window: int | None = 7  # Savitzky-Golay window for SAM/SID input; None = off
    lsma: bool = True
    lsma_shade: bool = True
    lsma_max_rmse: float = 0.02
    cem_min_score: float = 0.5
    cem_min_z: float = 3.0


@dataclass
class EnsembleResult:
    material_names: list[str]
    endmember_names: list[str]
    sam_spectrum: np.ndarray  # (rows, cols) library row, -1 none
    sam_angle: np.ndarray
    sid_value: np.ndarray
    samsid_material: np.ndarray  # material index + 1 where SAM & SID agree, 0 otherwise
    lsma_fractions: np.ndarray | None  # (E(+shade), rows, cols)
    lsma_rmse: np.ndarray | None
    lsma_dominant: np.ndarray | None  # material index + 1 of largest non-shade fraction, 0 = bad fit
    cem_score: np.ndarray  # (M, rows, cols); NaN for materials without an endmember
    cem_detect: np.ndarray  # (M, rows, cols) bool
    consensus: np.ndarray  # (rows, cols) 0-3, only where feature fitting detected something
    settings: EnsembleSettings = field(default_factory=EnsembleSettings)


def row_materials(library: SpectralLibrary, resolved: ResolvedExpert) -> np.ndarray:
    """Material index for each library row (-1 if the row is not a reference of any material)."""
    rm = np.full(len(library), -1, int)
    for ref in resolved.references:
        rm[ref.library_index] = ref.material
    return rm


def choose_endmembers(resolved: ResolvedExpert, result: IdentificationResult) -> dict[int, int]:
    """One reference spectrum per material: the one that won most often in feature fitting.

    This picks endmembers from the data rather than by hand. Materials never
    detected fall back to their first reference.
    """
    chosen = {}
    for mi in range(len(result.material_names)):
        refs = resolved.references_for(mi)
        if not refs:
            continue
        winners = result.best_reference[mi][result.detected[mi]]
        chosen[mi] = int(np.bincount(winners).argmax()) if winners.size else refs[0].library_index
    return chosen


def run_ensemble(cube: np.ndarray, wavelengths: np.ndarray, good: np.ndarray, library: SpectralLibrary,
                 resolved: ResolvedExpert, result: IdentificationResult,
                 settings: EnsembleSettings | None = None, progress: bool = True) -> EnsembleResult:
    s = settings or EnsembleSettings()
    shape = cube.shape[:-1]
    wl = np.asarray(wavelengths, float)[good]
    pix = cube.reshape(-1, cube.shape[-1])[:, good].astype(np.float64)
    lib = library.spectra[:, good].astype(np.float64)
    lib_ok = np.all(np.isfinite(lib), axis=1)  # SAM/SID need complete spectra
    lib_rows = np.flatnonzero(lib_ok)
    n_mat = len(result.material_names)

    # --- SAM + SID on continuum-removed (optionally smoothed) spectra -------------
    if progress:
        print("SAM + SID ...", flush=True)
    work = classic.smooth(pix, s.smooth_window) if s.smooth_window else pix
    pix_cr = classic.hull_removed(wl, work, progress=progress)
    lib_cr = classic.hull_removed(wl, lib[lib_rows])
    rm = row_materials(library, resolved)[lib_rows]
    m = classic.sam_sid(pix_cr, lib_cr, s.sam_max_rad, s.sid_max, row_material=rm)
    sam_spectrum = np.where(m.sam_index >= 0, lib_rows[np.maximum(m.sam_index, 0)], -1)

    # --- LSMA + CEM with one data-chosen endmember per material -------------------
    chosen = choose_endmembers(resolved, result)
    em_mats = sorted(chosen)
    em = library.spectra[[chosen[i] for i in em_mats]][:, good].astype(np.float64)
    em_names = [f"{result.material_names[i]} ({library.names[chosen[i]]})" for i in em_mats]

    frac = rmse = dominant = None
    if s.lsma and em_mats:
        if progress:
            print(f"LSMA with {len(em_mats)} endmembers{' + shade' if s.lsma_shade else ''} ...", flush=True)
        frac, rmse = classic.lsma(pix, em, shade=s.lsma_shade, progress=progress)
        mineral = frac[:, : len(em_mats)]
        with np.errstate(invalid="ignore"):
            dom = np.nanargmax(np.where(np.isfinite(mineral), mineral, -1), axis=1)
        dominant = np.where(np.isfinite(rmse) & (rmse <= s.lsma_max_rmse), np.asarray(em_mats)[dom] + 1, 0)

    cem_score = np.full((n_mat, pix.shape[0]), np.nan, np.float32)
    cem_detect = np.zeros((n_mat, pix.shape[0]), bool)
    if em_mats:
        if progress:
            print("CEM ...", flush=True)
        # CEM on continuum-removed band-depth spectra (CR - 1): brightness-independent,
        # so a dark pixel of the target scores like a bright one.
        em_cr = classic.hull_removed(wl, em) - 1.0
        scores = classic.cem(pix_cr - 1.0, em_cr)
        det = classic.cem_detections(scores, s.cem_min_score, s.cem_min_z)
        for j, mi in enumerate(em_mats):
            cem_score[mi] = scores[:, j]
            cem_detect[mi] = det[:, j]

    # --- consensus with the feature-fitting winner (group 0) ----------------------
    winner = result.group_class[0].ravel()
    consensus = np.zeros(pix.shape[0], np.int8)
    has = winner > 0
    consensus += (has & (m.agree_material == winner)).astype(np.int8)
    if dominant is not None:
        consensus += (has & (dominant == winner)).astype(np.int8)
    idx = np.flatnonzero(has)
    consensus[idx] += cem_detect[winner[idx] - 1, idx].astype(np.int8)

    def img(a):
        return None if a is None else a.reshape(a.shape[:-1] + shape) if a.ndim > 1 else a.reshape(shape)

    return EnsembleResult(
        material_names=result.material_names,
        endmember_names=em_names,
        sam_spectrum=sam_spectrum.reshape(shape),
        sam_angle=m.sam_angle.reshape(shape),
        sid_value=m.sid_value.reshape(shape),
        samsid_material=m.agree_material.reshape(shape),
        lsma_fractions=None if frac is None else np.moveaxis(frac, 0, -1).reshape((frac.shape[1], *shape)),
        lsma_rmse=img(rmse),
        lsma_dominant=img(dominant),
        cem_score=cem_score.reshape((n_mat, *shape)),
        cem_detect=cem_detect.reshape((n_mat, *shape)),
        consensus=consensus.reshape(shape),
        settings=s,
    )


def write_crosscheck(result: IdentificationResult, ens: EnsembleResult, path: str | Path) -> Path:
    """Per material: pixels found by each method and how often they agree with feature fitting."""
    path = Path(path)
    winner = result.group_class[0]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["material", "feature_fit_pixels", "samsid_pixels", "lsma_dominant_pixels", "cem_pixels",
                    "samsid_agree_%", "lsma_agree_%", "cem_agree_%", "all_three_agree_%", "status"])
        for i, name in enumerate(result.material_names):
            ff = winner == i + 1
            n = int(ff.sum())
            sam = ens.samsid_material == i + 1
            lsm = ens.lsma_dominant == i + 1 if ens.lsma_dominant is not None else np.zeros_like(ff)
            cm = ens.cem_detect[i]

            def pct(mask, n=n, ff=ff):
                return 100.0 * (ff & mask).sum() / n if n else float("nan")

            full = pct(ens.consensus == 3)
            best = max(pct(sam), pct(lsm), pct(cm)) if n else float("nan")
            status = "" if not n else "GOOD" if best >= 70 else "REVIEW" if best >= 50 else "POOR"
            w.writerow([name, n, int(sam.sum()), int(lsm.sum()), int(cm.sum()),
                        f"{pct(sam):.1f}", f"{pct(lsm):.1f}", f"{pct(cm):.1f}", f"{full:.1f}", status])
    return path
