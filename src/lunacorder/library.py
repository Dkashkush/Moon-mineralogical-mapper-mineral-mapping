"""Reference spectral libraries: reading, convolving to a sensor, saving.

Supported inputs
----------------
* **USGS splib07a ASCII** (Kokaly et al., 2017). Wavelength files sit at the
  root of ``ASCIIdata_splib07a/`` and are matched to each spectrum through the
  spectrometer code in its filename (``_BECKb_`` -> BECK, ``_ASDFRa_`` -> ASD,
  ...). The first line of every file is a title; deleted channels hold
  ``-1.23e34``. Wavelengths are in micrometres.
* **RELAB ``.tab``** files: first line is a row count, then columns of
  wavelength (nm), reflectance and (optionally) error. The count and the last
  row are not always consistent, so rows are only kept when both numbers parse.
* **Generic two-column text/CSV** (wavelength, reflectance), for any other lab
  or field spectra you want to add.

All spectra are convolved to the sensor with Gaussian spectral response
functions defined by band centre and FWHM, so the same library code serves
M3 target mode, M3 global mode, or any other imaging spectrometer.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

USGS_NODATA_THRESHOLD = -1e30


@dataclass
class Spectrum:
    """A single laboratory spectrum on its native wavelength grid (nm)."""

    name: str
    wavelengths: np.ndarray
    reflectance: np.ndarray
    source: str = ""


@dataclass
class SpectralLibrary:
    """Spectra resampled to a common set of sensor bands."""

    names: list[str]
    wavelengths: np.ndarray  # (bands,) nm
    spectra: np.ndarray  # (n_spectra, bands)
    fwhm: np.ndarray | None = None
    sources: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    def find(self, pattern: str) -> list[int]:
        """Indices of spectra whose name matches a regular expression (case-insensitive)."""
        rx = re.compile(pattern, re.IGNORECASE)
        return [i for i, n in enumerate(self.names) if rx.search(n)]

    def subset(self, indices) -> "SpectralLibrary":
        indices = list(indices)
        return SpectralLibrary(
            names=[self.names[i] for i in indices],
            wavelengths=self.wavelengths,
            spectra=self.spectra[indices],
            fwhm=self.fwhm,
            sources=[self.sources[i] for i in indices] if self.sources else [],
        )

    def save(self, path: str | os.PathLike) -> Path:
        path = Path(path)
        np.savez_compressed(
            path,
            names=np.array(self.names),
            wavelengths=self.wavelengths,
            spectra=self.spectra,
            fwhm=self.fwhm if self.fwhm is not None else np.array([]),
            sources=np.array(self.sources if self.sources else [""] * len(self.names)),
        )
        return path

    @classmethod
    def load(cls, path: str | os.PathLike) -> "SpectralLibrary":
        with np.load(path, allow_pickle=False) as data:
            fwhm = data["fwhm"]
            return cls(
                names=[str(n) for n in data["names"]],
                wavelengths=data["wavelengths"],
                spectra=data["spectra"],
                fwhm=fwhm if fwhm.size else None,
                sources=[str(s) for s in data["sources"]],
            )

    def to_envi_sli(self, path: str | os.PathLike) -> Path:
        """Export as an ENVI spectral library (.sli + .hdr) for use in ENVI/QGIS."""
        path = Path(path)
        self.spectra.astype("<f4").tofile(path)
        header = [
            "ENVI",
            f"samples = {self.spectra.shape[1]}",
            f"lines = {self.spectra.shape[0]}",
            "bands = 1",
            "header offset = 0",
            "file type = ENVI Spectral Library",
            "data type = 4",
            "interleave = bsq",
            "byte order = 0",
            "wavelength units = Nanometers",
            "wavelength = {" + ", ".join(f"{w:.3f}" for w in self.wavelengths) + "}",
            "spectra names = {" + ", ".join(n.replace(",", ";") for n in self.names) + "}",
        ]
        if self.fwhm is not None:
            header.append("fwhm = {" + ", ".join(f"{w:.3f}" for w in self.fwhm) + "}")
        path.with_suffix(".hdr").write_text("\n".join(header) + "\n")
        return path


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

def gaussian_srf_matrix(src_wl: np.ndarray, band_centres: np.ndarray, fwhm: np.ndarray) -> np.ndarray:
    """Weights W (bands, n_src) so that ``W @ spectrum`` integrates each Gaussian SRF.

    Weights include the source sample widths (trapezoid rule), so non-uniform
    lab sampling is handled correctly. Rows are *not* normalised here because
    normalisation must account for missing source samples; see :func:`convolve`.
    """
    src_wl = np.asarray(src_wl, dtype=float)
    sigma = np.asarray(fwhm, dtype=float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    widths = np.gradient(src_wl) if len(src_wl) > 1 else np.ones(1)
    z = (src_wl[None, :] - np.asarray(band_centres, float)[:, None]) / sigma[:, None]
    return np.exp(-0.5 * z**2) * np.abs(widths)[None, :]


def convolve(spectrum: Spectrum, band_centres: np.ndarray, fwhm: np.ndarray,
             min_coverage: float = 0.95) -> np.ndarray:
    """Resample a lab spectrum to sensor bands.

    A band is set to NaN unless at least ``min_coverage`` of its SRF weight
    falls on valid source samples, so bands beyond the lab spectrometer's range
    are left empty rather than extrapolated.
    """
    wl = np.asarray(spectrum.wavelengths, float)
    refl = np.asarray(spectrum.reflectance, float)
    valid = np.isfinite(wl) & np.isfinite(refl) & (refl > USGS_NODATA_THRESHOLD)
    wl, refl = wl[valid], refl[valid]
    order = np.argsort(wl)
    wl, refl = wl[order], refl[order]
    if len(wl) < 2:
        return np.full(len(band_centres), np.nan)

    weights = gaussian_srf_matrix(wl, band_centres, fwhm)
    total = weights.sum(axis=1)
    # Expected total weight of a fully sampled Gaussian: integral = sigma*sqrt(2*pi)
    sigma = np.asarray(fwhm, float) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    full = sigma * np.sqrt(2.0 * np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (weights @ refl) / total
    out[~(total / full >= min_coverage)] = np.nan
    return out


def build_library(spectra: list[Spectrum], band_centres: np.ndarray, fwhm: np.ndarray,
                  min_valid_fraction: float = 0.9,
                  required_range: tuple[float, float] | None = None) -> SpectralLibrary:
    """Convolve spectra to the sensor, dropping any that cover too little of it.

    ``min_valid_fraction`` is evaluated over the bands inside ``required_range``
    (nm), or over all bands if no range is given. For lunar work pass the
    identification range, e.g. (540, 2500), so ASD spectra that stop at
    2.5 um are not rejected for missing M3's thermal bands.
    """
    band_centres = np.asarray(band_centres, float)
    in_range = np.ones(len(band_centres), bool)
    if required_range is not None:
        in_range = (band_centres >= required_range[0]) & (band_centres <= required_range[1])
    names, rows, sources = [], [], []
    for sp in spectra:
        resampled = convolve(sp, band_centres, fwhm)
        if np.isfinite(resampled[in_range]).mean() >= min_valid_fraction:
            names.append(sp.name)
            rows.append(resampled)
            sources.append(sp.source)
    spectra_arr = np.array(rows, dtype=np.float32).reshape(len(rows), len(band_centres))
    return SpectralLibrary(names=names, wavelengths=np.asarray(band_centres, float),
                           spectra=spectra_arr, fwhm=np.asarray(fwhm, float), sources=sources)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def _extract_if_zip(path: Path, workdir: Path | None = None) -> Path:
    if path.is_file() and path.suffix.lower() == ".zip":
        target = (workdir or path.parent) / path.stem
        if not target.exists():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
        return target
    return path


def _read_number_column(path: Path, skip: int = 1) -> np.ndarray:
    values = []
    for line in path.read_text(errors="replace").splitlines()[skip:]:
        parts = line.split()
        if not parts:
            continue
        try:
            values.append(float(parts[0]))
        except ValueError:
            continue
    return np.array(values)


_USGS_WL_RX = re.compile(r"Wavelengths_([A-Za-z0-9]+)_", re.IGNORECASE)
# Spectrometer token just before the measurement type, e.g. ..._BECKb_AREF.txt
_USGS_SPEC_RX = re.compile(r"_([A-Za-z0-9]+)_(?:AREF|RREF|RTGC|TRAN|ABS)\.txt$", re.IGNORECASE)


def read_usgs_splib07(root: str | os.PathLike, include: list[str] | None = None,
                      exclude: list[str] | None = None) -> list[Spectrum]:
    """Read USGS splib07a ASCII spectra.

    Parameters
    ----------
    root : folder or .zip of ``ASCIIdata_splib07a``.
    include : optional regex patterns; a spectrum is kept if its name matches any.
    exclude : optional regex patterns; matching spectra are dropped.
    """
    root = _extract_if_zip(Path(root))
    txt_files = [p for p in root.rglob("*.txt") if not p.name.startswith("._")]

    wavelength_files = {}
    for p in txt_files:
        m = _USGS_WL_RX.search(p.name)
        if m:
            wavelength_files[m.group(1).upper()] = p
    if not wavelength_files:
        raise FileNotFoundError(f"No splib07 wavelength files (*Wavelengths_*.txt) under {root}")
    grids = {key: _read_number_column(p) * 1000.0 for key, p in wavelength_files.items()}  # um -> nm

    inc = [re.compile(p, re.I) for p in (include or [])]
    exc = [re.compile(p, re.I) for p in (exclude or [])]
    spectra = []
    for p in sorted(txt_files):
        if "Wavelengths_" in p.name or "Bandpass" in p.name:
            continue
        m = _USGS_SPEC_RX.search(p.name)
        if not m:
            continue
        token = m.group(1).upper()
        # Longest wavelength-grid key that the spectrometer token starts with
        keys = [k for k in grids if token.startswith(k)]
        if not keys:
            continue
        grid = grids[max(keys, key=len)]
        name = p.stem.removeprefix("splib07a_")
        if inc and not any(r.search(name) for r in inc):
            continue
        if exc and any(r.search(name) for r in exc):
            continue
        refl = _read_number_column(p)
        if len(refl) != len(grid):
            continue
        refl = np.where(refl < USGS_NODATA_THRESHOLD, np.nan, refl)
        spectra.append(Spectrum(name=name, wavelengths=grid, reflectance=refl, source="USGS splib07a"))
    return spectra


def read_relab_tab(path: str | os.PathLike) -> Spectrum:
    """Read one RELAB ``.tab`` file (count line, then wavelength-nm / reflectance rows)."""
    path = Path(path)
    wl, refl = [], []
    for line in path.read_text(errors="replace").splitlines()[1:]:
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            w, r = float(parts[0]), float(parts[1])
        except ValueError:
            continue  # keep the two columns aligned: append both or neither
        wl.append(w)
        refl.append(r)
    wl = np.array(wl)
    if wl.size and wl.max() < 100:  # some files are in micrometres
        wl = wl * 1000.0
    return Spectrum(name=path.stem, wavelengths=wl, reflectance=np.array(refl), source="RELAB")


def read_relab_folder(root: str | os.PathLike, catalog: dict[str, str] | None = None) -> list[Spectrum]:
    """Read every ``.tab`` under a folder (or zip). ``catalog`` can map file stems to readable names."""
    root = _extract_if_zip(Path(root))
    spectra = []
    for p in sorted(root.rglob("*.tab")):
        if p.name.startswith("._"):
            continue
        sp = read_relab_tab(p)
        if len(sp.wavelengths) < 10:
            continue
        if catalog and p.stem.lower() in catalog:
            sp.name = f"{catalog[p.stem.lower()]} [{p.stem}]"
        spectra.append(sp)
    return spectra


def read_two_column(path: str | os.PathLike, name: str | None = None, source: str = "user") -> Spectrum:
    """Read any whitespace/comma separated (wavelength, reflectance) text file."""
    path = Path(path)
    data = []
    for line in path.read_text(errors="replace").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            data.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    arr = np.array(data)
    wl = arr[:, 0] * (1000.0 if arr[:, 0].max() < 100 else 1.0)
    return Spectrum(name=name or path.stem, wavelengths=wl, reflectance=arr[:, 1], source=source)
