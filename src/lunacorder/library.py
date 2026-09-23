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

    def without_errorbars(self) -> "SpectralLibrary":
        """Drop USGS 'errorbars_for_*' rows (libraries built before v0.1.1 included them)."""
        keep = [i for i, n in enumerate(self.names) if not n.lower().startswith("errorbars")]
        return self.subset(keep)

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
        """Load a saved library. USGS error-bar rows (from builds before v0.1.1) are dropped."""
        with np.load(path, allow_pickle=False) as data:
            fwhm = data["fwhm"]
            lib = cls(
                names=[str(n) for n in data["names"]],
                wavelengths=data["wavelengths"],
                spectra=data["spectra"],
                fwhm=fwhm if fwhm.size else None,
                sources=[str(s) for s in data["sources"]],
            )
        clean = lib.without_errorbars()
        if len(clean) < len(lib):
            import warnings

            warnings.warn(f"{path}: dropped {len(lib) - len(clean)} USGS error-bar rows; rebuild the "
                          "library with lunacorder >= 0.1.1", stacklevel=2)
        return clean

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

BandMembers = list[tuple[np.ndarray, np.ndarray]]
"""Per sensor band: (centres, FWHMs) of the native channels summed to make it."""

_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def srf_matrix(src_wl: np.ndarray, band_centres: np.ndarray, fwhm: np.ndarray,
               members: BandMembers | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Spectral response weights W (bands, n_src) and each band's full-coverage weight.

    Each band is a Gaussian (``band_centres``, ``fwhm``) or, if ``members`` is
    given, the sum of its native channels' Gaussians. M3 global-mode bands are
    averages of 2 or 4 target-mode channels, recorded in the L2 header, so
    ``members`` reproduces their true, flat-topped response.

    Weights include the source sample widths (trapezoid rule), so non-uniform
    lab sampling is handled. Rows are not normalised here, because the
    normalisation must account for missing source samples (see :func:`convolve`).
    """
    src_wl = np.asarray(src_wl, dtype=float)
    widths = np.abs(np.gradient(src_wl)) if len(src_wl) > 1 else np.ones(1)
    if members is None:
        members = [(np.array([c]), np.array([f])) for c, f in zip(np.asarray(band_centres, float),
                                                                   np.asarray(fwhm, float))]
    weights = np.zeros((len(members), len(src_wl)))
    full = np.zeros(len(members))
    for i, (centres, fwhms) in enumerate(members):
        sigma = np.asarray(fwhms, float) * _FWHM_TO_SIGMA
        z = (src_wl[None, :] - np.asarray(centres, float)[:, None]) / sigma[:, None]
        weights[i] = np.exp(-0.5 * z**2).sum(axis=0) * widths
        full[i] = np.sum(sigma) * np.sqrt(2.0 * np.pi)
    return weights, full


def convolve(spectrum: Spectrum, band_centres: np.ndarray, fwhm: np.ndarray,
             min_coverage: float = 0.95, members: BandMembers | None = None) -> np.ndarray:
    """Resample a lab spectrum to sensor bands.

    A band is set to NaN unless at least ``min_coverage`` of its response
    weight falls on valid source samples, so bands beyond the lab
    spectrometer's range are left empty rather than extrapolated.
    """
    wl = np.asarray(spectrum.wavelengths, float)
    refl = np.asarray(spectrum.reflectance, float)
    valid = np.isfinite(wl) & np.isfinite(refl) & (refl > USGS_NODATA_THRESHOLD)
    wl, refl = wl[valid], refl[valid]
    order = np.argsort(wl)
    wl, refl = wl[order], refl[order]
    if len(wl) < 2:
        return np.full(len(band_centres), np.nan)

    weights, full = srf_matrix(wl, band_centres, fwhm, members)
    total = weights.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (weights @ refl) / total
    out[~(total / full >= min_coverage)] = np.nan
    return out


def build_library(spectra: list[Spectrum], band_centres: np.ndarray, fwhm: np.ndarray,
                  min_valid_fraction: float = 0.9,
                  required_range: tuple[float, float] | None = None,
                  members: BandMembers | None = None) -> SpectralLibrary:
    """Convolve spectra to the sensor, dropping any that cover too little of it.

    ``min_valid_fraction`` is evaluated over the bands inside ``required_range``
    (nm), or over all bands if no range is given. For lunar work pass the
    identification range, e.g. (540, 2500), so ASD spectra that stop at
    2.5 um are not rejected for missing M3's thermal bands. ``members`` gives
    the exact composite response of binned bands (see :func:`srf_matrix`).
    """
    band_centres = np.asarray(band_centres, float)
    in_range = np.ones(len(band_centres), bool)
    if required_range is not None:
        in_range = (band_centres >= required_range[0]) & (band_centres <= required_range[1])
    names, rows, sources = [], [], []
    for sp in spectra:
        resampled = convolve(sp, band_centres, fwhm, members=members)
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
    """Unzip into a local cache (not next to the zip).

    On Colab the zip usually lives on Google Drive; extracting there writes thousands
    of small files to Drive, which is slow and clutters the user's folder.
    """
    if path.is_file() and path.suffix.lower() == ".zip":
        import tempfile

        base = workdir or Path(os.environ.get("LUNACORDER_CACHE", Path(tempfile.gettempdir()) / "lunacorder"))
        target = Path(base) / path.stem
        if not (target / ".complete").exists():
            with zipfile.ZipFile(path) as zf:
                zf.extractall(target)
            (target / ".complete").touch()
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
        # Wavelength/bandpass tables and per-spectrum error bars are not spectra
        if "Wavelengths_" in p.name or "Bandpass" in p.name or p.name.lower().startswith("errorbars"):
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


_XML_NAME_TAGS = ("specimen_name", "sample_name", "mineral_name", "specimen_description", "title")


def relab_xml_name(tab_path: Path) -> str | None:
    """Readable specimen name from the PDS4 XML label next to a RELAB ``.tab`` file.

    Label schemas vary between RELAB releases, so the first non-empty element whose
    tag ends with one of ``specimen_name``, ``sample_name``, ``mineral_name``,
    ``specimen_description`` or ``title`` is used (in that order of preference).
    """
    import xml.etree.ElementTree as ET

    candidates = [tab_path.with_suffix(s) for s in (".xml", ".XML")]
    xml = next((c for c in candidates if c.exists()), None)
    if xml is None:
        return None
    try:
        root = ET.parse(xml).getroot()
    except ET.ParseError:
        return None
    found: dict[str, str] = {}
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1].lower()
        text = (el.text or "").strip()
        for key in _XML_NAME_TAGS:
            if tag.endswith(key) and text and key not in found:
                found[key] = " ".join(text.split())
    return next((found[k] for k in _XML_NAME_TAGS if k in found), None)


def read_relab_folder(root: str | os.PathLike, catalog: dict[str, str] | None = None) -> list[Spectrum]:
    """Read every ``.tab`` under a folder (or zip).

    Names come from ``catalog`` (file stem -> name) if given, else from the XML label
    beside each file, else the file stem. The stem is kept in brackets so every
    spectrum stays traceable to its RELAB file.
    """
    root = _extract_if_zip(Path(root))
    spectra = []
    for p in sorted(root.rglob("*.tab")):
        if p.name.startswith("._"):
            continue
        sp = read_relab_tab(p)
        if len(sp.wavelengths) < 10:
            continue
        name = catalog.get(p.stem.lower()) if catalog else None
        name = name or relab_xml_name(p)
        if name:
            sp.name = f"{name} [{p.stem}]"
        spectra.append(sp)
    return spectra


def read_mineral_list(path: str | os.PathLike) -> list[str]:
    """Read a mineral list (one name per line, '#' comments allowed) as regex patterns.

    Spaces match spaces or underscores, so "Olivine GDS70" matches splib07a's
    "Olivine_GDS70.a_Fo89_165u_BECKb_AREF".
    """
    patterns = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            patterns.append(r"[\s_]+".join(re.escape(w) for w in line.split()))
    return patterns


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
