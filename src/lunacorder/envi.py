"""Minimal, dependency-free reader for ENVI-format images and headers.

M3 Level-2 products on the PDS are raw binary cubes with ENVI ``.hdr`` files.
Rather than trusting assumptions about data type, interleave or band order,
everything here is read from the header itself. Cubes are memory-mapped, so a
2 GB M3 scene can be cropped without loading it into RAM.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ENVI data-type codes -> numpy dtypes
ENVI_DTYPES = {
    1: np.uint8,
    2: np.int16,
    3: np.int32,
    4: np.float32,
    5: np.float64,
    12: np.uint16,
    13: np.uint32,
    14: np.int64,
    15: np.uint64,
}


def parse_header(path: str | os.PathLike) -> dict:
    """Parse an ENVI header into a dict with lower-case keys.

    Brace-delimited values (``{...}``) may span several lines. Numeric lists
    such as ``wavelength`` and ``fwhm`` are returned as numpy arrays; other
    brace values are returned as lists of strings.
    """
    text = Path(path).read_text(errors="replace")
    header: dict = {}
    # key = value, where value is either {...} (possibly multi-line) or the rest of the line
    pattern = re.compile(r"^\s*([^=\n]+?)\s*=\s*(\{.*?\}|[^\n]*)", re.MULTILINE | re.DOTALL)
    for key, value in pattern.findall(text):
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("{"):
            items = [v.strip() for v in value[1:-1].replace("\n", " ").split(",")]
            items = [v for v in items if v != ""]
            header[key] = _maybe_numeric(items)
        else:
            header[key] = value
    return header


def _maybe_numeric(items: list[str]):
    try:
        return np.array([float(v) for v in items])
    except ValueError:
        return items


def find_header(image_path: str | os.PathLike) -> Path:
    """Locate the header for an image, trying every naming convention seen on the PDS.

    For ``scene_loc.img`` this tries ``scene_loc.img.hdr``, ``scene_loc.hdr`` and
    ``scene_loc_img.hdr``, all case-insensitively (M3 OBS files are often upper case).
    """
    image_path = Path(image_path)
    folder = image_path.parent if str(image_path.parent) else Path(".")
    stem = image_path.stem if image_path.suffix else image_path.name
    wanted = {
        f"{image_path.name}.hdr".lower(),
        f"{stem}.hdr".lower(),
        f"{stem}_{image_path.suffix.lstrip('.')}.hdr".lower(),
    }
    for candidate in sorted(folder.iterdir()):
        if candidate.name.lower() in wanted:
            return candidate
    raise FileNotFoundError(
        f"No ENVI header found for {image_path.name} in {folder}. "
        f"Tried (case-insensitive): {sorted(wanted)}"
    )


def find_file(folder: str | os.PathLike, name: str) -> Path:
    """Case-insensitive file lookup inside ``folder``."""
    folder = Path(folder)
    for candidate in folder.iterdir():
        if candidate.name.lower() == name.lower():
            return candidate
    raise FileNotFoundError(f"{name} not found in {folder}")


@dataclass
class EnviImage:
    """A memory-mapped ENVI image. Index with :meth:`read` to get (rows, cols, bands)."""

    path: Path
    header: dict = field(repr=False)

    @classmethod
    def open(cls, image_path: str | os.PathLike, header_path: str | os.PathLike | None = None):
        image_path = Path(image_path)
        header_path = Path(header_path) if header_path else find_header(image_path)
        return cls(path=image_path, header=parse_header(header_path))

    # -- basic geometry -------------------------------------------------
    @property
    def lines(self) -> int:
        return int(self.header["lines"])

    @property
    def samples(self) -> int:
        return int(self.header["samples"])

    @property
    def bands(self) -> int:
        return int(self.header["bands"])

    @property
    def interleave(self) -> str:
        return str(self.header.get("interleave", "bsq")).strip().lower()

    @property
    def dtype(self) -> np.dtype:
        code = int(self.header.get("data type", 4))
        if code not in ENVI_DTYPES:
            raise ValueError(f"Unsupported ENVI data type {code} in {self.path.name}")
        big_endian = int(self.header.get("byte order", 0)) == 1
        return np.dtype(ENVI_DTYPES[code]).newbyteorder(">" if big_endian else "<")

    @property
    def nodata(self) -> float | None:
        value = self.header.get("data ignore value")
        return None if value is None else float(value)

    @property
    def band_names(self) -> list[str]:
        names = self.header.get("band names")
        if names is None:
            return [f"band {i + 1}" for i in range(self.bands)]
        return [str(n) for n in names]

    @property
    def wavelengths(self) -> np.ndarray | None:
        wl = self.header.get("wavelength")
        return None if wl is None else np.asarray(wl, dtype=float)

    @property
    def fwhm(self) -> np.ndarray | None:
        fw = self.header.get("fwhm")
        return None if fw is None else np.asarray(fw, dtype=float)

    @property
    def bbl(self) -> np.ndarray:
        """Bad-band list as booleans (True = good band)."""
        bbl = self.header.get("bbl")
        if bbl is None:
            return np.ones(self.bands, dtype=bool)
        return np.asarray(bbl, dtype=float) > 0

    # -- data access ----------------------------------------------------
    def memmap(self) -> np.memmap:
        """Raw memory map in the file's native interleave order."""
        shapes = {
            "bsq": (self.bands, self.lines, self.samples),
            "bil": (self.lines, self.bands, self.samples),
            "bip": (self.lines, self.samples, self.bands),
        }
        if self.interleave not in shapes:
            raise ValueError(f"Unknown interleave '{self.interleave}' in {self.path.name}")
        offset = int(self.header.get("header offset", 0))
        expected = int(np.prod(shapes[self.interleave])) * self.dtype.itemsize + offset
        actual = os.path.getsize(self.path)
        if actual != expected:
            raise ValueError(
                f"{self.path.name} is {actual:,} bytes but its header describes "
                f"{expected:,} bytes. Are the .img and .hdr from the same product?"
            )
        return np.memmap(self.path, dtype=self.dtype, mode="r", offset=offset,
                         shape=shapes[self.interleave])

    def read(self, rows: slice = slice(None), cols: slice = slice(None),
             bands=slice(None), mask_nodata: bool = True) -> np.ndarray:
        """Read a window as a float32 array shaped (rows, cols, bands).

        Pixels equal to the header's ``data ignore value`` become NaN.
        """
        mm = self.memmap()
        if self.interleave == "bsq":
            block = np.moveaxis(np.asarray(mm[bands, rows, cols]), 0, -1)
        elif self.interleave == "bil":
            block = np.moveaxis(np.asarray(mm[rows, bands, cols]), 1, -1)
        else:
            block = np.asarray(mm[rows, cols, bands])
        block = block.astype(np.float32)
        if mask_nodata and self.nodata is not None:
            block[np.isclose(block, self.nodata)] = np.nan
        return block


def write_envi(path: str | os.PathLike, data: np.ndarray, band_names=None,
               wavelengths=None, nodata: float | None = None, extra: dict | None = None) -> Path:
    """Write a (rows, cols, bands) array as a BSQ ENVI image + header."""
    path = Path(path)
    if data.ndim == 2:
        data = data[:, :, None]
    rows, cols, nb = data.shape
    code = {np.dtype(v): k for k, v in ENVI_DTYPES.items()}[np.dtype(data.dtype).newbyteorder("=")]
    np.ascontiguousarray(np.moveaxis(data, -1, 0)).astype(data.dtype.newbyteorder("<")).tofile(path)
    lines = [
        "ENVI",
        f"samples = {cols}",
        f"lines = {rows}",
        f"bands = {nb}",
        "header offset = 0",
        "file type = ENVI Standard",
        f"data type = {code}",
        "interleave = bsq",
        "byte order = 0",
    ]
    if nodata is not None:
        lines.append(f"data ignore value = {nodata}")
    if band_names is not None:
        lines.append("band names = {" + ", ".join(str(n) for n in band_names) + "}")
    if wavelengths is not None:
        lines.append("wavelength = {" + ", ".join(f"{w:.3f}" for w in wavelengths) + "}")
    for key, value in (extra or {}).items():
        lines.append(f"{key} = {value}")
    path.with_suffix(".hdr").write_text("\n".join(lines) + "\n")
    return path
