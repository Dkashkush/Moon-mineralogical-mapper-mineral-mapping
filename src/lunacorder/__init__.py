"""lunacorder – Tetracorder-style mineral identification for Chandrayaan-1 M3 data."""

__version__ = "0.1.2"

from .expert import ExpertSystem
from .identify import IdentificationResult, identify, resolve
from .library import SpectralLibrary, Spectrum, build_library
from .m3 import M3Scene

__all__ = [
    "ExpertSystem",
    "IdentificationResult",
    "M3Scene",
    "SpectralLibrary",
    "Spectrum",
    "build_library",
    "identify",
    "resolve",
]
