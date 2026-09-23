"""lunacorder – Tetracorder-style mineral identification for Chandrayaan-1 M3 data."""

__version__ = "0.1.0"

from .expert import ExpertSystem  # noqa: E402
from .identify import IdentificationResult, identify, resolve  # noqa: E402
from .library import SpectralLibrary, Spectrum, build_library  # noqa: E402
from .m3 import M3Scene  # noqa: E402

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
