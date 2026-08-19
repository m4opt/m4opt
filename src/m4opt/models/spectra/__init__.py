"""
Frequency-dependent spectral shape models.
"""

__all__ = [
    "BlackbodySpectrum",
    "BrokenPowerLawSpectrum",
    "PowerLawSpectrum",
]

from .powerlaw import BrokenPowerLawSpectrum, PowerLawSpectrum
from .thermal import BlackbodySpectrum
