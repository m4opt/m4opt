"""
Astropy models for synthetic photometry modeling.
"""

from ._base import ComposedSEDModel, LightcurveModel, SEDModel, Spectrum
from ._lightcurves import (
    BazinLightcurve,
    BrokenPowerLawLightcurve,
    DelayedExponentialLightcurve,
    FREDLightcurve,
    GaussianPulseLightcurve,
    GREDLightcurve,
    LogNormalPulseLightcurve,
    PlateauPowerLawLightcurve,
    PowerLawLightcurve,
    SmoothBrokenPowerLawLightcurve,
    TopHatLightcurve,
    VillarLightcurve,
)
from ._parameters import PriorParameter
from ._priors import LogNormalPrior, NormalPrior, UniformPrior
from ._spectra import BlackbodySpectrum, BrokenPowerLawSpectrum, PowerLawSpectrum
from ._supernovae import VillarCoolingBlackbodySED
from ._tdes import VanVelzenTDESED

__all__ = (
    "BazinLightcurve",
    "BlackbodySpectrum",
    "BrokenPowerLawLightcurve",
    "BrokenPowerLawSpectrum",
    "ComposedSEDModel",
    "DelayedExponentialLightcurve",
    "FREDLightcurve",
    "GREDLightcurve",
    "GaussianPulseLightcurve",
    "LightcurveModel",
    "LogNormalPrior",
    "LogNormalPulseLightcurve",
    "NormalPrior",
    "PlateauPowerLawLightcurve",
    "PowerLawLightcurve",
    "PowerLawSpectrum",
    "PriorParameter",
    "SEDModel",
    "SmoothBrokenPowerLawLightcurve",
    "Spectrum",
    "TopHatLightcurve",
    "UniformPrior",
    "VanVelzenTDESED",
    "VillarCoolingBlackbodySED",
    "VillarLightcurve",
)
