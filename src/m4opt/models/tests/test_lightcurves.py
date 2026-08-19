"""
Tests for :mod:`m4opt.models.lightcurves`.

Each concrete :class:`~m4opt.models.core._base.Lightcurve` gets a two-line
test class inheriting the generic checks from
:class:`~m4opt.models.tests._contracts.LightcurveContract`. See that module's
docstring for what is actually being checked.
"""

from m4opt.models.core._base import Lightcurve
from m4opt.models.lightcurves import (
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

from ._contracts import LightcurveContract, assert_full_coverage


class TestTopHatLightcurve(LightcurveContract):
    model_class = TopHatLightcurve


class TestGaussianPulseLightcurve(LightcurveContract):
    model_class = GaussianPulseLightcurve


class TestFREDLightcurve(LightcurveContract):
    model_class = FREDLightcurve


class TestGREDLightcurve(LightcurveContract):
    model_class = GREDLightcurve


class TestBazinLightcurve(LightcurveContract):
    model_class = BazinLightcurve


class TestPowerLawLightcurve(LightcurveContract):
    model_class = PowerLawLightcurve


class TestBrokenPowerLawLightcurve(LightcurveContract):
    model_class = BrokenPowerLawLightcurve


class TestSmoothBrokenPowerLawLightcurve(LightcurveContract):
    model_class = SmoothBrokenPowerLawLightcurve


class TestDelayedExponentialLightcurve(LightcurveContract):
    model_class = DelayedExponentialLightcurve


class TestLogNormalPulseLightcurve(LightcurveContract):
    model_class = LogNormalPulseLightcurve


class TestPlateauPowerLawLightcurve(LightcurveContract):
    model_class = PlateauPowerLawLightcurve


class TestVillarLightcurve(LightcurveContract):
    model_class = VillarLightcurve


def test_all_lightcurves_covered():
    """Fail loudly if a new `Lightcurve` subclass is added without a `Test*` class above."""
    tested = {
        cls.model_class
        for name, cls in globals().items()
        if name.startswith("Test") and issubclass(cls, LightcurveContract)
    }
    assert_full_coverage(Lightcurve, tested)
