"""
Tests for :mod:`m4opt.models.spectra`.

Each concrete :class:`~m4opt.models.core._base.Spectrum` gets a two-line test
class inheriting the generic checks from
:class:`~m4opt.models.tests._contracts.SpectrumContract`. See that module's
docstring for what is actually being checked.
"""

from m4opt.models.core._base import Spectrum
from m4opt.models.spectra import (
    BlackbodySpectrum,
    BrokenPowerLawSpectrum,
    PowerLawSpectrum,
)

from ._contracts import SpectrumContract, assert_full_coverage


class TestBlackbodySpectrum(SpectrumContract):
    model_class = BlackbodySpectrum


class TestPowerLawSpectrum(SpectrumContract):
    model_class = PowerLawSpectrum


class TestBrokenPowerLawSpectrum(SpectrumContract):
    model_class = BrokenPowerLawSpectrum


def test_all_spectra_covered():
    """Fail loudly if a new `Spectrum` subclass is added without a `Test*` class above."""
    tested = {
        cls.model_class
        for name, cls in globals().items()
        if name.startswith("Test") and issubclass(cls, SpectrumContract)
    }
    assert_full_coverage(Spectrum, tested)
