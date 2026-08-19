"""
Tests for the composite SED models in :mod:`m4opt.models.supernovae` and
:mod:`m4opt.models.tdes`.

Each concrete :class:`~m4opt.models.core._base.SpectralModel` gets a two-line
test class inheriting the generic checks from
:class:`~m4opt.models.tests._contracts.SpectralModelContract`. This covers
both plain `SpectralModel` subclasses (e.g. `VillarCoolingBlackbodySED`) and
`ComposedSpectralModel` subclasses (e.g. `VanVelzenTDESED`) -- both expose
exactly the same public interface. See `_contracts`'s docstring for what is
actually being checked.
"""

from m4opt.models.core._base import ComposedSpectralModel, SpectralModel
from m4opt.models.supernovae import VillarCoolingBlackbodySED
from m4opt.models.tdes import VanVelzenTDESED

from ._contracts import SpectralModelContract, assert_full_coverage


class TestVillarCoolingBlackbodySED(SpectralModelContract):
    model_class = VillarCoolingBlackbodySED


class TestVanVelzenTDESED(SpectralModelContract):
    model_class = VanVelzenTDESED


def test_all_seds_covered():
    """Fail loudly if a new `SpectralModel` subclass is added without a `Test*` class above.

    `ComposedSpectralModel` itself is excluded: it is an extension point
    (`_LIGHTCURVE_CLASS`/`_SPECTRUM_CLASS` unset), not a real model.
    """
    tested = {
        cls.model_class
        for name, cls in globals().items()
        if name.startswith("Test") and issubclass(cls, SpectralModelContract)
    }
    assert_full_coverage(SpectralModel, tested, exclude={ComposedSpectralModel})
