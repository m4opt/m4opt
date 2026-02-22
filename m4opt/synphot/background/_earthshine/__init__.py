"""Earthshine (stray light) background model.

This module models the earthshine background: sunlight reflected off Earth that
scatters into the telescope. The baseline "High Earthshine" spectrum is taken
from `Table 6.4`_ of the HST STIS Instrument Handbook, measured at 38 degrees
from Earth's limb.

The spatial dependence on Earth limb angle is derived from the HST STIS
Instrument Handbook and the STScI Exposure Time Calculator documentation,
which provide discrete earthshine intensity levels at specific limb angles:

- 24 deg from limb: 2.0x the "high" spectrum ("extremely high")
- 38 deg from limb: 1.0x the "high" spectrum ("high", baseline)
- 50 deg from limb: 0.5x the "high" spectrum ("average")

The scale factor is interpolated in log2-space between these calibration
points and clamped at the boundaries. Targets below the Earth's limb
(occluded by Earth) receive zero earthshine.

This is relevant for satellites in Earth orbit (LEO, GEO, etc.) where
earthshine is a significant source of stray light.

.. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

References
----------
.. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
       Handbook," Version 21.0, (Baltimore: STScI)
"""

from importlib import resources
from typing import override

import numpy as np
from astropy import units as u
from astropy.table import QTable
from synphot import Empirical1D, SourceSpectrum, SpectralElement

from ....constraints._earth_limb import _get_angle_from_earth_limb
from ..._extrinsic import ExtrinsicScaleFactor
from .._core import BACKGROUND_SOLID_ANGLE
from . import data

# Calibration points from HST STIS Instrument Handbook, Table 6.4, and
# the STScI ETC documentation.
# See https://etc.stsci.edu/etcstatic/users_guide/1_ref_9_background.html
_LIMB_ANGLES_DEG = np.array([24.0, 38.0, 50.0])
_LOG2_SCALE_FACTORS = np.array([1.0, 0.0, -1.0])  # log2([2.0, 1.0, 0.5])


class EarthshineBackgroundScaleFactor(ExtrinsicScaleFactor):
    """Scale factor for earthshine that depends on the Earth limb angle.

    The scale factor is interpolated in log2-space between calibration points
    from the HST STIS Instrument Handbook. Targets below the Earth's limb
    receive a scale factor of zero.
    """

    @override
    def at(self, observer_location, target_coord, obstime):
        angle = _get_angle_from_earth_limb(observer_location, target_coord, obstime)
        angle_deg = angle.to_value(u.deg)

        log2_scale = np.interp(angle_deg, _LIMB_ANGLES_DEG, _LOG2_SCALE_FACTORS)
        scale = np.exp2(log2_scale)
        scale = np.where(angle_deg > 0, scale, 0.0)

        if np.ndim(angle_deg) == 0:
            return scale.item()
        return scale


class EarthshineBackground:
    """Earthshine sky background: sunlight reflected off Earth.

    This is the earthshine spectrum from the HST STIS Instrument Handbook
    [1]_, `Table 6.4`_, scaled by a factor that depends on the angular
    distance between the target and the Earth's limb.

    The default constructor returns a spatially-dependent model that must be
    evaluated within an :func:`~m4opt.synphot.observing` context. Use
    :meth:`high` to get the constant "high" spectrum without spatial
    dependence.

    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

    References
    ----------
    .. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
           Handbook," Version 21.0, (Baltimore: STScI)

    Examples
    --------

    Constant "high" earthshine spectrum (no spatial dependence):

    >>> from astropy import units as u
    >>> from m4opt.synphot.background import EarthshineBackground
    >>> background = EarthshineBackground.high()
    >>> float(background(5000 * u.angstrom).value) > 0
    True

    """  # noqa: E501

    def __new__(cls):
        return cls.high() * SpectralElement(EarthshineBackgroundScaleFactor())

    @staticmethod
    def high():
        """Earthshine background for "high" conditions (38 deg from limb)."""
        with (
            resources.files(data).joinpath("stis_earthshine_high.ecsv").open("rb") as f
        ):
            table = QTable.read(f, format="ascii.ecsv")
        return SourceSpectrum(
            Empirical1D,
            points=table["wavelength"],
            lookup_table=table["surface_brightness"] * BACKGROUND_SOLID_ANGLE,
        )
