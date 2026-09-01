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
points and extrapolated beyond them. It is then multiplied by the solar
illumination of the part of the limb that the line of sight passes, so that
earthshine falls to zero over the Earth's night side. Targets below the
Earth's limb (occluded by Earth) receive zero earthshine.

.. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

Warnings
--------
The calibration points are those of HST, which observes from low Earth orbit
where the Earth subtends a large solid angle. The angular scaling is a poor
substitute for a ray trace of a particular baffle design, and the further an
observatory is from the conditions under which the STIS numbers were measured
-- an observatory at geostationary orbit, for instance -- the less the
absolute normalization should be trusted. Use the ``factor`` argument to
renormalize the model to an observatory's own stray light budget.

References
----------
.. [1] Prichard, L., Welty, D. and Jones, A., et al. 2022 "STIS Instrument
       Handbook," Version 21.0, (Baltimore: STScI)
"""

from importlib import resources
from typing import override

import numpy as np
from astropy import units as u
from astropy.constants import R_earth
from astropy.coordinates import GCRS, get_sun
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


def _limb_illumination(observer_location, target_coord, obstime):
    """Solar illumination where the line of sight grazes the Earth's limb.

    The grazing point lies in the plane containing the observer and the target,
    at an angle :math:`\\arccos(R_\\oplus / r)` from the sub-observer point.
    The return value is the cosine of the solar incidence angle there, so it
    falls to zero as that part of the limb turns away from the Sun.
    """
    frame = GCRS(obstime=obstime)

    def unit_vectors(cartesian):
        # Components last, so that they broadcast against arrays of targets.
        return np.moveaxis((cartesian / cartesian.norm()).xyz.value, 0, -1)

    observer_cartesian = observer_location.get_gcrs(obstime).cartesian
    radius = observer_cartesian.norm()
    observer = unit_vectors(observer_cartesian)
    target = unit_vectors(target_coord.transform_to(frame).cartesian)
    sun = unit_vectors(get_sun(obstime).transform_to(frame).cartesian)

    # Component of the line of sight perpendicular to the sub-observer point,
    # which selects the side of the limb that the telescope is looking past.
    along = np.sum(target * observer, axis=-1, keepdims=True)
    perpendicular = target - along * observer
    norm = np.linalg.norm(perpendicular, axis=-1, keepdims=True)
    # Looking straight up or straight down leaves the direction undefined; the
    # limb is then equidistant all around, so any of it will do.
    perpendicular = np.divide(
        perpendicular, norm, out=np.zeros_like(perpendicular), where=norm > 0
    )

    with np.errstate(invalid="ignore"):
        gamma = np.arccos((R_earth / radius).to_value(u.dimensionless_unscaled))
    grazing = np.cos(gamma) * observer + np.sin(gamma) * perpendicular
    return np.clip(np.sum(grazing * sun, axis=-1), 0.0, None)


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

        # Extrapolate beyond the last calibration point using the slope
        # from the last two points, so earthshine decreases at large angles
        # rather than clamping at 0.5.
        slope = (_LOG2_SCALE_FACTORS[-1] - _LOG2_SCALE_FACTORS[-2]) / (
            _LIMB_ANGLES_DEG[-1] - _LIMB_ANGLES_DEG[-2]
        )
        log2_scale = np.where(
            angle_deg > _LIMB_ANGLES_DEG[-1],
            _LOG2_SCALE_FACTORS[-1] + slope * (angle_deg - _LIMB_ANGLES_DEG[-1]),
            log2_scale,
        )

        # Modulate by the illumination of the part of the limb that the
        # telescope is looking past, which is dark when the observer is over
        # the Earth's night side.
        scale = np.exp2(log2_scale) * _limb_illumination(
            observer_location, target_coord, obstime
        )

        # Zero for targets behind the Earth, and for observers so close to the
        # geocenter that the limb angle is undefined.
        scale = np.where(angle_deg > 0, scale, 0.0)

        if np.ndim(angle_deg) == 0:
            return scale.item()
        return scale


class EarthshineBackground:
    """Earthshine sky background: sunlight reflected off Earth.

    This is the earthshine spectrum from the HST STIS Instrument Handbook
    [1]_, `Table 6.4`_, scaled by a factor that depends on the angular
    distance between the target and the Earth's limb, and on the solar
    illumination of that part of the limb.

    The default constructor returns a spatially-dependent model that must be
    evaluated within an :func:`~m4opt.synphot.observing` context. Use
    :meth:`high` to get the constant "high" spectrum without spatial
    dependence.

    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

    Parameters
    ----------
    factor : float
        Overall normalization, for renormalizing to an observatory's own
        stray light budget (default: 1). See the warning in
        :mod:`m4opt.synphot.background`.

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

    """

    def __new__(cls, factor: float = 1):
        return factor * cls.high() * SpectralElement(EarthshineBackgroundScaleFactor())

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
