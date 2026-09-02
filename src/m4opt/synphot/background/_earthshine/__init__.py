"""Earthshine (stray light) background model.

See :class:`m4opt.synphot.background.EarthshineBackground`.
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


# Azimuths at which the visible limb is sampled. The illumination varies
# smoothly around the limb, so a coarse ring is plenty.
_LIMB_AZIMUTHS = np.linspace(0, 2 * np.pi, 36, endpoint=False)

# How sharply the average favors the near side of the limb over the far side.
_LIMB_WEIGHT_CONCENTRATION = 2.0


def _limb_illumination(observer_location, target_coord, obstime):
    """Solar illumination of the visible limb, weighted toward the line of sight."""
    frame = GCRS(obstime=obstime)

    def unit_vectors(cartesian):
        # Components last, so that they broadcast against arrays of targets.
        return np.moveaxis((cartesian / cartesian.norm()).xyz.value, 0, -1)

    observer_cartesian = observer_location.get_gcrs(obstime).cartesian
    radius = observer_cartesian.norm()
    observer = unit_vectors(observer_cartesian)
    target = unit_vectors(target_coord.transform_to(frame).cartesian)
    sun = unit_vectors(get_sun(obstime).transform_to(frame).cartesian)

    sin_angular_radius = (R_earth / radius).to_value(u.dimensionless_unscaled)
    with np.errstate(invalid="ignore"):
        angular_radius = np.arcsin(sin_angular_radius)

    # Orthonormal basis for the plane of the limb. The reference direction is
    # chosen to be the one least parallel to the observer, so that the cross
    # product is well conditioned wherever the observer happens to be.
    reference = np.zeros_like(observer)
    reference[..., 0] = 1.0
    alternative = np.zeros_like(observer)
    alternative[..., 2] = 1.0
    reference = np.where(np.abs(observer[..., :1]) < 0.9, reference, alternative)
    east = np.cross(observer, reference)
    east /= np.linalg.norm(east, axis=-1, keepdims=True)
    north = np.cross(observer, east)

    # Points around the visible limb, and the azimuth of each one.
    cos_phi = np.cos(_LIMB_AZIMUTHS)[:, np.newaxis]
    sin_phi = np.sin(_LIMB_AZIMUTHS)[:, np.newaxis]
    azimuth = cos_phi * east[..., np.newaxis, :] + sin_phi * north[..., np.newaxis, :]
    limb = sin_angular_radius * observer[..., np.newaxis, :] + (
        np.cos(angular_radius) * azimuth
    )
    incidence = np.clip(np.sum(limb * sun[..., np.newaxis, :], axis=-1), 0.0, None)

    # Weight each point of the limb by how far its azimuth lies toward the line
    # of sight. The component of the line of sight across the observer's axis
    # vanishes as the target approaches that axis, so the weights even out into
    # a plain average over the whole limb exactly where its near side stops
    # being well defined.
    along = np.sum(target * observer, axis=-1, keepdims=True)
    across = target - along * observer
    projection = np.sum(across[..., np.newaxis, :] * azimuth, axis=-1)
    weight = np.exp(_LIMB_WEIGHT_CONCENTRATION * projection)
    return np.sum(weight * incidence, axis=-1) / np.sum(weight, axis=-1)


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
    r"""Earthshine sky background: sunlight reflected off Earth.

    This is the earthshine spectrum from the HST STIS Instrument Handbook
    [1]_, `Table 6.4`_, measured 38 degrees from the Earth's limb and scaled
    by a factor that depends on the angular distance between the target and
    the limb, and on the solar illumination of the limb.

    The dependence on limb angle comes from the STIS Instrument Handbook and
    the STScI Exposure Time Calculator, which give earthshine levels at three
    limb angles:

    - 24 degrees from the limb: 2.0x the "high" spectrum ("extremely high")
    - 38 degrees from the limb: 1.0x the "high" spectrum ("high", baseline)
    - 50 degrees from the limb: 0.5x the "high" spectrum ("average")

    The scale factor is interpolated in log2-space between those points and
    extrapolated beyond them. It is then multiplied by the solar illumination
    of the visible limb, the circle of surface points where the observer's
    line of sight is tangent to the Earth, an angle
    :math:`\arccos(R_\oplus / r)` from the sub-observer point. The cosine of
    the solar incidence angle is averaged around that circle, weighted toward
    the part of the limb that the line of sight passes, so that earthshine is
    faintest over the Earth's night side. Averaging rather than taking the
    single nearest point of the limb matters for an observer far from the
    Earth, where the whole limb approaches the terminator and the incidence
    angle at any one point of it is a poor stand-in for how brightly the Earth
    shines into the telescope. Targets below the limb, occulted by the Earth,
    receive no earthshine at all.

    The default constructor returns a spatially-dependent model that must be
    evaluated within an :func:`~m4opt.synphot.observing` context. Use
    :meth:`high` to get the constant "high" spectrum without spatial
    dependence.

    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

    Parameters
    ----------
    factor : float
        Overall normalization, for renormalizing to an observatory's own
        stray light budget (default: 1). See the warnings below.

    Warnings
    --------
    The calibration points are those of HST, which observes from low Earth orbit
    where the Earth subtends a large solid angle. The angular scaling is a poor
    substitute for a ray trace of a particular baffle design, and the further an
    observatory is from the conditions under which the STIS numbers were measured
    -- an observatory at geostationary orbit, for instance -- the less the
    absolute normalization should be trusted. Use the ``factor`` argument to
    renormalize the model to an observatory's own stray light budget.

    Three separate things limit how far from the Earth this model may be used.
    The geometry of the limb itself is exact at any distance, but:

    - The scale factor is a function of the angle from the limb alone, in absolute
      degrees, so it carries no dependence on the observer's distance. The
      earthshine of a distant observer does not fall off as the inverse square of
      that distance the way it should, and ``factor`` has to absorb the difference.

    - The Sun is treated as a point, so the terminator is sharp. Its penumbra
      spans a fraction :math:`r / 217 R_\oplus` of the Earth's angular radius,
      which is 3% at geostationary orbit and 10% at :math:`22 R_\oplus`. At
      :math:`217 R_\oplus` the Sun and the Earth subtend the same angle and the
      terminator is all penumbra.

    - An :class:`~astropy.coordinates.EarthLocation` is fixed to the rotating
      Earth, so a distant observer is carried around at a large speed and acquires
      a correspondingly large stellar aberration: 2 arcseconds at geostationary
      orbit, but 320 arcseconds at :math:`1000 R_\oplus`, which exceeds the
      angular radius of the Earth itself. Beyond a few hundred Earth radii an
      ``EarthLocation`` no longer usefully describes the observer.

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

    .. plot::
        :caption: Earthshine seen from a grid of observer positions, all at
            :math:`5 R_\oplus` and all looking back at the Earth. The Earth
            occults the middle of each panel. An observer over the subsolar point
            sees a full Earth and a symmetric halo; one over the midnight meridian
            sees a new Earth and no earthshine at all.

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import (
            EarthLocation,
            SkyCoord,
            SphericalRepresentation,
            get_body,
        )
        from astropy.time import Time
        from astropy.wcs import WCS
        from matplotlib import pyplot as plt

        from m4opt.synphot.background._earthshine import EarthshineBackgroundScaleFactor

        scale_factor = EarthshineBackgroundScaleFactor()
        distance = 5 * u.Rearth
        # Near the March equinox, when the subsolar point is close to (0, 0).
        obstime = Time("2026-03-20T12:00:00")

        half_width = 45
        x = np.linspace(-half_width, half_width, 200)
        x, y = np.meshgrid(x, x)
        extent = [-half_width, half_width] * 2

        lons = np.arange(-180, 181, 45) * u.deg
        lats = np.arange(-45, 46, 45) * u.deg
        fig, axs = plt.subplots(
            len(lats),
            len(lons),
            figsize=(1.3 * len(lons), 1.3 * len(lats)),
            subplot_kw=dict(aspect=1, xticks=[], yticks=[]),
            gridspec_kw=dict(hspace=0.05, wspace=0.05),
        )
        for row, lat in zip(axs, lats[::-1]):
            row[0].set_ylabel(f"{lat:latex}")
            for ax, lon in zip(row, lons):
                observer_location = EarthLocation(
                    *SphericalRepresentation(lon, lat, distance).to_cartesian().xyz
                )
                earth_coord = get_body("earth", obstime, observer_location)

                wcs = WCS(naxis=2)
                wcs.wcs.crpix = [1, 1]
                wcs.wcs.cdelt = [-1, 1]
                wcs.wcs.crval = [earth_coord.ra.deg, earth_coord.dec.deg]
                wcs.wcs.ctype = ["RA---ARC", "DEC--ARC"]
                target_coord = SkyCoord(*wcs.all_pix2world(x, y, 0), unit=u.deg)

                image = ax.imshow(
                    scale_factor.at(observer_location, target_coord, obstime),
                    vmin=0,
                    vmax=2,
                    origin="lower",
                    extent=extent,
                )
        for ax, lon in zip(axs[-1], lons):
            ax.set_xlabel(f"{lon:latex}")

        fig.supxlabel("Geocentric longitude of observer")
        fig.supylabel("Geocentric latitude of observer")
        fig.colorbar(image, ax=axs, label="Earthshine scale factor")
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
