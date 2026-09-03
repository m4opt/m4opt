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

# Altitude at which those calibration points were measured. The angles above
# are meaningful only relative to how large the Earth looked from there.
_HST_ALTITUDE = 540 * u.km
_HST_ANGULAR_RADIUS_DEG = np.arcsin(R_earth / (R_earth + _HST_ALTITUDE)).to_value(u.deg)




# The Earth is sampled on a Fibonacci lattice, which spaces points over a
# sphere about as evenly as anything this simple can.
_SURFACE_SAMPLES = 512

# Exponent of the point source transmittance, the fraction of the light from an
# off-axis source that an instrument scatters onto its detector. A power law in
# the off-axis angle with this exponent reproduces the STIS calibration points
# above to about 15% when integrated over the Earth as HST sees it.
_PST_EXPONENT = 3.3

# Nearest the line of sight may come to a surface element before the point
# source transmittance, which diverges on axis, is held fixed.
_PST_MIN_ANGLE = np.radians(1.0)


def _fibonacci_sphere(samples):
    index = np.arange(samples) + 0.5
    z = 1 - 2 * index / samples
    radius = np.sqrt(1 - np.square(z))
    azimuth = np.pi * (1 + np.sqrt(5)) * index
    return np.stack([radius * np.cos(azimuth), radius * np.sin(azimuth), z], axis=-1)


_EARTH_SURFACE = _fibonacci_sphere(_SURFACE_SAMPLES)


def _stray_light(distance, observer, target, sun):
    """Earthshine scattered into the line of sight, in arbitrary units.

    Integrates the sunlit, visible part of the Earth, taking each surface
    element to reflect sunlight diffusely, and weighting it by the point source
    transmittance at its angle from the line of sight. Distances are in Earth
    radii and the vectors are unit vectors.

    Only dot products with the surface appear, never the surface vectors
    themselves, so this holds nothing larger than one value per target per
    surface sample.
    """
    surface = _EARTH_SURFACE
    distance = np.asarray(distance)[..., np.newaxis]
    along_observer = np.einsum("...i,ni->...n", observer, surface)
    along_target = np.einsum("...i,ni->...n", target, surface)
    illumination = np.clip(np.einsum("...i,ni->...n", sun, surface), 0.0, None)
    observer_target = np.sum(observer * target, axis=-1)[..., np.newaxis]

    # Distance from each surface element to the observer.
    separation = np.sqrt(
        np.maximum(np.square(distance) - 2 * distance * along_observer + 1, 1e-12)
    )
    # Elements on the far side of the Earth face away and are not visible.
    cos_emission = (distance * along_observer - 1) / separation
    cos_off_axis = np.clip(
        (along_target - distance * observer_target) / separation, -1.0, 1.0
    )
    transmittance = np.power(
        np.maximum(np.arccos(cos_off_axis), _PST_MIN_ANGLE), -_PST_EXPONENT
    )

    contribution = np.where(
        cos_emission > 0,
        illumination * cos_emission * transmittance / np.square(separation),
        0.0,
    )
    return contribution.sum(axis=-1) * 4 * np.pi / len(surface)


def _reference_stray_light():
    """The same integral for HST, 38 degrees from the limb of a full Earth.

    Dividing by this anchors the model to the spectrum, which is a measurement
    made in exactly that configuration.
    """
    distance = ((R_earth + _HST_ALTITUDE) / R_earth).to_value(u.dimensionless_unscaled)
    observer = np.array([0.0, 0.0, 1.0])
    separation = np.radians(_HST_ANGULAR_RADIUS_DEG + _LIMB_ANGLES_DEG[1])
    target = np.array([np.sin(separation), 0.0, -np.cos(separation)])
    return _stray_light(distance, observer, target, observer)


_REFERENCE_STRAY_LIGHT = _reference_stray_light()


class EarthshineBackgroundScaleFactor(ExtrinsicScaleFactor):
    """Scale factor for earthshine that depends on the Earth limb angle.

    The scale factor is interpolated in log2-space between calibration points
    from the HST STIS Instrument Handbook. Targets below the Earth's limb
    receive a scale factor of zero.
    """

    @override
    def at(self, observer_location, target_coord, obstime):
        frame = GCRS(obstime=obstime)

        def unit_vectors(cartesian):
            # Components last, so that they broadcast against arrays of targets.
            return np.moveaxis((cartesian / cartesian.norm()).xyz.value, 0, -1)

        cartesian = observer_location.get_gcrs(obstime).cartesian
        distance = (cartesian.norm() / R_earth).to_value(u.dimensionless_unscaled)
        observer = unit_vectors(cartesian)
        target = unit_vectors(target_coord.transform_to(frame).cartesian)
        sun = unit_vectors(get_sun(obstime).transform_to(frame).cartesian)

        scale = _stray_light(distance, observer, target, sun) / _REFERENCE_STRAY_LIGHT

        # Nothing to see through the Earth.
        angle = _get_angle_from_earth_limb(observer_location, target_coord, obstime)
        scale = np.where(angle.to_value(u.deg) > 0, scale, 0.0)

        if np.ndim(scale) == 0:
            return scale.item()
        return scale


class EarthshineBackground:
    r"""Earthshine sky background: sunlight reflected off Earth.

    This is the earthshine spectrum from the HST STIS Instrument Handbook
    [1]_, `Table 6.4`_, measured 38 degrees from the Earth's limb, scaled to
    the observer's own geometry: where they are, which part of the Earth is
    both sunlit and in view, and how far off axis it lies.

    The Earth is treated as an extended source of stray light. Each element of
    its surface reflects sunlight diffusely, in proportion to the cosine of the
    solar incidence angle there, and the sunlit part of the Earth that the
    observer can see is integrated, weighting each element by the *point source
    transmittance*: the fraction of the light from a source that far off axis
    that the instrument scatters onto its detector. Targets below the limb,
    occulted by the Earth, receive no earthshine at all.

    Lumping the solar spectrum and the Earth's albedo into the measured
    spectrum this way avoids modelling the albedo itself, which in the
    ultraviolet is dominated by ozone absorption and varies by four orders of
    magnitude across a few hundred angstroms.

    The point source transmittance is taken to be a power law in the off-axis
    angle. Its exponent is fixed by the STIS Instrument Handbook and the STScI
    Exposure Time Calculator, which give earthshine levels at three limb
    angles for HST:

    - 24 degrees from the limb: 2.0x the "high" spectrum ("extremely high")
    - 38 degrees from the limb: 1.0x the "high" spectrum ("high", baseline)
    - 50 degrees from the limb: 0.5x the "high" spectrum ("average")

    Integrating over the Earth as HST sees it reproduces those three levels to
    about 25% for an exponent of 3.3, which is an ordinary value for a baffled
    telescope. The integral is normalized by its own value in the configuration
    the "high" spectrum was measured in, so that the spectrum keeps its
    measured meaning and only the change in geometry is predicted.

    The default constructor returns a spatially-dependent model that must be
    evaluated within an :func:`~m4opt.synphot.observing` context. Use
    :meth:`high` to get the constant "high" spectrum without spatial
    dependence.

    .. _`Table 6.4`: https://hst-docs.stsci.edu/stisihb/chapter-6-exposure-time-calculations/6-6-tabular-sky-backgrounds

    .. plot::
        :include-source: False
        :caption: Earthshine seen from a grid of observer positions, all at
            :math:`5 R_\oplus` and all looking back at the Earth. The Earth
            occults the middle of each panel. An observer over the subsolar point
            sees a full Earth and a symmetric halo; one over the midnight
            meridian sees a new Earth and no earthshine at all. The Sun is
            marked where it falls inside a panel, and pointed to by an arrow
            where it does not.

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
        from ligo.skymap.plot import marker
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

        lons = np.arange(-180, 181, 30) * u.deg
        lats = np.arange(-60, 61, 30) * u.deg
        fig, axs = plt.subplots(
            len(lats),
            len(lons),
            figsize=(1.1 * len(lons), 1.1 * len(lats)),
            subplot_kw=dict(aspect=1, xticks=[], yticks=[]),
            gridspec_kw=dict(hspace=0.15, wspace=0.15),
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
                ax.set_xlim(-half_width, half_width)
                ax.set_ylim(-half_width, half_width)
                ax.set_autoscale_on(False)

                # Mark the Sun, as a check that the earthshine is oriented correctly.
                sun_coord = get_body("sun", obstime, observer_location)
                sun_x, sun_y = wcs.all_world2pix(sun_coord.ra.deg, sun_coord.dec.deg, 0)
                if earth_coord.separation(sun_coord) > 175 * u.deg:
                    # The Sun is directly behind the observer, so it has no direction
                    # on the sky to point to.
                    pass
                elif abs(sun_x) < half_width and abs(sun_y) < half_width:
                    ax.plot(
                        sun_x,
                        sun_y,
                        marker=marker.sun,
                        markersize=8,
                        markeredgewidth=1.5,
                        color="orange",
                    )
                else:
                    # Outside the field of view, so point to it from the border.
                    radius = np.hypot(sun_x, sun_y)
                    ax.arrow(
                        np.clip(sun_x, -half_width, half_width),
                        np.clip(sun_y, -half_width, half_width),
                        6 * sun_x / radius,
                        6 * sun_y / radius,
                        color="orange",
                        linewidth=1.5,
                        clip_on=False,
                        head_width=8,
                        head_length=8,
                    )
        for ax, lon in zip(axs[-1], lons):
            ax.set_xlabel(f"{lon:latex}")

        fig.supxlabel("Geocentric longitude of observer")
        fig.supylabel("Geocentric latitude of observer")
        fig.colorbar(image, ax=axs, label="Earthshine scale factor")

    Parameters
    ----------
    factor : float
        Ratio of the observatory's off-axis rejection to HST's, which the
        measured spectrum carries (default: 1). See the warnings below.

    Warnings
    --------
    The calibration points are those of HST, which observes from low Earth orbit
    where the Earth subtends a large solid angle. A profile in limb angle is a
    poor substitute for a ray trace of a particular baffle design, and the further an
    observatory is from the conditions under which the STIS numbers were measured
    -- an observatory at geostationary orbit, for instance -- the less the
    absolute normalization should be trusted. Use the ``factor`` argument to
    renormalize the model to an observatory's own stray light budget.

    Three separate things limit how far from the Earth this model may be used.
    The geometry of the limb itself is exact at any distance, but:

    - The oblateness of the Earth is neglected.

    - The point source transmittance is one power law, with an exponent fixed
      by three measurements spanning 24 to 50 degrees from HST's limb. Real
      instruments differ from one another by orders of magnitude in how well
      they reject off-axis light, and ``factor`` exists to carry that ratio.
      It is a renormalization to an observatory's own stray light budget, not
      a free parameter.

    - The Sun is treated as a point, so the terminator is sharp. Its penumbra
      spans a fraction :math:`r / 217 R_\oplus` of the Earth's angular radius,
      which is 3% at geostationary orbit and 10% at :math:`22 R_\oplus`. At
      :math:`217 R_\oplus` the Sun and the Earth subtend the same angle and the
      terminator is all penumbra.

    - An :class:`~astropy.coordinates.EarthLocation` is fixed to the rotating
      Earth, so a distant observer is carried around at a large speed and
      acquires a correspondingly large stellar aberration: 2 arcseconds at
      geostationary orbit, but 320 arcseconds at :math:`1000 R_\oplus`, which
      exceeds the angular radius of the Earth itself. Beyond a few hundred
      Earth radii an ``EarthLocation`` no longer usefully describes the
      observer.

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
