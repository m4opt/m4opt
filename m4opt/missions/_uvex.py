import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord, get_body
from astropy.time import Time
from regions import RectangleSkyRegion
from synphot import Gaussian1D, SpectralElement

from .. import skygrid
from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..dynamics import EigenAxisSlew, nominal_roll
from ..observer import TleObserverLocation
from ..synphot import Detector
from ..synphot.background import GalacticBackground, ZodiacalBackground
from ._core import Mission

uvex = Mission(
    name="uvex",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=3.5 * u.deg, height=3.5 * u.deg
    ),
    constraints=(
        EarthLimbConstraint(25 * u.deg)
        & SunSeparationConstraint(46 * u.deg)
        & MoonSeparationConstraint(25 * u.deg)
    ),
    detector=Detector(
        npix=4 * np.pi,
        # "This is Nyquist sampled by the 1 arcsec pixels."
        plate_scale=1 * u.arcsec**2,
        # "...an effective aperture of 75cm."
        area=np.pi * np.square(0.5 * 75 * u.cm),
        bandpasses={
            "FUV": SpectralElement(
                Gaussian1D,
                amplitude=0.15,
                mean=1600 * u.angstrom,
                stddev=100 * u.angstrom,
            ),
            "NUV": SpectralElement(
                Gaussian1D,
                amplitude=0.2,
                mean=2300 * u.angstrom,
                stddev=180 * u.angstrom,
            ),
        },
        background=GalacticBackground() + ZodiacalBackground(),
        # Made up to match plot
        read_noise=2,
        dark_noise=1e-3 * u.Hz,
        gain=0.85,
    ),
    # UVEX will be in a highly elliptical TESS-like orbit.
    # This is the TESS TLE downloaded from Celestrak at 2024-09-10T00:43:57Z.
    observer_location=TleObserverLocation(
        "1 43435U 18038A   24262.33225493 -.00001052  00000+0  00000+0 0  9993",
        "2 43435  51.7454  60.8303 4593193 124.3403   0.2501  0.07594463  1386",
    ),
    # Sky grid optimized for full coverage of the sky by circles circumscribed
    # within the square field of view (so that each field is fully covered
    # at all roll angles).
    skygrid=skygrid.geodesic(7.7 * u.deg**2, class_="III", base="icosahedron"),
    # Made up slew model.
    slew=EigenAxisSlew(
        max_angular_velocity=0.6 * u.deg / u.s,
        max_angular_acceleration=0.006 * u.deg / u.s**2,
        settling_time=60 * u.s,
    ),
)
uvex.__doc__ = r"""UVEX, the UltraViolet EXplorer.

`UVEX <https://www.uvex.caltech.edu/>`_ is a NASA Medium Explorer mission to
map the transient sky in the ultraviolet, expected to launch in 2030. UVEX has
a long-slit spectrograph and a 3.5° square field of view camera with two UV
filters.

Note that the imaging mode exposure time calculator is a toy model based on
the publicly available description of the mission from the UVEX science paper
:footcite:`2021arXiv211115608K`, and that roughly reproduces the
`public sensitivity plots <https://www.uvex.caltech.edu/page/for-astronomers>`_.
It will be replaced with realistic filter bandpasses when those are publicly
released.

We make these simplifying assumptions:

- The filter bandpasses are Gassians that mimic the filter shapes on the UVEX
  web site.
- Assume that the PSF is critically sampled.

References
----------
.. footbibliography::

Examples
--------

.. plot::
    :include-source: False
    :caption: Median limiting magnitude, averaged over target coordinates and observation time.

    from astropy import units as u
    from astropy.coordinates import EarthLocation, ICRS
    from astropy_healpix import HEALPix
    from astropy.time import Time
    from matplotlib import pyplot as plt
    from m4opt.missions import uvex
    from m4opt.synphot import observing
    import numpy as np
    from synphot import ConstFlux1D, SourceSpectrum

    dwell = u.def_unit("dwell", 900 * u.s)
    exptime = np.arange(1, 11) * dwell
    obstime = Time("2024-01-01") + np.linspace(0, 1) * u.year
    hpx = HEALPix(128, frame=ICRS())
    target_coords = hpx.healpix_to_skycoord(np.arange(hpx.npix))
    observer_location = EarthLocation(0 * u.m, 0 * u.m, 0 * u.m)

    limmags = []
    for filt in uvex.detector.bandpasses.keys():
        with observing(
            observer_location,
            target_coords[np.newaxis, :, np.newaxis],
            obstime[np.newaxis, np.newaxis, :],
        ):
            limmags.append(
                uvex.detector.get_limmag(
                    5 * np.sqrt(dwell / exptime[:, np.newaxis, np.newaxis]),
                    1 * dwell,
                    SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag),
                    filt,
                ).to_value(u.mag)
            )
    median_limmags = np.median(limmags, axis=[2, 3])

    ax = plt.axes()
    ax.set_xlim(1, 10)
    ax.set_ylim(24.5, 26.5)
    ax.invert_yaxis()
    for filt, limmag in zip(uvex.detector.bandpasses.keys(), median_limmags):
        ax.plot(exptime, limmag, "-o", label=filt)
    ax.legend()
    ax.set_xlabel("Number of stacked 900 s dwells")
    ax.set_ylabel(r"5-$\sigma$ Limiting magnitude (AB)")
    plt.savefig("test.png")

.. plot::
    :include-source: False
    :caption: UVEX filter bandpasses.

    from astropy.visualization import quantity_support
    from astropy import units as u
    from matplotlib import pyplot as plt
    import numpy as np
    from m4opt.missions import uvex

    wavelength = np.linspace(1250, 3000) * u.angstrom
    with quantity_support():
        ax = plt.axes()
        for label, bandpass in uvex.detector.bandpasses.items():
            ax.plot(wavelength, bandpass(wavelength), label=label)
        ax.legend()
"""


def uvex_downlink_orientation(
    time: Time,
) -> tuple[SkyCoord, Angle]:
    r"""
    Get the target coordinates and roll for a UVEX downlink.

    For UVEX, calculate the telescope boresight target coordinate and the roll
    angle of the satellite for a ground contact. This is the orientation that
    points the high gain antenna toward the Earth while maintaining pointing
    constraints.

    Parameters
    ----------
    time:
        Time of the downlink.

    Returns
    -------
    :
        Target coordinate and roll angle.

    Notes
    -----
    UVEX's high gain antenna is at a 45° angle to the spacecraft axes, and
    points toward the cool side and opposite the telescope boresight. In terms
    of the unit vector of the axes of the spacecraft coordinate system shown in
    the illustration for the :meth:`~m4opt.dynamics.nominal_roll` method, the
    direction :math:`\hat{\mathbf{a}}` of the antenna is:

    .. math::
        \hat{\mathbf{a}} = -\frac{\sqrt{2}}{2} \hat{\mathbf{x}} + \frac{\sqrt{2}}{2} \hat{\mathbf{z}}.

    For a ground contact, the following conditions should be met, as shown in
    the diagram below:

    * Solar array drive axis is perpendicular to the direction of the Sun and
      the Earth
    * Earth is 135° from the telescope boresight
    * Sun is ≥45° from the telescope boresight

    The example below shows that all of these constraints are met for downlinks
    at any time over the course of a year.

    .. plot::
        :include-source: False
        :caption: Orientation of the UVEX spacecraft for a downlink.

        from matplotlib import pyplot as plt
        from matplotlib import patches
        import numpy as np

        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(aspect=1)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_frame_on(False)

        # Origin
        ax.plot(0, 0, ".", markersize=10, color="black")

        # Antenna
        ax.text(0.45, -0.05, "Antenna", ha="center", va="top")
        ax.annotate(
            "",
            xy=(0, 0),
            xytext=(0.9, 0),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="<-"),
        )

        # Earth
        ax.text(1, 0, "    Earth", va="center")
        markersize = 16
        for marker in "o+":
            ax.plot(
                1,
                0,
                marker=marker,
                markerfacecolor="white",
                color="black",
                markeredgewidth=1,
                markersize=markersize,
                clip_on=False,
            )

        # Sun
        # ax.text(-1, 0, "Sun   ", ha="right", va="center")
        for sun_angle in np.arange(15, 180, 15):
            color = str(sun_angle / 180)
            angle = np.deg2rad(180 + sun_angle)
            x = np.cos(angle)
            y = np.sin(angle)
            ax.plot(
                x,
                y,
                marker="o",
                markerfacecolor="white",
                color=color,
                markeredgewidth=1,
                markersize=markersize,
                clip_on=False,
            )
            ax.plot(
                x,
                y,
                marker=".",
                color=color,
                markeredgewidth=1,
                clip_on=False,
            )

        ax.add_patch(
            patches.Arc((0, 0), width=0.5, height=0.5, theta1=135, theta2=180, linestyle="--")
        )
        angle = np.deg2rad((135 + 180) / 2)
        ax.text(0.25 * np.cos(angle), 0.25 * np.sin(angle), "≥45°", ha="right", va="baseline")
        ax.annotate(
            "Sun  ",
            xy=(0, 0),
            xytext=(-0.9, 0),
            xycoords="data",
            textcoords="data",
            ha="right",
            va="center",
            arrowprops=dict(arrowstyle="<-", linestyle="--"),
        )

        # Boresight
        ax.annotate(
            "Boresight",
            xy=(0, 0),
            xytext=(-np.sqrt(2) / 2, np.sqrt(2) / 2),
            xycoords="data",
            textcoords="data",
            arrowprops=dict(arrowstyle="<-"),
            ha="right",
            va="bottom",
            rotation=-45,
            clip_on=False,
        )

        # Mark angle
        ax.add_patch(patches.Arc((0, 0), width=0.5, height=0.5, theta1=0, theta2=135))
        angle = np.deg2rad(135 / 2)
        ax.text(0.25 * np.cos(angle), 0.25 * np.sin(angle), "135°", ha="left", va="bottom")

    .. plot::
        :caption: Satisfaction of downlink constraints as a function of time.

        import numpy as np
        from astropy import units as u
        from astropy.coordinates import (
            SkyCoord,
            CartesianRepresentation,
            get_body,
        )
        from astropy.time import Time
        from m4opt.missions import uvex as mission, uvex_downlink_orientation
        from matplotlib import pyplot as plt

        time = Time("2025-01-01") + np.linspace(0, 1, 1000) * u.year
        target, roll = uvex_downlink_orientation(time)

        observer_location = mission.observer_location(time)
        sun = get_body("sun", time, observer_location)
        earth = get_body("earth", time, observer_location)
        spacecraft_frame = target.skyoffset_frame(roll)
        antenna = SkyCoord(
            CartesianRepresentation(-1 / np.sqrt(2), 0, 1 / np.sqrt(2)), frame=spacecraft_frame
        )
        solar_array = SkyCoord(CartesianRepresentation(0, 1, 0), frame=spacecraft_frame)

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.yaxis.set_major_locator(plt.MultipleLocator(45))
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%g°"))
        dt = time.datetime
        ax.plot(dt, target.separation(earth), label=r"Target $\leftrightarrow$ Earth")
        ax.plot(dt, target.separation(sun), label=r"Target $\leftrightarrow$ Sun")
        ax.plot(dt, solar_array.separation(sun), label=r"Solar axis $\leftrightarrow$ Sun")
        ax.plot(dt, antenna.separation(earth), label=r"Target $\leftrightarrow$ antenna")
        ax.legend(title="Separation", loc="center right", bbox_to_anchor=(0.975, 0.25))
    """
    observer_location = uvex.observer_location(time)
    sun_coord = get_body("sun", time, observer_location)
    earth_coord = get_body("earth", time, observer_location)
    roll = nominal_roll(observer_location, earth_coord, time)
    offset_frame = earth_coord.skyoffset_frame(roll)
    target_coord = SkyCoord(180 * u.deg, -45 * u.deg, frame=offset_frame)
    violates_sun_constraint = target_coord.separation(sun_coord) <= 45 * u.deg
    target_coord = SkyCoord(
        180 * u.deg,
        np.where(violates_sun_constraint, 45, -45) * u.deg,
        frame=offset_frame,
    ).transform_to(earth_coord.frame)
    roll = nominal_roll(observer_location, target_coord, time)
    roll[~violates_sun_constraint] += 180 * u.deg
    return target_coord, Angle(roll).wrap_at(180 * u.deg)
