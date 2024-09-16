import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
from synphot import Box1D, SpectralElement

from .. import skygrid
from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..models import Detector
from ..models.background import ZodiacalBackground
from ..orbit import TLE
from ..utils.dynamics import Slew
from ._core import Mission


def box_for_lo_hi(lo, hi):
    return SpectralElement(Box1D, x_0=0.5 * (lo + hi), width=hi - lo)


uvex = Mission(
    name="uvex",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=3.5 * u.deg, height=3.5 * u.deg
    ),
    constraints=[
        EarthLimbConstraint(25 * u.deg),
        SunSeparationConstraint(46 * u.deg),
        MoonSeparationConstraint(25 * u.deg),
    ],
    detector=Detector(
        # "The OTA provides a field-averaged point spread function (PSF) with
        # half-power diameter (HPD) <2.25 arcsec."
        #
        # Assuming a Gaussian PSF with profile exp(-(r/a)^2/2),
        # a = HPD / (2 sqrt(2 ln(2))).
        #
        # npix = \int_0^\infty \int_0^\infty |x y| \exp\left[-((x/a)^2+(y/a)^2)/2\right]\,dx\,dy
        #      = 2 a^2 / pi
        npix=np.square(2.25) / (4 * np.pi * np.log(2)),
        # Assume PSF photometry, so we recover all of the flux.
        aperture_correction=1,
        # "This is Nyquist sampled by the 1 arcsec pixels."
        plate_scale=1 * u.arcsec**2,
        # Wild guesses
        dark_noise=0.0001 / u.s,
        read_noise=1,
        # "...an effective aperture of 75cm."
        area=np.pi * (0.5 * 75) * u.cm**2,
        bandpasses={
            "NUV": box_for_lo_hi(2030 * u.angstrom, 2700 * u.angstrom),
            "FUV": box_for_lo_hi(1390 * u.angstrom, 1900 * u.angstrom),
        },
        background=ZodiacalBackground(),
    ),
    # UVEX will be in a highly elliptical TESS-like orbit.
    # This is the TESS TLE downloaded from Celestrak at 2024-09-10T00:43:57Z.
    orbit=TLE(
        "1 43435U 18038A   24262.33225493 -.00001052  00000+0  00000+0 0  9993",
        "2 43435  51.7454  60.8303 4593193 124.3403   0.2501  0.07594463  1386",
    ),
    # Sky grid optimized for full coverage of the sky by circles circumscribed
    # within the square field of view (so that each field is fully covered
    # at all roll angles).
    skygrid=skygrid.geodesic(7.7 * u.deg**2, class_="III", base="icosahedron"),
    # Made up slew model.
    slew=Slew(
        max_angular_velocity=0.1 * u.deg / u.s,
        max_angular_acceleration=0.2 * u.deg / u.s**2,
    ),
)
uvex.__doc__ = """UVEX, the UltraViolet EXplorer.

`UVEX <https://www.uvex.caltech.edu/>`_ is a NASA Medium Explorer mission to
map the transient sky in the ultraviolet, expected to launch in 2030. UVEX has
a long-slit spectrograph and a 3.5° square field of view camera with two UV
filter bandpasses.

Note that the bandpasses and parameters are mockups based on the publicly
available description of the mission from the UVEX science paper,
:arXiv:`2111.15608`. They will be replaced with realistic filter bandpasses
when those become publicly available.

Examples
--------

.. plot::
    :include-source: False

    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from astropy import units as u
    from matplotlib import pyplot as plt
    import numpy as np
    from m4opt.missions import uvex
    from m4opt.models import observing
    from synphot import ConstFlux1D, SourceSpectrum

    source_spectrum = SourceSpectrum(ConstFlux1D, amplitude=25 * u.ABmag)
    exptime = np.linspace(0, 900) * u.s
    obstime = Time("2021-10-31")
    with observing(observer_location=uvex.orbit(obstime).earth_location, target_coord=SkyCoord("0deg 0deg"), obstime=obstime):
        snr = uvex.detector.get_snr(exptime, source_spectrum, "NUV")

    ax = plt.axes()
    ax.plot(exptime.to_value(u.s), snr)
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel("S/N")

.. plot::
    :include-source: False

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
