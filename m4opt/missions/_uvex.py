import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
from synphot import Box1D, SpectralElement

from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..models import Detector, DustExtinction
from ..models.background import ZodiacalBackground
from ..orbit import Spice
from ._core import Mission


def box_for_lo_hi(lo, hi):
    return SpectralElement(Box1D, x_0=0.5 * (lo + hi), width=hi - lo)


uvex = Mission(
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
        extinction=DustExtinction(),
    ),
    # UVEX will be in a highly elliptical TESS-like orbit.
    orbit=Spice(
        "MGS SIMULATION",
        "https://archive.stsci.edu/missions/tess/models/TESS_EPH_PRE_LONG_2021252_21.bsp",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00010.tpc",
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
    with observing(observer_location=uvex.orbit(obstime), target_coord=SkyCoord("0deg 0deg"), obstime=obstime):
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
