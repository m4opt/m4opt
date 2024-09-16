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
from ..models.background import GalacticBackground, ZodiacalBackground
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
        # "...an effective aperture of 75cm."
        area=np.pi * np.square(0.5 * 75 * u.cm),
        bandpasses={
            "NUV": box_for_lo_hi(2030 * u.angstrom, 2700 * u.angstrom),
            "FUV": box_for_lo_hi(1390 * u.angstrom, 1900 * u.angstrom),
        },
        background=GalacticBackground() + ZodiacalBackground(),
        # These are made-up numbers that happen to make median limiting
        # magnitudes agree with https://www.uvex.caltech.edu/page/for-astronomers.
        gain=0.08,
        read_noise=10,
        dark_noise=1e-1 * u.Hz,
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
    :caption: Median limiting magnitude, averaged over target coordinates and observation time.

    from astropy import units as u
    from astropy.coordinates import EarthLocation, ICRS
    from astropy_healpix import HEALPix
    from astropy.time import Time
    from matplotlib import pyplot as plt
    from m4opt.missions import uvex
    from m4opt.models import observing
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
                    1000 * u.angstrom,
                    SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag),
                    filt,
                ).to_value(u.ABmag)
            )
    median_limmags = np.median(limmags, axis=[2, 3])

    ax = plt.axes()
    ax.set_xlim(1, 10)
    ax.set_ylim(24.5, 26.5)
    ax.invert_yaxis()
    for filt, limmag in zip(uvex.detector.bandpasses.keys(), median_limmags):
        ax.plot(exptime, limmag, "-o", label=f"{filt}, stacked")
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
