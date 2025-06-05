import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import RectangleSkyRegion
from synphot import Gaussian1D, SpectralElement

from .. import skygrid
from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..dynamics import EigenAxisSlew
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
