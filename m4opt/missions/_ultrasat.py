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
    GalacticLatitudeConstraint
)
from ..models import Detector
from ..models.background import GalacticBackground, ZodiacalBackground
from ..orbit import TLE
from ..utils.dynamics import Slew
from ._core import Mission

ultrasat = Mission(
    name="ultrasat",
    fov=RectangleSkyRegion(
        center=SkyCoord(0 * u.deg, 0 * u.deg), width=14.28 * u.deg, height=14.28 * u.deg
    ),
    constraints=[
        EarthLimbConstraint(28 * u.deg),
        SunSeparationConstraint(46 * u.deg),
        MoonSeparationConstraint(23 * u.deg),
       # GalacticLatitudeConstraint(10 * u.deg),
    ],
    detector=Detector(
        # Total number of pixels
        npix=89.9e6,
        # Pixel scale, 5.4 arcsec / pixel
        plate_scale=(5.4 * u.arcsec)**2,

        # Area of an effective aperture of is 33 cm,
        area=np.pi * np.square(0.5 * 33 * u.cm),
        bandpasses={
            "NUV": SpectralElement(
                Gaussian1D,
                amplitude=0.25,
                mean=2600 * u.angstrom,
                stddev=340 * u.angstrom,
            ),
        },
        # Zodiacal light, Cerenkov radiation, and Stray light dominate ULTRASAT’s background noise.
        background=GalacticBackground() + ZodiacalBackground(),
        # Made up to match plot
        read_noise=6,
        dark_noise=12/300 * u.Hz,
        gain=1,
    ),
    
    # ULTRASAT will be in a geosynchronous orbit similar to GOES-17.
    # This is the TESS TLE downloaded from Celestrak at 2024-11-15T09:15:20Z.
    # https://celestrak.org/NORAD/elements/weather.txt 
    
    orbit=TLE(
        "1 43226U 18022A   24320.05692005 -.00000082  00000+0  00000+0 0  9997",
        "2 43226   0.0007  47.5006 0003498 198.5164  84.4417  1.00271931 24622",
    ),

    # Sky grid optimized for ULTRASAT's wide field of view.
    skygrid=skygrid.healpix(200 * u.deg**2),
    # Slew model tailored for ULTRASAT's operational requirements.
    slew=Slew(
        max_angular_velocity=0.872 * u.deg / u.s,
        max_angular_acceleration=0.244 * u.deg / u.s**2,
    ),
)
ultrasat.__doc__ = r"""ULTRASAT, the Ultraviolet Transient Astronomy Satellite.

`ULTRASAT <http://www.weizmann.ac.il/ultrasat>`_ is a proposed Israeli ultraviolet
satellite designed to monitor the transient sky with a wide-field imager (arXiv:2304.14482).
Expecte to launch in  2027, ULTRASAT aims to provide continuous monitoring of
large areas of the sky to detect and study transient astronomical events in the
ultraviolet spectrum.
"""