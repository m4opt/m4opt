import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from synphot import Gaussian1D, SpectralElement

from ..constraints import (
    EarthLimbConstraint,
    MoonSeparationConstraint,
    SunSeparationConstraint,
)
from ..synphot import Detector
from ..synphot.background import GalacticBackground, ZodiacalBackground
from ._core import Mission

ultrasat = Mission(
    name="lsst",
    fov=CircleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), radius=1.75 * u.deg),
    constraints=[
        EarthLimbConstraint(48 * u.deg),
        SunSeparationConstraint(70 * u.deg),
        MoonSeparationConstraint(35 * u.deg),
    ],
    detector=Detector(
        npix=4 * np.pi,
        plate_scale=(0.2 * u.arcsec) ** 2,
        # Circular aperture with a diameter of 6.423 m
        area=np.pi * np.square(0.5 * 6.423 * 100 * u.cm),
        bandpasses={
            "NUV": SpectralElement(
                Gaussian1D,
                amplitude=0.25,
                mean=2600 * u.angstrom,
                stddev=340 * u.angstrom,
            ),
        },
        background=GalacticBackground() + ZodiacalBackground(),
        read_noise=9,
        dark_noise=0.2 * u.Hz,
        gain=1,
    ),
)
