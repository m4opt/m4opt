import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from synphot import Gaussian1D, SpectralElement

from ..constraints import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    MoonSeparationConstraint,
)
from ..dynamics import Slew
from ..synphot import Detector
from ..synphot.background import GalacticBackground, ZodiacalBackground
from ._core import Mission

ultrasat = Mission(
    name="lsst",
    fov=CircleSkyRegion(center=SkyCoord(0 * u.deg, 0 * u.deg), radius=1.75 * u.deg),
    constraints=[
        AirmassConstraint(max=2.5, min=1),
        AltitudeConstraint(min=20 * u.deg, max=85 * u.deg),
        AtNightConstraint.twilight_civil(),
        AtNightConstraint.twilight_astronomical(),
        MoonSeparationConstraint(30 * u.deg),
    ],
    detector=Detector(
        npix=6,  # https://github.com/lsst/rubin_sim/blob/2bf176a6d98ff4c84c352912c5e0721e330fc217/rubin_sim/skybrightness/sky_model.py#L144C19-L144C26
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
    slew=Slew(
        max_angular_velocity=6.3 * u.deg / u.s,
        max_angular_acceleration=5.25 * u.deg / u.s**2,
        settling_time=15 * u.s,
    ),
)
