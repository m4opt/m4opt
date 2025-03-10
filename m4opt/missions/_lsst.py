import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from regions import CircleSkyRegion
from synphot import Gaussian1D, SpectralElement

from .. import skygrid
from ..constraints import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    MoonSeparationConstraint,
)
from ..dynamics import Slew
from ..observer import EarthFixedObserverLocation
from ..synphot import Detector
from ..synphot.background import GalacticBackground, ZodiacalBackground
from ._core import Mission

lsst = Mission(
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
            "sdssu": SpectralElement(
                Gaussian1D,
                amplitude=0.25,
                mean=3600 * u.angstrom,
                stddev=400 * u.angstrom,
            ),
            "ps1__g": SpectralElement(
                Gaussian1D,
                amplitude=0.49,
                mean=4750 * u.angstrom,
                stddev=750 * u.angstrom,
            ),
            "ps1__r": SpectralElement(
                Gaussian1D,
                amplitude=0.6,
                mean=6250 * u.angstrom,
                stddev=750 * u.angstrom,
            ),
            "ps1__i": SpectralElement(
                Gaussian1D,
                amplitude=0.69,
                mean=7550 * u.angstrom,
                stddev=950 * u.angstrom,
            ),
            "ps1__z": SpectralElement(
                Gaussian1D,
                amplitude=0.69,
                mean=8750 * u.angstrom,
                stddev=750 * u.angstrom,
            ),
            "ps1__y": SpectralElement(
                Gaussian1D,
                amplitude=0.35,
                mean=10000 * u.angstrom,
                stddev=1000 * u.angstrom,
            ),
        },
        background=GalacticBackground() + ZodiacalBackground(),
        read_noise=9,
        dark_noise=0.2 * u.Hz,
        gain=1,
    ),
    # The LSST (Vera C. Rubin Observatory) is a ground-based telescope
    observer_location=EarthFixedObserverLocation.of_site("LSST"),
    # Sky grid optimized for LSST's large field of view.
    skygrid=skygrid.geodesic(9.6 * u.deg**2, class_="III", base="icosahedron"),
    # Slew model tailored for LSST (Vera C. Rubin Observatory), a ground-based telescope in Chile.
    slew=Slew(
        max_angular_velocity=6.3 * u.deg / u.s,
        max_angular_acceleration=5.25 * u.deg / u.s**2,
        settling_time=15 * u.s,
    ),
)
lsst.__doc__ = r"""LSST, the Legacy Survey of Space and Time.

`LSST <https://rubinobservatory.org/>`_ is a ground-based optical survey telescope 
located in Chile as part of the Vera C. Rubin Observatory. It is designed 
to conduct a 10-year survey of the southern sky with a large field of view 
to detect transient events, including potential gravitational wave counterparts 
(:arxiv:`0805.2366`).
"""
