import numpy as np
from astropy import units as u

from ... import skygrid
from ...constraints import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    MoonSeparationConstraint,
)
from ...dynamics import EigenAxisSlew
from ...observer import EarthFixedObserverLocation
from ...synphot import Detector, bandpass_from_svo
from ...synphot.background import GalacticBackground, ZodiacalBackground
from .._core import Mission
from .lsst_camera import LSSTfieldOfView

lsst_fov = LSSTfieldOfView()

lsst = Mission(
    name="lsst",
    fov=lsst_fov.make_fov(),
    constraints=[
        AirmassConstraint(max=2.5),
        AltitudeConstraint(min=20 * u.deg, max=85 * u.deg),
        AtNightConstraint.twilight_astronomical(),
        MoonSeparationConstraint(30 * u.deg),
    ],
    detector=Detector(
        npix=6,  # https://github.com/lsst/rubin_sim/blob/2bf176a6d98ff4c84c352912c5e0721e330fc217/rubin_sim/skybrightness/sky_model.py#L144C19-L144C26
        plate_scale=(0.2 * u.arcsec) ** 2,
        # Circular aperture with a diameter of 6.423 m
        area=np.pi * np.square(0.5 * 6.423 * 100 * u.cm),
        bandpasses={band: bandpass_from_svo(f"LSST/LSST.{band}") for band in "ugrizy"},
        background=GalacticBackground() + ZodiacalBackground(),
        read_noise=9,
        dark_noise=0.2 * u.Hz,
        gain=1,
    ),
    # The LSST (Vera C. Rubin Observatory) is a ground-based telescope
    observer_location=EarthFixedObserverLocation.of_site("LSST"),
    # Sky grid optimized for LSST's large field of view.
    # FIXME: Add the correct area
    skygrid=skygrid.geodesic(9.6 * u.deg**2, class_="III", base="icosahedron"),
    # Slew model tailored for LSST (Vera C. Rubin Observatory), a ground-based telescope in Chile.
    # FIXME: The slew values differ between the paper and the repository.
    # Verify which source is correct and update accordingly.
    slew=EigenAxisSlew(
        max_angular_velocity=3.5 * u.deg / u.s,
        max_angular_acceleration=3.5 * u.deg / u.s**2,
        settling_time=3 * u.s,
    ),
)
lsst.__doc__ = r"""LSST, the Legacy Survey of Space and Time.

`LSST <https://rubinobservatory.org/>`_ is a ground-based optical survey telescope 
located in Chile as part of the Vera C. Rubin Observatory. It is designed 
to conduct a 10-year survey of the southern sky with a large field of view 
to detect transient events, including potential gravitational wave counterparts 
(:arxiv:`0805.2366`).
"""
