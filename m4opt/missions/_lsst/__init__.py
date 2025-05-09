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
from .camera import LSSTCameraFOV

lsst_fov = LSSTCameraFOV()

lsst = Mission(
    name="lsst",
    fov=lsst_fov.make_fov(),
    constraints=[
        AirmassConstraint(2.5),
        AltitudeConstraint(20 * u.deg, 85 * u.deg),
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
    # The LSST (Vera C. Rubin Observatory) is a ground-based telescope in Chile.
    observer_location=EarthFixedObserverLocation.of_site("LSST"),
    # Sky grid optimized for LSST’s large field of view.
    skygrid=skygrid.geodesic(3.5 * u.deg**2, class_="III", base="icosahedron"),
    # Slew model tailored for LSST (Vera C. Rubin Observatory), a ground-based telescope in Chile.
    # FIXME: The Telescope Mount Assembly is faster than the dome for long slews.
    # Therefore, we use the dome setup instead of the slew model
    # https://github.com/lsst/rubin_scheduler/blob/main/rubin_scheduler/scheduler/model_observatory/kinem_model.py#L232-L233
    slew=EigenAxisSlew(
        max_angular_velocity=1.5 * u.deg / u.s,
        max_angular_acceleration=0.75 * u.deg / u.s**2,
        settling_time=1 * u.s,
    ),
)
lsst.__doc__ = r"""LSST, the Legacy Survey of Space and Time.

`LSST <https://rubinobservatory.org/>`_ is a ground-based optical survey telescope 
located in Chile as part of the Vera C. Rubin Observatory. It is designed 
to conduct a 10-year survey of the southern sky with a large field of view 
to detect transient events, including potential gravitational wave counterparts 
(:footcite `2019ApJ...873..111I`).

The LSST camera's focal plane consists of 189 detectors arranged in 21 rafts, 
each containing a :math:`3 \times 3` CCD, placed in a :math:`5 \times 5` grid, 
as shown in `Figure 12  from :footcite:`2019ApJ...873..111I`. 
We have two types of detectors: `ITL`, `E2V`.

"""
