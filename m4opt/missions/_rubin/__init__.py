from importlib import resources

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from regions import RectangleSkyRegion, Regions

from ... import skygrid
from ...constraints import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    MoonSeparationConstraint,
)
from ...dynamics import AltAzSlew, SlewComponent
from ...observer import EarthFixedObserverLocation
from ...synphot import Detector, bandpass_from_svo
from ...synphot.background import SkyBackground, ZodiacalBackground
from .._core import Mission
from . import data


def _make_fov():
    """Generate LSST FOV as rectangular sky regions from detector positions."""
    file_path = resources.files(data) / "lsstCamSim.yaml"
    with file_path.open() as file:
        yaml_data = yaml.safe_load(file)
    cams = Table(list(yaml_data["CCDs"].values()))

    PLATE_SCALE = 0.2 * u.arcsec
    return Regions(
        [
            RectangleSkyRegion(
                SkyCoord(*(row["offset"][:2] * PLATE_SCALE / row["pixelSize"])),
                *(row["bbox"][1] * PLATE_SCALE),
            )
            for row in cams
            if row["detectorType"] == 0  # Science only detectors
        ]
    )


# Initialize Components for Rubin's Slew Model
rubin_loc = EarthLocation(
    lat=-30.244633 * u.deg, lon=-70.749417 * u.deg, height=2647 * u.m
)
mount_alt = SlewComponent(
    max_angular_velocity=3.5 * u.deg / u.s,
    max_angular_acceleration=3.5 * u.deg / u.s**2,
    max_angular_jerk=14.0 * u.deg / u.s**3,
    settling_time=3 * u.s,
)
mount_az = SlewComponent(
    max_angular_velocity=7 * u.deg / u.s,
    max_angular_acceleration=7 * u.deg / u.s**2,
    max_angular_jerk=28 * u.deg / u.s**3,
    settling_time=3 * u.s,
)
dome_alt = SlewComponent(
    max_angular_velocity=1.75 * u.deg / u.s,
    max_angular_acceleration=0.75 * u.deg / u.s**2,
    max_angular_jerk=3 * u.deg / u.s**3,
)
dome_az = SlewComponent(
    max_angular_velocity=1.5 * u.deg / u.s,
    max_angular_acceleration=0.875 * u.deg / u.s**2,
    max_angular_jerk=3.5 * u.deg / u.s**3,
    settling_time=1 * u.s,
)

rubin = Mission(
    name="rubin",
    fov=_make_fov(),
    constraints=(
        AirmassConstraint(2.5)
        & AltitudeConstraint(20 * u.deg, 85 * u.deg)
        & AtNightConstraint.twilight_astronomical()
        & MoonSeparationConstraint(30 * u.deg)
    ),
    observer_location=EarthFixedObserverLocation(EarthLocation.of_site("LSST")),
    # Sky grid optimized for LSST’s large field of view.
    skygrid=skygrid.geodesic(3.5 * u.deg**2, class_="III", base="icosahedron"),
    slew=AltAzSlew(
        comp1=mount_alt,
        comp2=mount_az,
        comp3=dome_alt,
        comp4=dome_az,
        location=rubin_loc,
    ),
    # Parameters from SMTN-002: https://smtn-002.lsst.io/
    detector=Detector(
        # Effective clear aperture diameter: 6.423 m
        area=np.pi * np.square(0.5 * 6.423 * u.m),
        plate_scale=(0.2 * u.arcsec) ** 2,
        # Combined camera read noise requirement
        read_noise=8.8,
        # "gain can safely be assumed to be 1" for SNR calculations
        gain=1,
        # Dark current requirement: 0.2 e-/s/pixel
        # Table 4 of https://smtn-002.lsst.io/
        dark_noise=0.2 * u.Hz,
        bandpasses={band: bandpass_from_svo(f"LSST/LSST.{band}") for band in "ugrizy"},
        background=SkyBackground.medium() + ZodiacalBackground(),
    ),
)
rubin.__doc__ = r"""Vera C Rubin Observatory.

The Legacy Survey of Space and Time (LSST) is a 10-year synoptic time-domain
survey of the Southern sky, conducted with the Simonyi Survey Telescope at the
`Vera C. Rubin Observatory <https://rubinobservatory.org>`_. The LSST camera's
focal plane consists of 189 detectors arranged in 21 rafts, as shown in Figure
12 from :footcite:`2019ApJ...873..111I`.

Note
----
Detector parameters are from :doc:`SMTN-002 <smtn-002:index>`:

- Effective clear aperture diameter: 6.423 m
- Plate scale: 0.2 arcsec/pixel
- Read noise: 8.8 e- (combined camera requirement)
- Gain: 1 (recommended for SNR calculations)
- Dark current: 0.2 e-/s/pixel

References
----------
.. footbibliography::

Examples
--------

.. plot::
    :include-source: False
    :caption: Rubin limiting magnitude vs. exposure time at zenith.

    from astropy import units as u
    from astropy.coordinates import AltAz, SkyCoord
    from astropy.time import Time
    from matplotlib import pyplot as plt
    from m4opt.missions import rubin
    from m4opt.synphot import observing
    import numpy as np
    from synphot import ConstFlux1D, SourceSpectrum

    exptime = np.arange(30, 330, 30) * u.s
    obstime = Time("2025-03-19T07:00:00")
    loc = rubin.observer_location(obstime)
    frame = AltAz(location=loc, obstime=obstime)
    coord = SkyCoord(alt=90 * u.deg, az=0 * u.deg, frame=frame)

    ax = plt.axes()
    with observing(loc, coord, obstime):
        for filt in rubin.detector.bandpasses.keys():
            limmag = rubin.detector.get_limmag(
                5,
                exptime,
                SourceSpectrum(ConstFlux1D, amplitude=0 * u.ABmag),
                filt,
            )
            ax.plot(exptime, limmag, "-o", label=filt)
    ax.invert_yaxis()
    ax.legend()
    ax.set_xlabel("Exposure time (s)")
    ax.set_ylabel(r"5-$\sigma$ Limiting magnitude (AB)")
"""
