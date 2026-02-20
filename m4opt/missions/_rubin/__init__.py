from importlib import resources

import numpy as np
import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import RectangleSkyRegion, Regions

from ... import skygrid
from ...constraints import (
    AirmassConstraint,
    AltitudeConstraint,
    AtNightConstraint,
    MoonSeparationConstraint,
)
from ...dynamics import EigenAxisSlew
from ...observer import EarthFixedObserverLocation
from .._core import Mission
from ...synphot import Detector, bandpass_from_svo
from ...synphot.background import SkyBackground, ZodiacalBackground
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


rubin = Mission(
    name="rubin",
    fov=_make_fov(),
    constraints=(
        AirmassConstraint(2.5)
        & AltitudeConstraint(20 * u.deg, 85 * u.deg)
        & AtNightConstraint.twilight_astronomical()
        & MoonSeparationConstraint(30 * u.deg)
    ),
    observer_location=EarthFixedObserverLocation.of_site("LSST"),
    # Sky grid optimized for LSST’s large field of view.
    skygrid=skygrid.geodesic(3.5 * u.deg**2, class_="III", base="icosahedron"),
    # FIXME: The Telescope Mount Assembly is faster than the dome for long slews.
    # Therefore, we use the dome setup instead of the slew model
    # https://github.com/lsst/rubin_scheduler/blob/main/rubin_scheduler/scheduler/model_observatory/kinem_model.py#L232-L233
    slew=EigenAxisSlew(
        max_angular_velocity=1.5 * u.deg / u.s,
        max_angular_acceleration=0.75 * u.deg / u.s**2,
        settling_time=1 * u.s,
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
Detector parameters are from `SMTN-002 <https://smtn-002.lsst.io/>`_
("Calculating LSST limiting magnitudes and SNR"):

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
