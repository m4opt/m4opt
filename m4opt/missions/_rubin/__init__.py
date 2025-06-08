from importlib import resources

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
)
rubin.__doc__ = r"""Vera C Rubin Observatory.

The Legacy Survey of Space and Time (LSST) is a 10-year synoptic time-domain
survey of the Southern sky, conducted with the Simonyi Survey Telescope at the
`Vera C. Rubin Observatory <https://rubinobservatory.org>`_. The LSST camera's
focal plane consists of 189 detectors arranged in 21 rafts, as shown in Figure
12 from :footcite:`2019ApJ...873..111I`.

References
----------
.. footbibliography::
"""
