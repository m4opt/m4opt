from importlib import resources

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from regions import PolygonSkyRegion, Regions

from ...constraints import (
    AirmassConstraint,
    AtNightConstraint,
    DeclinationConstraint,
    HourAngleConstraint,
    MoonSeparationConstraint,
)
from ...dynamics import EigenAxisSlew
from ...observer import EarthFixedObserverLocation
from .._core import Mission
from . import data


def _make_fov():
    # Table 1 from Bellm et al. (2019)
    # http://adsabs.harvard.edu/abs/2019PASP..131a8002B
    ns_nchips = 4
    ew_nchips = 4
    ns_npix = 6144
    ew_npix = 6160
    plate_scale = 1.01 * u.arcsec
    ns_chip_gap = 0.205 * u.deg
    ew_chip_gap = 0.140 * u.deg

    # ns_total = ns_nchips * ns_npix * plate_scale + (ns_nchips - 1) * ns_chip_gap
    # ew_total = ew_nchips * ew_npix * plate_scale + (ew_nchips - 1) * ew_chip_gap

    rcid = np.arange(64)

    chipid, rc_in_chip_id = np.divmod(rcid, 4)
    ns_chip_index, ew_chip_index = np.divmod(chipid, ew_nchips)
    ns_rc_in_chip_index = np.where(rc_in_chip_id <= 1, 1, 0)
    ew_rc_in_chip_index = np.where((rc_in_chip_id == 0) | (rc_in_chip_id == 3), 0, 1)

    ew_offsets = (
        ew_chip_gap * (ew_chip_index - (ew_nchips - 1) / 2)
        + ew_npix * plate_scale * (ew_chip_index - ew_nchips / 2)
        + 0.5 * ew_rc_in_chip_index * plate_scale * ew_npix
    )
    ns_offsets = (
        ns_chip_gap * (ns_chip_index - (ns_nchips - 1) / 2)
        + ns_npix * plate_scale * (ns_chip_index - ns_nchips / 2)
        + 0.5 * ns_rc_in_chip_index * plate_scale * ns_npix
    )

    ew_ccd_corners = 0.5 * plate_scale * np.asarray([ew_npix, 0, 0, ew_npix])
    ns_ccd_corners = 0.5 * plate_scale * np.asarray([ns_npix, ns_npix, 0, 0])

    coords = SkyCoord(
        ew_offsets[:, np.newaxis] + ew_ccd_corners[np.newaxis, :],
        ns_offsets[:, np.newaxis] + ns_ccd_corners[np.newaxis, :],
    )
    return Regions([PolygonSkyRegion(coord) for coord in coords])


def _read_skygrid():
    table = Table.read(
        resources.files(data) / "ZTF_Fields.txt",
        format="ascii.fixed_width_no_header",
        delimiter=" ",
        comment="%",
    )
    return SkyCoord(table["col2"], table["col3"], unit=u.deg)


ztf = Mission(
    name="ztf",
    fov=_make_fov(),
    constraints=(
        AirmassConstraint(2.5)
        & AtNightConstraint.twilight_astronomical()
        & MoonSeparationConstraint(25 * u.deg)
        &
        # The rest of these are positional constraints from
        # https://github.com/ZwickyTransientFacility/ztf_sim/blob/5176ebaa9e1f8e5448593df4102d077c0e880886/ztf_sim/QueueManager.py#L1235-L1255
        HourAngleConstraint(-5.95 * u.hourangle, 5.95 * u.hourangle)
        & (
            HourAngleConstraint(-17.6 * u.deg, 180 * u.deg)
            | DeclinationConstraint(-22 * u.deg, 90 * u.deg)
        )
        & (
            HourAngleConstraint(-180 * u.deg, -17.6 * u.deg)
            | DeclinationConstraint(-45 * u.deg, 90 * u.deg)
        )
        & (
            HourAngleConstraint(-3 * u.deg, 3 * u.deg)
            | DeclinationConstraint(-46 * u.deg, 90 * u.deg)
        )
        & DeclinationConstraint(-90 * u.deg, 87.5 * u.deg)
    ),
    observer_location=EarthFixedObserverLocation.of_site("Palomar"),
    skygrid=_read_skygrid(),
    # From Section 4.2:
    #
    # > The new servo motors ... drive the HA axis at 0.4°/s^2 acceleration and
    # > 2.5°/s maximum velocity and the decl. axis at 0.5°/s&2 acceleration and
    # > 3°/s maximum velocity, about twice the speed of the previous drives.
    #
    # FIXME: Implement non-uniform slew rate about different axes.
    slew=EigenAxisSlew(2.5 * u.deg / u.s, 0.4 * u.deg / u.s**2),
)
ztf.__doc__ = """Zwicky Transient Facility (ZTF).

`ZTF <https://www.ztf.caltech.edu>`_ is a ground-based optical survey with a 47
square degree camera on the Samuel Oschin 48 Inch Telescope at Palomar
Observatory.

The FOV region precisely models the layout of ZTF's :math:`4 \times 4` CCD
mosaic including chip gaps as described in Table 1 of
:footcite:`2019PASP..131a8002B`.

References
----------
.. footbibliography::
"""
