import numpy as np
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    SkyCoord,
    UnitSphericalRepresentation,
    get_body,
)
from astropy.time import Time


def nominal_roll(
    observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
) -> u.Quantity[u.physical.angle]:
    """Determine the nominal roll angle for a space telescope.

    Many space telescopes have a gross physical configuration that consists of
    a telescope boresight along the +X axis, solar panels that are on the +Y
    axis (which may be free to rotate around that axis to track the sun), and a
    +Z axis that points away from the sun shield toward the "dark" or "cool"
    side of the spacecraft. See below for an example from Chandra.

    .. figure:: https://cxc.harvard.edu/proposer/POG/html/images/sc-config.png
        :alt: Chandra X-ray Observatory spacecraft coordinate system

        Spacecraft coordinate system typical of most space telescopes.
        Reproduced with permission from SAO/CXC (see
        `original <https://cxc.harvard.edu/proposer/POG/html/chap1.html#tth_sEc1.3>`_).

    In space telescopes with this general physical plan, it is common to prefer
    or require that the roll of the telescope about the boresight places the +Y
    axis perpendicular to the direction of the sun, so that the solar array can
    be oriented for optimal power. This is called the nominal roll angle.

    (It is assumed that when roll=0, the +Z axis points to celestial north.)

    This function determines the nominal roll angle for a spacecraft at a given
    location, observing a given target at a given time.

    Parameters
    ----------
    observer_location:
        Location of the spacecraft.
    target_coord:
        Orientation of the boresight of the telescope.
    obstime:
        The time of the observation.

    Returns
    -------
    :
        The nominal roll angle for the observation.

    Examples
    --------

    .. plot::
        :caption: Nominal roll angle over one year for a selection of ecliptic latitudes
        :include-source: False

        from astropy.coordinates import EarthLocation, GeocentricMeanEcliptic, SkyCoord
        from astropy.time import Time
        from astropy import units as u
        from matplotlib import pyplot as plt
        from m4opt.dynamics import nominal_roll
        import numpy as np

        # Place the observer at the center of the Earth.
        # Note that for observers in Earth orbit, the impact of the orbital phase
        # is negligible because the orbit is much smaller than the Earth-Sun distance.
        observer_location = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
        obstime = Time('2024-01-1') + np.linspace(0, 1, 1000) * u.year

        fig, ax = plt.subplots()
        for lat in np.arange(0, 100, 10) * u.deg:
            target_coord = SkyCoord(0 * u.deg, lat, frame=GeocentricMeanEcliptic)
            roll = nominal_roll(observer_location, target_coord, obstime)
            ax.plot(
                obstime.to_datetime(),
                np.unwrap(roll, period=360 * u.deg),
                label=lat.to_string(format='latex'))
        ax.set_xlabel('Date')
        ax.set_ylabel('Nominal roll angle (°)')
        ax.legend(title='Ecliptic latitude')

    .. plot::
        :caption: Maximum rate of change of nominal roll angle as a function of ecliptic latitude
        :include-source: False

        from astropy.coordinates import EarthLocation, GeocentricMeanEcliptic, SkyCoord
        from astropy.time import Time
        from astropy import units as u
        from matplotlib import pyplot as plt
        from m4opt.dynamics import nominal_roll
        import numpy as np

        # Place the observer at the center of the Earth.
        # Note that for observers in Earth orbit, the impact of the orbital phase
        # is negligible because the orbit is much smaller than the Earth-Sun distance.
        observer_location = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
        obstime = Time('2024-01-1') + np.linspace(0, 1, 1000) * u.year

        lat = np.linspace(0, 90, 2000) * u.deg
        target_coord = SkyCoord(0 * u.deg, lat[:, np.newaxis], frame=GeocentricMeanEcliptic)
        roll = nominal_roll(observer_location, target_coord, obstime)
        d_roll = np.abs(np.diff(np.unwrap(roll, period=360 * u.deg, axis=1), axis=1)).max(axis=1)
        d_t = obstime[1] - obstime[0]

        fig, ax = plt.subplots()
        ax.plot(lat, d_roll / d_t)
        ax.set_yscale('log')
        ax.set_xlabel('Ecliptic latitude')
        ax.set_ylabel('Max rate of chang of roll angle (°/day)')
        ax.grid()
        ax.set_xlim(0, 90)
        ax.set_ylim(1, 1e3)

    You can compute the nominal roll angle for a particular observation:

    >>> from astropy.coordinates import GCRS, EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.dynamics import nominal_roll
    >>> observer_location = EarthLocation.from_geocentric(
    ...     6000 * u.km, 8000 * u.km, -3000 * u.km)
    >>> obstime = Time("2024-12-25 12:00:00")
    >>> target_coord = SkyCoord.from_name("NGC 4993")
    >>> roll = nominal_roll(observer_location, target_coord, obstime)
    >>> roll
    <Quantity -72.56082178 deg>

    You can create an appropriately rolled coordinate frame in the spacecraft
    coordinates by using the
    :meth:`~astropy.coordinates.SkyCoord.skyoffset_frame` method:

    >>> obsgeoloc, obsgeovel = observer_location.get_gcrs_posvel(obstime)
    >>> spacecraft_frame = target_coord.transform_to(
    ...     GCRS(obstime=obstime, obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
    ... ).skyoffset_frame(roll)

    """
    sun = get_body("sun", obstime, observer_location)

    v1 = target_coord.transform_to(sun.frame).represent_as(UnitSphericalRepresentation)
    v2 = v1.cross(sun.data).represent_as(UnitSphericalRepresentation)
    v3 = v1.cross(v2).represent_as(UnitSphericalRepresentation)

    # This is a frame with the spatial origin at the observer
    # and the +x axis pointing to the target.
    frame = target_coord.transform_to(sun.frame).skyoffset_frame()
    y = (
        SkyCoord(90 * u.deg, 0 * u.deg, frame=frame)
        .transform_to(sun.frame)
        .represent_as(UnitSphericalRepresentation)
    )
    return np.arctan2(v3.dot(y), v2.dot(y)).to(u.deg)
