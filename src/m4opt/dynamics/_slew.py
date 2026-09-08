from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    AltAz,
    Angle,
    EarthLocation,
    SkyCoord,
)
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.time import Time


def matrix_trace(matrix):
    return np.trace(matrix, axis1=-2, axis2=-1)


# FIXME: drop if https://github.com/astropy/astropy/pull/19923 is merged
u.def_physical_type(u.rad / u.s**3, {"angular jerk", "angular jolt"})


@dataclass
class AngularMotionProfile:
    """
    Angular motion profile model.

    This is a model of a general S-curve motion profile with optional limits
    on angular velocity, acceleration, and jerk. The time is solved using a
    `general third-order point-to-point motion profile`__.

    __ https://www.jpe-innovations.com/precision-point/third-order-point-to-point-motion-profile/
    """

    max_angular_velocity: u.Quantity[u.physical.angular_velocity]
    """Maximum angular rate."""

    max_angular_acceleration: u.Quantity[u.physical.angular_acceleration]
    """Maximum angular acceleration."""

    max_angular_jerk: u.Quantity[u.physical.angular_jerk] = np.inf * u.rad / u.s**3
    """Maximum angular jerk."""

    settling_time: u.Quantity[u.physical.time] = 0 * u.second
    """Time to settle to rest after a slew."""

    def _time(
        self,
        x: u.Quantity[u.physical.angle],
    ) -> u.Quantity[u.physical.time]:
        if np.isposinf(self.max_angular_jerk):
            xc = np.square(self.max_angular_velocity) / self.max_angular_acceleration
            return (
                np.where(
                    x <= xc,
                    2 * np.sqrt(x / self.max_angular_acceleration),
                    (x + xc) / self.max_angular_velocity,
                )
                + self.settling_time
            )
        else:
            total_time = u.Quantity(np.zeros(x.shape), u.s)
            va = self.max_angular_acceleration**2 / self.max_angular_jerk
            sa = 2 * self.max_angular_acceleration**3 / self.max_angular_jerk**2
            if (
                self.max_angular_velocity * self.max_angular_jerk
                < self.max_angular_acceleration**2
            ):
                sv = (
                    self.max_angular_velocity
                    * 2
                    * np.sqrt(self.max_angular_velocity / self.max_angular_jerk)
                )
            else:
                sv = self.max_angular_velocity * (
                    self.max_angular_velocity / self.max_angular_acceleration
                    + self.max_angular_acceleration / self.max_angular_jerk
                )
            case1 = (self.max_angular_velocity < va) & (x >= sa)
            case2 = (self.max_angular_velocity >= va) & (x < sa)
            case3 = (self.max_angular_velocity < va) & (x < sa) & (x >= sv)
            case4 = (self.max_angular_velocity < va) & (x < sa) & (x < sv)
            case5 = (self.max_angular_velocity >= va) & (x >= sa) & (x >= sv)
            case6 = (self.max_angular_velocity >= va) & (x >= sa) & (x < sv)
            tj13, tv13 = (
                np.sqrt(self.max_angular_velocity / self.max_angular_jerk),
                x / self.max_angular_velocity,
            )
            ta13 = tj13
            t13 = tj13 + tv13 + ta13
            tj24 = np.cbrt(0.5 * x / self.max_angular_jerk)
            tv24 = 2 * tj24
            ta24 = tj24
            t24 = tj24 + tv24 + ta24
            tj5, ta5, tv5 = (
                self.max_angular_acceleration / self.max_angular_jerk,
                self.max_angular_velocity / self.max_angular_acceleration,
                x / self.max_angular_velocity,
            )
            t5 = tj5 + tv5 + ta5
            tj6, ta6 = (
                tj5,
                0.5
                * (
                    np.sqrt(
                        (
                            4 * x * self.max_angular_jerk**2
                            + self.max_angular_acceleration**3
                        )
                        / (self.max_angular_acceleration * self.max_angular_jerk**2)
                    )
                    - self.max_angular_acceleration / self.max_angular_jerk
                ),
            )
            tv6 = ta6 + tj6
            t6 = tj6 + ta6 + tv6
            total_time = np.where(case1 | case3, t13, total_time)
            total_time = np.where(case2 | case4, t24, total_time)
            total_time = np.where(case5, t5, total_time)
            total_time = np.where(case6, t6, total_time)
            return total_time + self.settling_time

    def _distance(self, t: u.Quantity[u.physical.time]) -> u.Quantity[u.physical.angle]:
        """Calculate the distance that can be reached in a given duration."""
        tt = t - self.settling_time
        if np.isposinf(self.max_angular_jerk):
            tc = 2 * self.max_angular_velocity / self.max_angular_acceleration
            return np.where(
                tt < 0 * u.s,
                np.nan,
                np.where(
                    tt < tc,
                    0.25 * self.max_angular_acceleration * tt**2,
                    self.max_angular_velocity * (tt - 0.5 * tc),
                ),
            )
        else:
            case134 = (
                self.max_angular_velocity
                < self.max_angular_acceleration**2 / self.max_angular_jerk
            )
            case256 = (
                self.max_angular_velocity
                >= self.max_angular_acceleration**2 / self.max_angular_jerk
            )
            aoverj = self.max_angular_acceleration / self.max_angular_jerk
            sa = 2 * self.max_angular_acceleration**3 / (self.max_angular_jerk**2)
            sv1 = 2 * np.sqrt(self.max_angular_velocity**3 / self.max_angular_jerk)
            sv2 = self.max_angular_velocity * (
                self.max_angular_velocity / self.max_angular_acceleration + aoverj
            )
            opt1 = tt * self.max_angular_velocity - 2 * np.sqrt(
                self.max_angular_velocity**3 / self.max_angular_jerk
            )
            opt2 = tt**3 * self.max_angular_jerk / 32
            opt3 = (
                tt * self.max_angular_velocity
                - aoverj * self.max_angular_velocity
                - self.max_angular_velocity**2 / self.max_angular_acceleration
            )
            opt4 = (
                0.25 * self.max_angular_acceleration * ((tt - aoverj) ** 2 - aoverj**2)
            )
            case1 = case134 & (opt1 >= sa)
            case2 = case256 & (opt2 < sa)
            case3 = case134 & (opt1 < sa) & (opt1 >= sv1)
            case4 = case134 & (opt2 < sa) & (opt2 < sv1)
            case5 = case256 & (opt3 >= sa) & (opt3 >= sv2)
            case6 = case256 & (opt4 >= sa) & (opt4 < sv2)
            dist = u.Quantity(np.zeros(t.shape), u.deg)
            dist = np.where(case1 | case3, opt1, dist)
            dist = np.where(case2 | case4, opt2, dist)
            dist = np.where(case5, opt3, dist)
            dist = np.where(case6, opt4, dist)
            return dist


class Slew(ABC):
    """Base class for spacecraft slew time models."""

    @abstractmethod
    def time(
        self,
        center1: SkyCoord,
        center2: SkyCoord,
        roll1: u.Quantity[u.physical.angle] = 0 * u.rad,
        roll2: u.Quantity[u.physical.angle] = 0 * u.rad,
    ) -> u.Quantity[u.physical.time]:
        """Calculate the time to execute an optimal slew.

        Parameters
        ----------
        center1:
            Initial boresight position.
        center2:
            Final boresight position.
        roll1:
            Initial roll angle.
        roll2:
            Final roll angle.

        Returns
        -------
        :
            Time to slew between the two orientations.
        """
        raise NotImplementedError


@dataclass
class EigenAxisSlew(Slew, AngularMotionProfile):
    """Model slew time for a spacecraft employing an eigenaxis maneuver.

    An eigenaxis maneuver is a rotation along the path of shortest angular
    separation, about a single axis. The motion profile along that axis is
    provided by :class:`AngularMotionProfile`.

    Notes
    -----
    An eigenaxis maneuver is generally *not* the fastest possible slew
    maneuver, even for a spacecraft with symmetric moment of inertia and
    symmetric torque limits :footcite:`1993JGCD...16..446B`.

    References
    ----------
    .. footbibliography::
    """

    @override
    def time(
        self,
        center1: SkyCoord,
        center2: SkyCoord,
        roll1: u.Quantity[u.physical.angle] = 0 * u.rad,
        roll2: u.Quantity[u.physical.angle] = 0 * u.rad,
    ) -> u.Quantity[u.physical.time]:
        """Calculate the time to execute an optimal slew.

        Parameters
        ----------
        center1:
            Initial boresight position.
        center2:
            Final boresight position.
        roll1:
            Initial roll angle.
        roll2:
            Final roll angle.

        Returns
        -------
        :
            Time to slew between the two orientations.
        """
        return self._time(self.separation(center1, center2, roll1, roll2))

    @staticmethod
    def separation(
        center1: SkyCoord,
        center2: SkyCoord,
        roll1: u.Quantity[u.physical.angle] = 0 * u.rad,
        roll2: u.Quantity[u.physical.angle] = 0 * u.rad,
    ) -> u.Quantity[u.physical.angle]:
        """
        Determine the smallest angle to slew between two attitudes.

        Parameters
        ----------
        center1:
            Initial boresight position.
        center2:
            Final boresight position.
        roll1:
            Initial roll angle.
        roll2:
            Final roll angle.

        Returns
        -------
        :
            Shortest possible angular separation of the two orientations.

        Examples
        --------
        >>> from astropy.coordinates import SkyCoord
        >>> from astropy import units as u
        >>> from m4opt.dynamics import EigenAxisSlew
        >>> c1 = SkyCoord(0 * u.deg, 20 * u.deg)
        >>> c2 = SkyCoord(0 * u.deg, 40 * u.deg)
        >>> roll1 = 20 * u.deg
        >>> roll2 = 40 * u.deg
        >>> EigenAxisSlew.separation(c1, c2)
        <Angle 20. deg>
        >>> EigenAxisSlew.separation(c1, c1, roll1, roll2)
        <Angle 20. deg>
        >>> EigenAxisSlew.separation(c1, c2, roll1, roll2)
        <Angle 28.21208852 deg>

        """
        assert center1.is_equivalent_frame(center2)
        center1 = center1.spherical
        center2 = center2.spherical
        mat = (
            rotation_matrix(roll2 - roll1, "x")
            @ rotation_matrix(-center1.lat, "y")
            @ rotation_matrix(center1.lon - center2.lon, "z")
            @ rotation_matrix(center2.lat, "y")
        )
        return Angle(np.arccos(0.5 * (matrix_trace(mat) - 1)) * u.rad).to(u.deg)


@dataclass
class SlewComponent(AngularMotionProfile):
    """Model slew time of a component of a ground-based telescope.

    Ground-based telescopes can have multiple components with varying
    angular accelerations, jerks, etc. These include components like the
    dome along each axis and telescope mount in each axis. This model
    assumes free/unlimited rotation.
    """

    frame: GCRS | AltAz | None = None

    def separation(
        self,
        init_pos: u.Quantity[u.physical.angle],
        fin_pos: u.Quantity[u.physical.angle],
    ) -> u.Quantity[u.physical.angle]:
        """
        Determine the angular separation between an initial and final
        position.

        Parameters
        ----------
        init_pos:
            Initial position of the telescope/dome component.
        fin_pos:
            Final desired position of the telescope/dome component.

        Returns
        -------
        separation:
            Angular separation of the two positions.
        """

        separation = np.abs(Angle(fin_pos - init_pos).wrap_at(180 * u.deg))
        return separation

    def time(
        self,
        init_pos: u.Quantity[u.physical.angle],
        fin_pos: u.Quantity[u.physical.angle],
    ) -> u.Quantity[u.physical.time]:
        """
        Determine the slew time between an initial and final position.

        Parameters
        ----------
        init_pos:
            Initial position of the telescope/dome component.
        fin_pos:
            Final desired position of the telescope/dome component.

        Returns
        -------
        :
            Time to slew between the two positions.
        """
        return self._time(self.separation(init_pos, fin_pos))


@dataclass
class GroundSlew:
    """Model slew time of a ground-based telescope that may have
    components with pointing limits.

    This class calculates a telescope's slew time between two positions
    by taking the maximum of that of each individual component. If the
    telescope is AltAz, a location must be specified.
    """

    location: EarthLocation
    """Location of the observatory."""

    comp1: SlewComponent
    comp2: SlewComponent
    comp3: SlewComponent | None = None
    comp4: SlewComponent | None = None
    """Components of the telescope separated by part and axis (e.g. mount 
    in altitude, mount in azimuth, dome in altitude, and dome in azimuth 
    may make up the four components). Only two are required. Odd 
    components must be along the RA or Alt axes while even components 
    must be along the Dec or Az axes."""


class AltAzSlew(GroundSlew):
    def time(
        self,
        coord1: SkyCoord,
        coord2: SkyCoord,
        time_obs: Time,
    ) -> u.Quantity[u.physical.time]:
        """
        Determine the time to slew between two positions based
        on the maximum of each component's slew time.

        Parameters
        ----------
        coord1:
            Initial coordinates.
        coord2:
            Final coordinates.
        time_obs:
            Time of observation.

        Returns
        -------
        slew_time:
            Time to slew between the two orientations based
            on the maximum slew time across all components.
        """
        altaz_frame = AltAz(obstime=time_obs, location=self.location)
        altaz_coord1 = coord1.transform_to(altaz_frame)
        altaz_coord2 = coord2.transform_to(altaz_frame)
        time1 = self.comp1.time(altaz_coord1.alt, altaz_coord2.alt)
        time2 = self.comp2.time(altaz_coord1.az, altaz_coord2.az)
        time3 = (
            self.comp3.time(altaz_coord1.alt, altaz_coord2.alt)
            if self.comp3
            else (0 * u.s)
        )
        time4 = (
            self.comp4.time(altaz_coord1.az, altaz_coord2.az)
            if self.comp4
            else (0 * u.s)
        )
        slew_time = np.maximum(np.maximum(time1, time2), np.maximum(time3, time4))
        return slew_time


class EquatorialSlew(GroundSlew):
    def time(
        self,
        coord1: SkyCoord,
        coord2: SkyCoord,
        time_obs: Time,
    ) -> u.Quantity[u.physical.time]:
        """
        Determine the time to slew between two positions based
        on the maximum of each component's slew time.

        Parameters
        ----------
        coord1:
            Initial coordinates.
        coord2:
            Final coordinates.
        time_obs:
            Time of observation.

        Returns
        -------
        slew_time:
            Time to slew between the two orientations based
            on the maximum slew time across all components.
        """
        obsgeoloc, obsgeovel = self.location.get_gcrs_posvel(time_obs)
        gcrs_frame = GCRS(obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
        gcrs_coord1 = coord1.transform_to(gcrs_frame)
        gcrs_coord2 = coord2.transform_to(gcrs_frame)
        time1 = self.comp1.time(gcrs_coord1.ra, gcrs_coord2.ra)
        time2 = self.comp2.time(gcrs_coord1.dec, gcrs_coord2.dec)
        time3 = (
            self.comp3.time(gcrs_coord1.ra, gcrs_coord2.ra) if self.comp3 else (0 * u.s)
        )
        time4 = (
            self.comp4.time(gcrs_coord1.dec, gcrs_coord2.dec)
            if self.comp4
            else (0 * u.s)
        )
        slew_time = np.maximum(np.maximum(time1, time2), np.maximum(time3, time4))
        return slew_time


class MixedCoordSlew(GroundSlew):
    def time(
        self,
        coord1: SkyCoord,
        coord2: SkyCoord,
        time_obs: Time,
    ) -> u.Quantity[u.physical.time]:
        """
        Determine the time to slew between two positions based
        on the maximum of each component's slew time.

        Parameters
        ----------
        coord1:
            Initial coordinates.
        coord2:
            Final coordinates.
        time_obs:
            Time of observation.

        Returns
        -------
        slew_time:
            Time to slew between the two orientations based
            on the maximum slew time across all components.
        """
        # GCRS Coordinates
        obsgeoloc, obsgeovel = self.location.get_gcrs_posvel(time_obs)
        gcrs_frame = GCRS(obsgeoloc=obsgeoloc, obsgeovel=obsgeovel)
        gcrs_coord1 = coord1.transform_to(gcrs_frame)
        gcrs_coord2 = coord2.transform_to(gcrs_frame)
        # AltAz Coordinates
        altaz_frame = AltAz(obstime=time_obs, location=self.location)
        altaz_coord1 = coord1.transform_to(altaz_frame)
        altaz_coord2 = coord2.transform_to(altaz_frame)
        times = []
        components = [self.comp1, self.comp2, self.comp3, self.comp4]
        for i, comp in enumerate(components):
            time = 0 * u.s
            if comp and comp.frame is AltAz:
                if (i == 0) or (i == 2):
                    time = comp.time(altaz_coord1.alt, altaz_coord2.alt)
                if (i == 1) or (i == 3):
                    time = comp.time(altaz_coord1.az, altaz_coord2.az)
            if comp and comp.frame is GCRS:
                if (i == 0) or (i == 2):
                    time = comp.time(gcrs_coord1.ra, gcrs_coord2.ra)
                if (i == 1) or (i == 3):
                    time = comp.time(gcrs_coord1.dec, gcrs_coord2.dec)
            times.append(time)
        slew_time = np.maximum(
            np.maximum(times[0], times[1]), np.maximum(times[2], times[3])
        )
        return slew_time
