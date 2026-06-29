from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates.matrix_utilities import rotation_matrix


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


class GroundSlew(ABC):
    """Base class for ground-based telescope slew time models."""

    @abstractmethod
    def time(
        self,
        init_pos: u.Quantity[u.physical.angle],
        fin_pos: u.Quantity[u.physical.angle],
        motion_range: tuple[float, float] | None = None,
    ) -> u.Quantity[u.physical.time]:
        """Calculate the time to execute an optimal slew.

        Parameters
        ----------
        init_pos:
            Initial position of the telescope/dome component.
        fin_pos:
            Final desired position of the telescope/dome component.
        motion_range:
            Range of motion of the telescope/dome component. Defaults
            to None if the component can freely rotate.

        Returns
        -------
        :
            Time to slew between the initial and final positions.
        """


@dataclass
class GroundSlewComponent(GroundSlew):
    """Model slew time for a ground telescope component (e.g. dome along
    the azimuth axis, telescope mount along the altitude axis).

    The slew time is calculated using JPE's third order point-to-point
    motion profile formulae <insert citation here> if jerk is specified.
    Otherwise, the slew time is calculated as specified in EigenAxisSlew.

    Note: code is written for
    """

    max_angular_velocity: u.Quantity[u.physical.angular_velocity]
    """Maximum angular rate."""

    max_angular_acceleration: u.Quantity[u.physical.angular_acceleration]
    """Maximum angular acceleration."""

    max_angular_jerk: u.Quantity | None = None
    """Maximum angular jerk. Defaults to None if a value isn't specified."""

    settling_time: u.Quantity[u.physical.time] = 0 * u.s
    """Time to settle to rest after a slew."""

    @staticmethod
    def separation(
        init_pos: u.Quantity[u.physical.angle],
        fin_pos: u.Quantity[u.physical.angle],
        motion_range: tuple[float, float] | None = None,
    ) -> u.Quantity[u.physical.angle]:
        """
        Determine the smallest angular separation between an initial and final position.

        Parameters
        ----------
        init_pos:
            Initial position of the telescope/dome component. Must be
            within the bounds of motion_range (if not freely rotating).
        fin_pos:
            Final desired position of the telescope/dome component. Does
            not necessarily have to be within the bounds of motion_range.
            (E.g. if rotation is allowed between -270 and 270 degrees, a
            desired position of 300 degrees is still allowed.)
        motion_range:
            Range of motion of the telescope/dome component in degrees.
            Defaults to None if the component can freely rotate. Assumes
            continuity between lower and upper bounds.

        Returns
        -------
        min_dist:
            Shortest possible angular separation of the two positions.
        """
        if not init_pos.unit.is_equivalent(fin_pos.unit):
            raise ValueError(
                "Initial and final position values should be in the same angular units."
            )

        full_rot = 360 * u.deg

        # Free Rotation Case
        if motion_range is None:
            min_dist = Angle(fin_pos - init_pos).wrap_at(180 * u.deg)

        # Limited Rotation Case
        else:
            lo, hi = u.Quantity(motion_range, u.deg)
            init_pos = np.atleast_1d(init_pos)
            fin_pos = np.atleast_1d(fin_pos)
            if init_pos.shape != fin_pos.shape:
                raise ValueError("init_pos and fin_pos must have matching shapes.")
            if (np.any(init_pos < lo)) or (np.any(init_pos > hi)):
                raise ValueError(
                    "Current positions and/or bounds are entered incorrectly. At least one position is out of bounds."
                )
            min_dist = np.empty(init_pos.shape) * init_pos.unit
            for idx in np.ndindex(init_pos.shape):
                position = fin_pos[idx]
                lowest = position - full_rot * (np.floor((position - lo) / full_rot))
                possible = u.Quantity(
                    np.arange(
                        lowest / position.unit,
                        hi / position.unit,
                        full_rot / position.unit,
                    )
                    * position.unit,
                    u.deg,
                )
                if len(possible) == 0:
                    raise ValueError(
                        f"No reachable solution for target position {position}."
                    )
                diff = possible - init_pos[idx]
                min_dist[idx] = diff[np.argmin(np.abs(diff))]
        return min_dist if min_dist.size > 1 else min_dist.flatten()[0]

    @staticmethod
    def _movetime(
        dist: u.Quantity[u.physical.angle],
        vmax: u.Quantity[u.physical.angular_velocity],
        amax: u.Quantity[u.physical.angular_acceleration],
        jmax: u.Quantity | None = None,
        settling_time: u.Quantity[u.physical.time] = 0 * u.s,
    ) -> u.Quantity[u.physical.time]:
        """See https://www.jpe-innovations.com/precision-point/third-order-point-to-point-motion-profile/"""
        dist = np.abs(dist)  # ensures positive distance
        total_time = u.Quantity(np.zeros(dist.shape), u.s)

        # Infinite Jerk Case
        if jmax is None:
            xcrit = np.square(vmax) / amax
            total_time = np.where(
                dist <= xcrit, 2 * np.sqrt(dist / amax), vmax / amax + dist / vmax
            )

        # Finite Jerk Case
        else:
            va = amax**2 / jmax
            sa = 2 * amax**3 / jmax**2
            if vmax * jmax < amax**2:
                sv = vmax * 2 * np.sqrt(vmax / jmax)
            else:
                sv = vmax * (vmax / amax + amax / jmax)
            va = va.to(vmax.unit)
            sa = sa.to(dist.unit)
            sv = sv.to(dist.unit)
            case1 = (vmax < va) & (dist > sa)
            case2 = (vmax > va) & (dist < sa)
            case3 = (vmax < va) & (dist < sa) & (dist > sv)
            case4 = (vmax < va) & (dist < sa) & (dist < sv)
            case5 = (vmax > va) & (dist > sa) & (dist > sv)
            case6 = (vmax > va) & (dist > sa) & (dist < sv)
            tj13, tv13 = np.sqrt(vmax / jmax), dist / vmax
            ta13 = tj13
            t13 = tj13 + tv13 + ta13
            tj24 = np.cbrt(0.5 * dist / jmax)
            tv24 = 2 * tj24
            ta24 = tj24
            t24 = tj24 + tv24 + ta24
            tj5, ta5, tv5 = amax / jmax, vmax / amax, dist / vmax
            t5 = tj5 + tv5 + ta5
            tj6, ta6 = (
                tj5,
                0.5
                * (
                    np.sqrt((4 * dist * jmax**2 + amax**3) / (amax * jmax**2))
                    - amax / jmax
                ),
            )
            tv6 = ta6 + tj6
            t6 = tj6 + ta6 + tv6
            total_time = np.where(case1 | case3, t13, total_time)
            total_time = np.where(case2 | case4, t24, total_time)
            total_time = np.where(case5, t5, total_time)
            total_time = np.where(case6, t6, total_time)
        total_time = u.Quantity(total_time, u.s)
        total_time = np.where(
            total_time != 0 * u.s, total_time + settling_time, total_time
        )
        return u.Quantity(total_time, u.s)

    @override
    def time(
        self,
        init_pos: u.Quantity[u.physical.angle],
        fin_pos: u.Quantity[u.physical.angle],
        motion_range: tuple[float, float] | None = None,
    ) -> u.Quantity[u.physical.time]:
        """Calculate the time to execute an optimal slew.

        Parameters
        ----------
        init_pos:
            Initial position of the telescope/dome component.
        fin_pos:
            Final desired position of the telescope/dome component.
        motion_range:
            Range of motion of the telescope/dome component. Defaults
            to None if the component can freely rotate.

        Returns
        -------
        slew_time:
            Time to slew between the initial and final positions.
        """
        sep = self.separation(init_pos, fin_pos, motion_range)
        slew_time = self._movetime(
            sep,
            self.max_angular_velocity,
            self.max_angular_acceleration,
            self.max_angular_jerk,
            self.settling_time,
        )
        return slew_time
