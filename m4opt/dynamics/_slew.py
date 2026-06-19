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
            case1 = (self.max_angular_velocity < va) & (x > sa)
            case2 = (self.max_angular_velocity > va) & (x < sa)
            case3 = (self.max_angular_velocity < va) & (x < sa) & (x > sv)
            case4 = (self.max_angular_velocity < va) & (x < sa) & (x < sv)
            case5 = (self.max_angular_velocity > va) & (x > sa) & (x > sv)
            case6 = (self.max_angular_velocity > va) & (x > sa) & (x < sv)
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
