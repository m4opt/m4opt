from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates.matrix_utilities import rotation_matrix


def matrix_trace(matrix):
    return np.trace(matrix, axis1=-2, axis2=-1)


@dataclass
class Slew:
    """Model the slew time for a spacecraft.

    The optimal slew consists of an acceleration phase at the maximum
    acceleration, possibly a coasting phase at the maximum angular velocity,
    and a deceleration phase at the maximum acceleration.
    """

    max_angular_velocity: u.Quantity[u.physical.angular_velocity]
    """Maximum angular rate."""

    max_angular_acceleration: u.Quantity[u.physical.angular_acceleration]
    """Maximum angular acceleration."""

    @staticmethod
    def _time(
        x: u.Quantity[u.physical.angle],
        v: u.Quantity[u.physical.angular_velocity],
        a: u.Quantity[u.physical.angular_acceleration],
    ) -> u.Quantity[u.physical.time]:
        xc = np.square(v) / a
        return np.where(x <= xc, np.sqrt(4 * x / a), (x + xc) / v)

    def time(
        self,
        center1: SkyCoord,
        center2: SkyCoord,
        roll1: u.Quantity[u.physical.angle] = 0 * u.rad,
        roll2: u.Quantity[u.physical.angle] = 0 * u.rad,
    ):
        """Calculate the time to execute an optimal slew."""
        return self._time(
            self.separation(center1, center2, roll1, roll2),
            self.max_angular_velocity,
            self.max_angular_acceleration,
        )

    @staticmethod
    def separation(
        center1: SkyCoord,
        center2: SkyCoord,
        roll1: u.Quantity[u.physical.angle] = 0 * u.rad,
        roll2: u.Quantity[u.physical.angle] = 0 * u.rad,
    ) -> u.Quantity[u.physical.angle]:
        """
        Determine the smallest angle to slew between two attitudes.

        Examples
        --------
        >>> from astropy.coordinates import SkyCoord
        >>> from astropy import units as u
        >>> from m4opt.utils.dynamics import Slew
        >>> c1 = SkyCoord(0 * u.deg, 20 * u.deg)
        >>> c2 = SkyCoord(0 * u.deg, 40 * u.deg)
        >>> roll1 = 20 * u.deg
        >>> roll2 = 40 * u.deg
        >>> Slew.separation(c1, c2)
        <Angle 20. deg>
        >>> Slew.separation(c1, c1, roll1, roll2)
        <Angle 20. deg>
        >>> Slew.separation(c1, c2, roll1, roll2)
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
