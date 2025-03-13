from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates.matrix_utilities import rotation_matrix

from ..utils.typing_extensions import override


def matrix_trace(matrix):
    return np.trace(matrix, axis1=-2, axis2=-1)


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
class EigenAxisSlew(Slew):
    """Model slew time for a spacecraft employing an eigenaxis maneuver.

    An eigenaxis maneuver is a rotation along the path of shortest angular
    separation, about a single axis. Assuming that the spaceraft has a maximum
    angular acceleration and angular rate, the fastest possible eigenaxis
    maneuver is a "bang-bang" trajectory consisting of an acceleration phase at
    the maximum acceleration, possibly a coasting phase at the maximum angular
    velocity, and a deceleration phase at the maximum acceleration, as shown in
    the figure below.

    .. plot::
        :include-source: False
        :show-source-link: False

        from matplotlib import pyplot as plt
        import numpy as np

        def cases(*args):
            *args, else_value = args
            while args:
                *args, cond, if_value = args
                else_value = np.where(cond, if_value, else_value)
            return else_value


        fig_width, fig_height = plt.rcParams['figure.figsize']
        scale = 0.5
        fig, axs = plt.subplots(
            3, 2, sharex=True, sharey='row',
            figsize=(2 * scale * fig_width, 3 * scale * fig_height),
            tight_layout=True)

        t = np.linspace(0, 0.6, 1000)
        axs[0, 0].plot(t, cases(t <= 0.2, 1, t <= 0.4, -1, 0))
        axs[1, 0].plot(t, cases(t <= 0.2, t, t <= 0.4, 0.4 - t, 0))
        axs[2, 0].plot(t, cases(t <= 0.2, 0.5 * t**2, t <= 0.4, 0.02 + 0.2 * (t - 0.2) - 0.5 * (t - 0.2)**2, 0.04))
        t = np.linspace(0, 1, 1000)
        axs[1, 1].plot(t, cases(t <= 0.3, t, t <= 0.5, 0.3, t <= 0.8, 0.8 - t, 0))
        axs[0, 1].plot(t, cases(t <= 0.3, 1, t <= 0.5, 0, t <= 0.8, -1, 0))
        axs[2, 1].plot(t, cases(t <= 0.3, 0.5 * t**2, t <= 0.5, 0.045 + 0.3 * (t - 0.3), t <= 0.8, 0.105 + 0.3 * (t - 0.5) - 0.5 * (t - 0.5)**2, 0.15))
        axs[2, 0].set_xlabel('time')
        axs[2, 1].set_xlabel('time')
        axs[0, 0].set_ylabel('acceleration')
        axs[1, 0].set_ylabel('velocity')
        axs[2, 0].set_ylabel('position')
        axs[0, 0].set_title('short slew')
        axs[0, 1].set_title('long slew')

        axs[0, 0].set_yticks([-1, 0, 1])
        axs[0, 0].set_yticklabels(['-|max|', '0', '+|max|'])
        axs[1, 0].set_yticks([0, 0.3])
        axs[1, 0].set_yticklabels(['0', 'max'])
        axs[2, 0].set_yticks([0])
        axs[2, 0].set_yticklabels(['0'])
        axs[2, 0].set_xticks([])
        axs[2, 1].set_xticks([])

    Notes
    -----
    An eigenaxis maneuver is generally *not* the fastest possible slew
    maneuver, even for a spacecraft with symmetric moment of inertia and
    symmetric torque limits :footcite:`1993JGCD...16..446B`.

    References
    ----------
    .. footbibliography::
    """

    max_angular_velocity: u.Quantity[u.physical.angular_velocity]
    """Maximum angular rate."""

    max_angular_acceleration: u.Quantity[u.physical.angular_acceleration]
    """Maximum angular acceleration."""

    settling_time: u.Quantity[u.physical.time] = 0 * u.second
    """Time to settle to rest after a slew."""

    @staticmethod
    def _time(
        x: u.Quantity[u.physical.angle],
        v: u.Quantity[u.physical.angular_velocity],
        a: u.Quantity[u.physical.angular_acceleration],
        s: u.Quantity[u.physical.time],
    ) -> u.Quantity[u.physical.time]:
        xc = np.square(v) / a
        return np.where(x <= xc, 2 * np.sqrt(x / a), (x + xc) / v) + s

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
        return self._time(
            self.separation(center1, center2, roll1, roll2),
            self.max_angular_velocity,
            self.max_angular_acceleration,
            self.settling_time,
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
