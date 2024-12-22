from abc import ABC, abstractmethod

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time


class Orbit(ABC):
    """Base class for an Earth satellite with a specified orbit."""

    @property
    @abstractmethod
    def period(self) -> u.Quantity[u.physical.time]:
        """The orbital period."""

    @abstractmethod
    def __call__(self, time: Time) -> SkyCoord:
        """Get the position and velocity of the satellite.

        Parameters
        ----------
        time
            The time of the observation.

        Returns
        -------
        :
            The coordinates of the satellite in the ITRS frame.
        """
