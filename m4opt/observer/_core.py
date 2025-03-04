from abc import ABC, abstractmethod

from astropy.coordinates import EarthLocation
from astropy.time import Time


class ObserverLocation(ABC):
    """Base class for an Earth satellite with a specified orbit."""

    @abstractmethod
    def __call__(self, time: Time) -> EarthLocation:
        """Get the position and velocity of the satellite.

        Parameters
        ----------
        time
            The time of the observation.

        Returns
        -------
        :
            The Earth-relative coordinates of the satellite.
        """
