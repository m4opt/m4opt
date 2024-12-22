from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


class Constraint(ABC):
    """Base class for field of regard constraints."""

    @abstractmethod
    def __call__(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ) -> npt.NDArray[np.bool_]:
        """Evaluate the constraint at a given observer location, target position, and time."""
