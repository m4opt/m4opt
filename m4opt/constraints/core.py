from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


class Constraint(ABC):
    @abstractmethod
    def __call__(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ) -> bool | npt.NDArray[np.bool_]:
        raise NotImplementedError
