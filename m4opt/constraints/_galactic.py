import numpy as np
from astropy import units as u

from ._core import Constraint


class GalacticLatitudeConstraint(Constraint):
    """
    Constrain the distance from the Galactic plane.

    Parameters
    ----------
    min : :class:`astropy.units.Quantity`
        Minimum of absolute value of Galactic latitude.
    """

    def __init__(self, min: u.Quantity[u.physical.angle]):
        self.min = min

    def __call__(self, observer_location, target_coord, obstime):
        return np.abs(target_coord.galactic.b) >= self.min
