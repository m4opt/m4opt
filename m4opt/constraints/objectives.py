
import AbstractConstraints

"""
Objective constraints.

Current list:
Integral of a weight function over the combined observation footprint
Wavelength-dependent and time-dependent weights (light curve)
Sum of arbitrary weights for targets (TAC score, priority)
Integral over a utility function and a depth map - how is this different
from the first one?
"""

class SkymapWeightedObservationConstraint(AbstractConstraints.GeneralConstraint):

    """
    Instantiate this constraint
    """
    def __init__(self, moc_data: numpy.ndarray, weight_function: function,
                 tiles: List[dict]):
        """
        Needs the skymap in MOC form, some kind of weight function (that
        takes a MOC array as input), and a list of the observed tiles
        """

class LightcurveConstraint(AbstractConstraints.GeneralConstraint):

    """
    Instantiate this constraint
    """
    def __init__(self, light_curve: dict, moc_data: numpy.ndarray,
                 tiles: List[dict]):
        """
        Needs the light curve (dictionary with filter as key and
        2 x N array with time and absolute magnitude), skymap in MOC form,
        and a list of the observed tiles
        """

class IndividualTargetConstraint(AbstractConstraints.GeneralConstraint):

    """
    Instantiate this constraint
    """
    def __init__(self, targets_weight: numpy.ndarray,
                 targets: List[dict]):
        """
        Needs the targets (dictionary with name as key and information such
        as Right Ascension and Declination),
        and a 1D-array same length as the list of targets
        """





