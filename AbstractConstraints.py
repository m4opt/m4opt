from abc import ABC, abstractmethod

"""
This is where all of the abstract classes that define constraints will live.
"""


"""
The general Constraint class is abstract because there is no platonic ideal 
constraint, just more specific constraints that inherit this
"""


class Constraint(ABC):
    """
    Add this constraint to the model
    """

    @abstractmethod
    def add_self_to_model(self, model: object) -> None:
        pass

    """
    Remove this constraint from the model
    """

    @abstractmethod
    def remove_self_from_model(self, model: object) -> None:
        pass


"""
This is the parent class for constraints applicable to both space and ground 
based telescopes, so things like total observing time, overhead time and etc.
"""


class GeneralConstraint(ABC, Constraint):
    observerType = 'common'


"""
This is the parent class for all constraints specific to a ground-based 
observer
"""

class GroundConstraint(ABC, Constraint):
    observerType = 'ground'

"""
This is the parent class for al constraints specific to an orbiting observer
"""

class OrbitingConstraint(ABC, Constraint):
    observerType = 'orbiting'

