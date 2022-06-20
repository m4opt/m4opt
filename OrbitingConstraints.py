import AbstractConstraints

"""
These are all pretty damn similar. As implementation moves along, might group
some into subclases of to-be-defined classes if the math is similar enough
"""

class EclipsingConstraint(AbstractConstraints.OrbitingConstraint):
    """
    Instantiate this constraint
    """

    def __init__(self, observerData: dict, constraintParams: dict):
        """
        Constraint-specific info goes here
        We want ID, and then any specific info needed for this constraint
        """

    """
    Add this constraint to the model
    """

    def add_self_to_model(self, model: object) -> None:

    """
    Remove this constraint from the model
    """

    def remove_self_from_model(self, model: object) -> None:

    """
    Define our string method so you can easily represent this in a print
    statement
    """

    def __str__(self) -> str:

class IntegratedSolarFluxConstraint(AbstractConstraints.OrbitingConstraint):
    """
    Instantiate this constraint
    """

    def __init__(self, observerData: dict, constraintParams: dict):
        """
        Constraint-specific info goes here
        We want ID, and then any specific info needed for this constraint
        """

    """
    Add this constraint to the model
    """

    def add_self_to_model(self, model: object) -> None:

    """
    Remove this constraint from the model
    """

    def remove_self_from_model(self, model: object) -> None:

    """
    Define our string method so you can easily represent this in a print
    statement
    """

    def __str__(self) -> str:

class SolarChargedParticlesConstraint(AbstractConstraints.OrbitingConstraint):
    """
    Instantiate this constraint
    """

    def __init__(self, observerData: dict, constraintParams: dict):
        """
        Constraint-specific info goes here
        We want ID, and then any specific info needed for this constraint
        """

    """
    Add this constraint to the model
    """

    def add_self_to_model(self, model: object) -> None:

    """
    Remove this constraint from the model
    """

    def remove_self_from_model(self, model: object) -> None:

    """
    Define our string method so you can easily represent this in a print
    statement
    """

    def __str__(self) -> str:

class AngleFromSunConstraint(AbstractConstraints.OrbitingConstraint):
    """
    Instantiate this constraint
    """

    def __init__(self, observerData: dict, constraintParams: dict):
        """
        Constraint-specific info goes here
        We want ID, and then any specific info needed for this constraint
        """

    """
    Add this constraint to the model
    """

    def add_self_to_model(self, model: object) -> None:

    """
    Remove this constraint from the model
    """

    def remove_self_from_model(self, model: object) -> None:

    """
    Define our string method so you can easily represent this in a print
    statement
    """

    def __str__(self) -> str:

class AngleFromEarthConstraint(AbstractConstraints.OrbitingConstraint):
    """
    Instantiate this constraint
    """

    def __init__(self, observerData: dict, constraintParams: dict):
        """
        Constraint-specific info goes here
        We want ID, and then any specific info needed for this constraint
        """

    """
    Add this constraint to the model
    """

    def add_self_to_model(self, model: object) -> None:

    """
    Remove this constraint from the model
    """

    def remove_self_from_model(self, model: object) -> None:

    """
    Define our string method so you can easily represent this in a print
    statement
    """

    def __str__(self) -> str: