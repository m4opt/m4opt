import AbstractConstraints

"""
As I've said in the other 2, these constraints are all going to look the same
in their structure and methods like this. The thing that sets them apart is
implementation.
"""

class SkyAreaRestrictionConstraint(AbstractConstraints.GroundConstraint):
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


class FilterConstraint(AbstractConstraints.GeneralConstraint):
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

