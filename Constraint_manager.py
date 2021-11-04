import AbstractConstraints


"""
This is a class created by the observer and is the primary way the user
interacts with constraints. 

It should receive data from the observer about what type of observer it is,
where it is located on earth/in orbit, what filter/sensors the observer has and
etc. 

This class exists for ease of use for the user

"""
class Constraint_manager:


    def __init__(self, observerData: dict, model):
        """
        Initalize all of your local variables about the observer (or maybe
        stick with the dict?), an empty list of constraints added, and the
        model
        """


    """
    Add the given constraint to the model and create an instance of the 
    constraint class which is added to the list of current constraints
    
    Object returned is self to allow for method chaining
    
    Should throw value error if given constraint not in the list or if you try
    to add a ground-based constraint to orbiting observer or vice versa
    
    Should also check that there doesn't exist a constraint with the same ID
    already
    """
    def add_constraint(self,
                       constraintType: object,
                       constraintParams: dict) -> object:
        """
        something like:
        if (constraintType.observerType != 'common' and
            constraintType.observerType != self.observerData['type']:
            throw ValueError

        constraint_to_be_added = constraintType(observerData, constraintParams)
        constraint_to_be_added.add_self_to_model(model)
        self.constraint_list.append(constraint_to_be_added)

        return self
        """


    """
    Remove a constraint with the given identifier
    
    Object returned is self to allow for method chaining
    """
    def remove_constraint(self,
                          constraintToRemoveID: str) -> object:
        """
        something like:

        constraint_to_remove = __find_constraint_by_id(constraintToRemoveID)
        constraint_to_remove.remove_self_from_model(model)
        self.constraint_list.pop(constraint_to_remove)
            (pop might not work if it's instance vs. pointer, but do something
            similar)

        return self
        """

    """
    Private method to find a constraint by a unique ID string (a name or
    something). Used to find a constraint to remove it and potentially for more
    down the road
    
    returns the constraint with that ID
    """
    def __find_constraint_by_id(self, constraintID: str) -> object:

        """
        loop through self.constraint_list and see which one has constraintID
        """