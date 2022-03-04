from astropy.utils.state import ScienceState

error = """\
Unknown {name}. Please evaluate the model by providing the target position \
and observing time in a `with:` statement, like this:
    from m4opt.models import position, time
    with position.set(skycoord), time.set(obstime):
        ...  # evaluate model here \
"""


class ObservingState(ScienceState):

    _value = None

    @classmethod
    def validate(cls, value):
        if value is None:
            raise ValueError(error.format(name=cls.__name__))
        return value


class position(ObservingState):
    """Context manager for setting global target sky position."""


class time(ObservingState):
    """Context manager for setting global observation time."""
