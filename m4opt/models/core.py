from dataclasses import dataclass
from typing import Optional

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy.utils.state import ScienceState

error = """\
Unknown target. Please evaluate the model by providing the position and \
observing time in a `with:` statement, like this:
    from m4opt.models import state
    with state.set_observing(target_coord=coord, obstime=time):
        ...  # evaluate model here\
"""


@dataclass
class ObservingState:
    obstime: Optional[Time] = None
    target_coord: Optional[SkyCoord] = None
    observatory_loc: Optional[EarthLocation] = None


class state(ScienceState):
    """Context manager for global target coordinates and observing time."""

    _value = None

    @classmethod
    def validate(cls, value):
        if value is None:
            raise ValueError(error)
        return value

    @classmethod
    def set_observing(cls, *args, **kwargs):
        return cls.set(ObservingState(*args, **kwargs))
