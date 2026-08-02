from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.modeling import Model
from astropy.time import Time
from astropy.utils.state import ScienceState

error = """\
Unknown target. Please evaluate the model by providing the position and \
observing time in a `with:` statement, like this:
    from m4opt.synphot import observing
    with observing(observer_location=loc, target_coord=coord, obstime=time):
        ...  # evaluate model here\
"""


@dataclass
class State:
    observer_location: EarthLocation
    target_coord: SkyCoord
    obstime: Time


class state(ScienceState):
    """Context manager for global target coordinates and observing time."""

    _value: State | None = None

    @staticmethod
    def validate(value):
        if value is None:
            raise ValueError(error)
        # Check that observing state elements are broadcastable to a common shape
        np.broadcast_shapes(
            *(
                arg.shape
                for arg in (value.observer_location, value.target_coord, value.obstime)
            )
        )
        return value


def observing(*args, **kwargs):
    return state.set(State(*args, **kwargs))


class ScaleFactor(ABC, Model):
    """Scale factor for spectral models that depend only on extrinsic parameters."""

    n_inputs = 1
    n_outputs = 1

    @property
    @abstractmethod
    def value(self) -> float:
        """Return the value of the scale factor."""

    def __call__(self, x):
        return self.evaluate(x)

    def evaluate(self, x):
        value = self.value
        shape = np.broadcast_shapes(np.shape(value), np.shape(x))
        return np.broadcast_to(value, shape)


class ExtrinsicScaleFactor(ScaleFactor):
    """Scale factor for spectral models that depend only on extrinsic parameters."""

    @abstractmethod
    def at(
        self, observer_location: EarthLocation, target_coord: SkyCoord, obstime: Time
    ) -> float:
        """Evaluate the scale factor at a particular observerd location, target coordinate, and time."""

    @property
    def value(self) -> float:
        obs = state.get()
        return self.at(
            observer_location=obs.observer_location,
            target_coord=obs.target_coord,
            obstime=obs.obstime,
        )


class TabularScaleFactor(ScaleFactor):
    def __init__(self, array, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._value = array

    @property
    def value(self):
        return self._value
