from dataclasses import dataclass
from io import IOBase
from typing import Optional

from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time


@dataclass
class Observer:

    name: [str, None]

    def __call__(self, time: Time) -> SkyCoord:
        raise NotImplementedError


class EarthFixedObserver(Observer):

    earth_location: EarthLocation

    @classmethod
    def defContraints(self): # form constraints based on observer type 
        pass 
        # Contrainst interaction with User to get (interact with B.)
        # pull from constraint class separated into orbital and earth fixed oberserver constraints.
        # Number of instruments & How many can be used in sync

    def __init__(self, earth_location: EarthLocation, name: Optional[str] = None):
        self.earth_location = earth_location
        self.name = name

    @classmethod
    def at_site(cls, site: str) -> EarthFixedObserver:
        return cls(EarthLocation.of_site(site))

    def __call__(self, time: Time):
        return self.earth_location.itrs


class EarthOrbitObserver(Observer):

    @classmethod
    def defContraints(self): # form constraints based on observer type 
        pass 
        # Contrainst interaction with User to get (interact with B.)
        # pull from constraint class separated into orbital and earth fixed oberserver constraints.
        # Number of instruments & How many can be used in sync

    @classmethod
    def from_tle(cls, tle: [str, IOBase]):
        raise NotImplementedError

    @classmethod
    def from_spice_kernel(cls, spice):
        raise NotImplementedError

class Trajectory:

    def __call__(self, time: Time) -> SkyCoord:
        raise NotImplementedError


class TLETrajectory(Trajectory):
    pass


class SpiceTrajectory(Trajectory):
    pass