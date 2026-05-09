from abc import ABC, abstractmethod

from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time
from synphot import SourceSpectrum

BACKGROUND_SOLID_ANGLE = 1 * u.arcsec**2
"""Standard solid angle used for sky brightness models."""


class ContextualBackground(ABC):
    """Base class for backgrounds dependent on observation context."""

    @abstractmethod
    def __call__(
        self,
        observer_location: EarthLocation,
        obstime: Time,
    ) -> SourceSpectrum:
        """Compute background spectrum at specified observation context."""
        pass


def update_missions(
    mission,
    observer_location: EarthLocation,
    obstime: Time,
) -> None:
    """Update all contextual backgrounds in all missions."""

    detector = mission.detector
    if detector is None:
        return

    try:
        for field_name in detector.__dataclass_fields__:
            field_value = getattr(detector, field_name)

            if isinstance(field_value, ContextualBackground):
                spectrum = field_value(
                    observer_location=observer_location,
                    obstime=obstime,
                )
                setattr(detector, field_name, spectrum)

    except (AttributeError, TypeError):
        pass
