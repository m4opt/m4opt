"""Unit-aware priors for `~m4opt.synphot.models.PriorParameter`."""

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import numpy as np
import scipy.stats


@dataclass
class _UnitPrior:
    """Pairs a `scipy.stats`-style distribution with a unit.

    Parameters
    ----------
    distribution
        Anything with an ``.rvs(size, random_state)`` method (e.g. a
        frozen `scipy.stats` distribution), evaluated in ``unit``.
    unit
        The unit ``distribution`` is expressed in.
    """

    distribution: Any
    unit: u.UnitBase

    def rvs(self, size=None, random_state=None):
        """Draw ``size`` samples, as a `~astropy.units.Quantity` in ``unit``."""
        return self.distribution.rvs(size=size, random_state=random_state) * self.unit


class UniformPrior(_UnitPrior):
    """A uniform prior between ``lower`` and ``upper``.

    Parameters
    ----------
    lower, upper
        The bounds of the distribution, as plain numbers expressed in
        ``unit``.
    unit
        The unit ``lower``/``upper`` are expressed in.
    """

    def __init__(self, lower, upper, unit):
        super().__init__(scipy.stats.uniform(loc=lower, scale=upper - lower), unit)


class NormalPrior(_UnitPrior):
    """A normal (Gaussian) prior with the given mean and standard deviation.

    Parameters
    ----------
    loc, scale
        The mean and standard deviation, as plain numbers expressed in
        ``unit``.
    unit
        The unit ``loc``/``scale`` are expressed in.
    """

    def __init__(self, loc, scale, unit):
        super().__init__(scipy.stats.norm(loc=loc, scale=scale), unit)


class LogNormalPrior(_UnitPrior):
    """A log-normal prior.

    ``log(x / unit)`` is normally distributed with mean ``mu`` and
    standard deviation ``sigma``.

    Parameters
    ----------
    mu, sigma
        The mean and standard deviation of ``log(x / unit)``.
    unit
        The unit ``x`` is expressed in.
    """

    def __init__(self, mu, sigma, unit):
        super().__init__(scipy.stats.lognorm(s=sigma, scale=np.exp(mu)), unit)
