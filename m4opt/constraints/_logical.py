from functools import reduce

import numpy as np

from ..utils.typing_extensions import override
from ._core import Constraint


class LogicalReductionConstraint(Constraint):
    def __init__(self, *operands: Constraint):
        self._operands = tuple(operands)

    @override
    def __call__(self, *args):
        return reduce(self._operator, (operand(*args) for operand in self._operands))


class LogicalAndConstraint(LogicalReductionConstraint):
    """Combine two or more constraints using a logical "and" operation.

    See Also
    --------
    LogicalOrConstraint, LogicalNotConstraint

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import (
    ...     AtNightConstraint, LogicalAndConstraint, SunSeparationConstraint)
    >>> constraint = (
    ...     AtNightConstraint.twilight_astronomical() &
    ...     SunSeparationConstraint(40 * u.deg))
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint(location, target, time)
    np.True_
    """

    _operator = np.logical_and


class LogicalOrConstraint(LogicalReductionConstraint):
    """Combine two or more constraints using a logical "or" operation.

    See Also
    --------
    LogicalAndConstraint, LogicalNotConstraint

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import (
    ...     AtNightConstraint, LogicalOrConstraint, SunSeparationConstraint)
    >>> constraint = (
    ...     AtNightConstraint.twilight_astronomical() |
    ...     SunSeparationConstraint(40 * u.deg))
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint(location, target, time)
    np.True_
    """

    _operator = np.logical_or


class LogicalNotConstraint(Constraint):
    """Perform a logical "not" on a constraint.

    See Also
    --------
    LogicalAndConstraint, LogicalOrConstraint

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation, SkyCoord
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.constraints import (AtNightConstraint, LogicalNotConstraint)
    >>> constraint = ~AtNightConstraint.twilight_astronomical()
    >>> time = Time("2017-08-17T00:41:04Z")
    >>> target = SkyCoord.from_name("NGC 4993")
    >>> location = EarthLocation.of_site("Rubin Observatory")
    >>> constraint(location, target, time)
    True
    """

    def __init__(self, operand: Constraint):
        self._operand = operand

    @override
    def __call__(self, *args):
        return np.logical_not(super().__call__(*args))
