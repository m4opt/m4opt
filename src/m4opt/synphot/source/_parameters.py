from astropy import units as u
from astropy.modeling import Parameter
from astropy.modeling.parameters import ParameterDefinitionError

from ._priors import _UnitPrior


class PriorParameter(Parameter):
    """A `~astropy.modeling.Parameter` that requires a resamplable prior.

    ``prior`` must be a `~m4opt.synphot.models._priors._UnitPrior` (e.g.
    `NormalPrior`, `UniformPrior`, `LogNormalPrior`) with a unit equivalent
    to this parameter's ``unit``. Otherwise it's an ordinary
    `~astropy.modeling.Parameter`.
    """

    def __init__(self, *args, prior=None, default=None, unit=None, **kwargs):
        name = kwargs.get("name") or "PriorParameter"
        if not isinstance(prior, _UnitPrior):
            raise ParameterDefinitionError(
                f"{name} must declare a prior as a _UnitPrior (e.g. "
                "NormalPrior, UniformPrior, LogNormalPrior)"
            )
        declared_unit = unit
        if declared_unit is None and isinstance(default, u.Quantity):
            declared_unit = default.unit
        if declared_unit is not None and not prior.unit.is_equivalent(
            u.Unit(declared_unit)
        ):
            raise ParameterDefinitionError(
                f"{name} prior unit {prior.unit!r} is not equivalent to "
                f"parameter unit {declared_unit!r}"
            )
        super().__init__(*args, prior=prior, default=default, unit=unit, **kwargs)
