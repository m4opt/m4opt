from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from typing import ClassVar

import numpy as np
from astropy.units import Quantity, UnitConversionError

from m4opt.models._typing import (
    FloatArray,
    PhysicalInput,
    RNGInput,
    ScalarPhysicalValue,
)
from m4opt.models._utils import ensure_numpy_array
from m4opt.models.core.priors import Prior

__all__ = ["Parameter"]


@dataclass(slots=True)
class Parameter:
    r"""
    A single physical parameter of a model.

    A :class:`Parameter` describes how one physical quantity is drawn: a
    :class:`~m4opt.models.core.priors.Prior` to sample from, a
    characteristic physical :attr:`scale`, and an optional :attr:`transform` that
    lets the prior be defined on a more convenient variable than the physical value
    itself.

    Internally, sampling proceeds through a *latent* variable :math:`z`, related to
    the physical value :math:`x` by

    .. math::

        z = T(x / x_0), \qquad x = T^{-1}(z) \cdot x_0,

    where :math:`x_0` is :attr:`scale` and :math:`T` is :attr:`transform` (the
    identity by default). This is what lets, e.g., a strictly-positive,
    many-orders-of-magnitude parameter be sampled from a well-behaved
    :class:`~m4opt.models.core.priors.NormalPrior` via ``transform="log"``,
    without every prior implementation needing to separately handle scale and shape.

    Examples
    --------
    .. code-block:: python

        rise_time = Parameter(
            prior=NormalPrior(mean=0.0, sigma=1.0),
            scale=1.0 * u.day,
            transform="log",
            description="Rise timescale",
            latex=r"\tau_{\rm rise}",
        )
        rise_time.sample(size=1000, rng=0)

        # Pin it to a constant for a diagnostic run, then release it again.
        rise_time.fix(2.5 * u.day)
        rise_time.is_fixed  # True
        rise_time.unfix()
    """

    # -------------------------------------- #
    # Class-Level Parameters                 #
    # -------------------------------------- #
    _TRANSFORMS: ClassVar[
        dict[str, tuple[Callable[..., FloatArray], Callable[..., FloatArray]]]
    ] = {
        "log": (np.log, np.exp),
        "log10": (np.log10, lambda z: np.power(10.0, z)),
    }
    """dict: Built-in string ``transform`` names, mapped to ``(transform, inverse)`` pairs.

    See :meth:`available_transforms` for the list of names.
    """

    # -------------------------------------- #
    # Dataclass Fields                       #
    # -------------------------------------- #
    prior: Prior
    """~m4opt.models.core.priors.Prior: The prior distribution for the latent variable :math:`z`."""

    scale: ScalarPhysicalValue
    """~astropy.units.Quantity, float, or int: The characteristic scale :math:`x_0` for this parameter.

    Every physical value is divided by this scale before the prior is consulted, so
    the prior itself can be defined in convenient, unit-free terms. Use a
    :class:`~astropy.units.Quantity` whenever the parameter has physical units; a
    bare ``float``/``int`` is only appropriate for a genuinely dimensionless parameter.
    """

    transform: str | Callable[[FloatArray], FloatArray] | None = None
    """str or callable, optional: The transform :math:`T` mapping the scaled variable to the latent variable.

    If ``None`` (the default), :math:`T` is the identity. If a string, it must name a
    built-in transform (see :meth:`available_transforms`). If a callable,
    :attr:`inverse_transform` must also be supplied as a callable.
    """

    inverse_transform: Callable[[FloatArray], FloatArray] | None = None
    """callable, optional: The inverse :math:`T^{-1}` of :attr:`transform`.

    Only needs to be supplied when :attr:`transform` is a custom callable; filled in
    automatically for the built-in string transforms.
    """

    description: str | None = None
    """str, optional: A short, human-readable description of the parameter."""

    latex: str | None = None
    """str, optional: A LaTeX symbol for the parameter, for use in plot labels."""

    fixed_value: ScalarPhysicalValue | None = None
    """Quantity, float, or int, optional: A constant physical value this parameter is pinned to.

    When set, :meth:`sample` always returns this value instead of drawing from
    :attr:`prior`. Prefer :meth:`fix`/:meth:`unfix` over setting this field directly
    -- they validate that the value is actually usable, which direct assignment does
    not.
    """

    # ----------------------------------- #
    # Initialization / Dunders            #
    # ----------------------------------- #
    def __post_init__(self) -> None:
        """Validate fields and resolve :attr:`transform`/:attr:`inverse_transform` after construction."""
        if not isinstance(self.prior, Prior):
            raise TypeError(
                "Parameter 'prior' must be an instance of m4opt.models.core.priors.Prior, "
                f"but got {type(self.prior).__name__}.",
            )

        if isinstance(self.scale, bool) or not isinstance(
            self.scale, (Quantity, float, int)
        ):
            raise TypeError(
                f"Parameter 'scale' must be an astropy Quantity, float, or int, but got {type(self.scale).__name__}.",
            )

        self._resolve_transform()

        if self.fixed_value is not None:
            self._check_fixed_value(self.fixed_value)

    def _resolve_transform(self) -> None:
        """Fill in :attr:`transform`/:attr:`inverse_transform` from whatever was provided."""
        if self.transform is None:
            self.transform = lambda z: z
            self.inverse_transform = lambda z: z
        elif isinstance(self.transform, str):
            try:
                self.transform, self.inverse_transform = self._TRANSFORMS[
                    self.transform
                ]
            except KeyError:
                raise ValueError(
                    f"Unsupported parameter transform {self.transform!r}. "
                    f"Supported transforms are {self.available_transforms()}."
                ) from None
        elif callable(self.transform):
            if not callable(self.inverse_transform):
                raise ValueError(
                    "Parameter 'transform' was provided as a callable, so 'inverse_transform' "
                    "must also be supplied as a callable.",
                )
        else:
            raise TypeError(
                f"Parameter 'transform' must be a string or callable, not {type(self.transform).__name__}.",
            )

    def __call__(
        self,
        size: int = 1,
        *,
        rng: RNGInput = None,
    ) -> PhysicalInput:
        """Draw random samples from the prior. A convenience alias for :meth:`sample`."""
        return self.sample(size=size, rng=rng)

    def __copy__(self) -> "Parameter":
        # By the time `self` exists, `__post_init__` has already resolved
        # `transform`/`inverse_transform` to callables (see `_resolve_transform`),
        # so re-running `__init__` via `replace` takes the `callable` branch there
        # and is a no-op on them -- safe to call again.
        return replace(self)

    def __deepcopy__(self, memo: dict) -> "Parameter":
        if id(self) in memo:
            return memo[id(self)]

        result = replace(
            self,
            **{
                f.name: deepcopy(getattr(self, f.name), memo)
                for f in fields(self)
                if f.init
            },
        )
        memo[id(self)] = result
        return result

    # ----------------------------------- #
    # Class Utilities                     #
    # ----------------------------------- #
    @classmethod
    def available_transforms(cls) -> tuple[str, ...]:
        """Tuple of str: Names of the built-in string transforms accepted by :attr:`transform`."""
        return tuple(cls._TRANSFORMS)

    # ----------------------------------- #
    # Parameter Conversion                #
    # ----------------------------------- #
    def transform_physical_to_latent(self, x: PhysicalInput) -> FloatArray:
        """
        Convert a physical value :math:`x` to the latent variable :math:`z = T(x / x_0)`.

        Parameters
        ----------
        x : Quantity or array_like
            Physical-unit value(s). Must be compatible with :attr:`scale`'s units.

        Returns
        -------
        numpy.ndarray
            The corresponding latent value(s).
        """
        # `x` may arrive as a plain list/tuple (valid `array_like`), which does
        # not itself support `/`; normalize to an ndarray first unless it is a
        # Quantity, whose division already handles the (possibly-Quantity)
        # `scale` directly.
        x_arr: Quantity | FloatArray = (
            x if isinstance(x, Quantity) else np.asarray(x, dtype=np.float64)
        )
        y = ensure_numpy_array(x_arr / self.scale)

        assert callable(self.transform)
        return self.transform(y)

    def transform_latent_to_physical(self, z: FloatArray) -> Quantity | FloatArray:
        r"""
        Convert a latent value :math:`z` back to the physical value :math:`x = T^{-1}(z) \cdot x_0`.

        Parameters
        ----------
        z : array_like
            Latent value(s).

        Returns
        -------
        Quantity or numpy.ndarray
            The corresponding physical value(s), carrying :attr:`scale`'s units if
            :attr:`scale` is itself a :class:`~astropy.units.Quantity`.
        """
        assert callable(self.inverse_transform)
        y = self.inverse_transform(np.asarray(z, dtype=np.float64))

        return y * self.scale

    # ----------------------------------- #
    # Sampling                            #
    # ----------------------------------- #
    def sample_latent_variable(
        self,
        size: int = 1,
        *,
        rng: RNGInput = None,
    ) -> FloatArray:
        """
        Draw ``size`` samples of the latent variable :math:`z`.

        If :attr:`is_fixed`, every draw is the (transformed) fixed value; otherwise
        this delegates to :attr:`prior`.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw. The default is ``1``.
        rng : numpy.random.Generator, int, or None, optional
            Random-number source. Unused when :attr:`is_fixed`.

        Returns
        -------
        numpy.ndarray
            Latent-space samples with shape ``(size,)``.
        """
        if self.is_fixed:
            latent_value = self.transform_physical_to_latent(self.fixed_value)

            return np.full(size, latent_value, dtype=np.float64)

        return self.prior.sample(size=size, rng=rng)

    def sample(
        self,
        size: int = 1,
        *,
        rng: RNGInput = None,
    ) -> PhysicalInput:
        """
        Draw ``size`` physical-unit samples of this parameter.

        Parameters
        ----------
        size : int, optional
            Number of samples to draw. The default is ``1``.
        rng : numpy.random.Generator, int, or None, optional
            Random-number source. Unused when :attr:`is_fixed`.

        Returns
        -------
        Quantity or numpy.ndarray
            Physical-unit samples with shape ``(size,)``.
        """
        latent_samples = self.sample_latent_variable(size=size, rng=rng)

        return self.transform_latent_to_physical(latent_samples)

    # ----------------------------------- #
    # Priors                              #
    # ----------------------------------- #
    def set_prior(self, prior: Prior) -> None:
        """
        Set the prior distribution for this parameter.

        Parameters
        ----------
        prior : m4opt.models.core.priors.Prior
            The new prior distribution to use for this parameter.
        """
        if not isinstance(prior, Prior):
            raise TypeError(
                "Parameter 'prior' must be an instance of m4opt.models.core.priors.Prior, "
                f"but got {type(prior).__name__}.",
            )
        self.prior = prior

    # ----------------------------------- #
    # Fixing                              #
    # ----------------------------------- #
    @property
    def is_fixed(self) -> bool:
        """bool: Whether this parameter is pinned to a constant value (see :meth:`fix`)."""
        return self.fixed_value is not None

    def _check_fixed_value(self, value: ScalarPhysicalValue) -> None:
        """Validate a candidate fixed value, shared by :meth:`__post_init__` and :meth:`fix`."""
        if isinstance(value, bool) or not isinstance(value, (Quantity, float, int)):
            raise TypeError(
                f"Fixed parameter value must be an astropy Quantity, float, or int, got {type(value).__name__}."
            )

        # A non-finite result (e.g. log of a non-positive value) is expected input here,
        # not a numerical error, so suppress the warning numpy would otherwise raise.
        with np.errstate(invalid="ignore", divide="ignore"):
            try:
                latent_value = float(self.transform_physical_to_latent(value))
            except UnitConversionError as exc:
                raise TypeError(
                    f"Fixed value {value!r} has units incompatible with this "
                    f"parameter's scale ({self.scale!r})."
                ) from exc

        if not np.isfinite(latent_value):
            raise ValueError(
                f"Fixed value {value!r} maps to a non-finite latent value ({latent_value})."
            )

    def fix(self, value: ScalarPhysicalValue) -> None:
        """
        Pin this parameter to a constant physical value, bypassing :attr:`prior`.

        ``value`` need not lie within :attr:`prior`'s support -- fixing is
        exactly the escape hatch for exploring physical values the prior
        wouldn't itself generate (e.g. a diagnostic run at a fiducial value
        from the literature). It only needs to be well-defined: compatible
        with :attr:`scale`'s units and mapping, through :attr:`transform`, to
        a finite latent value.

        Parameters
        ----------
        value : Quantity, float, or int
            The physical value to fix this parameter to.

        Raises
        ------
        TypeError
            If ``value`` is not a Quantity, float, or int, or if its units are
            incompatible with :attr:`scale`'s.
        ValueError
            If ``value`` maps to a non-finite latent value (e.g. a non-positive
            physical value under a ``transform="log"``).

        See Also
        --------
        unfix : Release a fixed parameter.
        """
        self._check_fixed_value(value)
        self.fixed_value = value

    def unfix(self) -> None:
        """
        Release a fixed parameter, restoring sampling from :attr:`prior`.

        A no-op if the parameter is not currently fixed.
        """
        self.fixed_value = None
