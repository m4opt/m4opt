from typing import ClassVar

import numpy as np
from astropy import units as u
from numpy.typing import NDArray

from m4opt.models.core import Lightcurve, LogNormalPrior, NormalPrior, Parameter

from .._typing import CGSParameterValue
from .._utils import _BOL_LUM_UNIT

__all__ = [
    "BazinLightcurve",
    "BrokenPowerLawLightcurve",
    "DelayedExponentialLightcurve",
    "FREDLightcurve",
    "GREDLightcurve",
    "GaussianPulseLightcurve",
    "LogNormalPulseLightcurve",
    "PlateauPowerLawLightcurve",
    "PowerLawLightcurve",
    "SmoothBrokenPowerLawLightcurve",
    "TopHatLightcurve",
    "VillarLightcurve",
]


class TopHatLightcurve(Lightcurve):
    r"""
    A constant luminosity for a fixed duration, zero before and after.

    .. math::

        L(t) = \begin{cases} A & 0 \le t \le \Delta t \\ 0 & t > \Delta t \end{cases}

    The simplest possible pulse shape: a plateau at the peak amplitude :math:`A` for a
    duration :math:`\Delta t`, with an instantaneous rise and cutoff.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Plateau bolometric luminosity.
       * - ``duration``
         - :math:`\Delta t`
         - Duration of the plateau.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Plateau bolometric luminosity.",
            latex=r"A",
        ),
        "duration": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Duration of the plateau.",
            latex=r"\Delta t",
        ),
    }

    @classmethod
    def _eval(
        cls, t: NDArray[np.float64], **parameters: CGSParameterValue
    ) -> NDArray[np.float64]:
        amplitude, duration = parameters["amplitude"], parameters["duration"]

        with np.errstate(divide="ignore"):
            return np.where(t <= duration, np.log(amplitude), -np.inf)


class GaussianPulseLightcurve(Lightcurve):
    r"""
    A Gaussian pulse in luminosity, peaked at ``t_peak``.

    .. math::

        L(t) = A \exp\left(-\frac{(t - t_\mathrm{peak})^2}{2\sigma^2}\right)

    :math:`L(t_\mathrm{peak}) = A` exactly, by construction.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity since explosion.
       * - ``sigma``
         - :math:`\sigma`
         - Gaussian width of the pulse.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time of peak luminosity since explosion.",
            latex=r"t_\mathrm{peak}",
        ),
        "sigma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Gaussian width of the pulse.",
            latex=r"\sigma",
        ),
    }

    @classmethod
    def _eval(
        cls, t: NDArray[np.float64], **parameters: CGSParameterValue
    ) -> NDArray[np.float64]:
        amplitude, t_peak, sigma = (
            parameters["amplitude"],
            parameters["t_peak"],
            parameters["sigma"],
        )

        return np.log(amplitude) - 0.5 * ((t - t_peak) / sigma) ** 2


class FREDLightcurve(Lightcurve):
    r"""
    A fast-rise, exponential-decay pulse (the Norris et al. 1996 GRB pulse shape).

    .. math::

        L(t) = A \exp\left(2\sqrt{\tau_1/\tau_2} - \frac{\tau_1}{t} - \frac{t}{\tau_2}\right), \quad t > 0

    with :math:`L(0) = 0`. Despite the apparent singularity at :math:`\tau_1/t`, this is
    well-behaved: as :math:`t \to 0^+`, the :math:`-\tau_1/t` term dominates and drives
    :math:`L \to 0`. The pulse peaks exactly at :math:`t_\mathrm{peak} = \sqrt{\tau_1 \tau_2}`,
    where :math:`L(t_\mathrm{peak}) = A`; :math:`\tau_1 \ll \tau_2` gives the characteristic
    fast rise / slow, exponential-looking decay.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``rise``
         - :math:`\tau_1`
         - Rise timescale tau_1.
       * - ``decay``
         - :math:`\tau_2`
         - Decay timescale tau_2.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.5 * u.day,
            description="Rise timescale tau_1.",
            latex=r"\tau_1",
        ),
        "decay": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3.0 * u.day,
            description="Decay timescale tau_2.",
            latex=r"\tau_2",
        ),
    }

    @classmethod
    def _eval(
        cls, t: NDArray[np.float64], **parameters: CGSParameterValue
    ) -> NDArray[np.float64]:
        amplitude, rise, decay = (
            parameters["amplitude"],
            parameters["rise"],
            parameters["decay"],
        )

        # `t` includes `t = 0` by convention (see `Lightcurve`'s class docstring), which
        # divides by zero here. That's the correct limit, not an error -- `rise / t -> inf`
        # drives the exponent to `-inf` and `L -> 0` -- so the warning is suppressed rather
        # than worked around.
        with np.errstate(divide="ignore"):
            return np.log(amplitude) + 2 * np.sqrt(rise / decay) - rise / t - t / decay


class GREDLightcurve(Lightcurve):
    r"""
    A Gaussian rise followed by an exponential decline, peaked at ``t_peak``.

    .. math::

        L(t) = A \times \begin{cases}
            \exp\left(-\dfrac{(t - t_\mathrm{peak})^2}{2\sigma_\mathrm{rise}^2}\right)
                & t \le t_\mathrm{peak} \\[4pt]
            \exp\left(-\dfrac{t - t_\mathrm{peak}}{\tau_\mathrm{decline}}\right)
                & t > t_\mathrm{peak}
        \end{cases}

    The two pieces agree at :math:`t = t_\mathrm{peak}`, where :math:`L(t_\mathrm{peak}) = A`
    exactly, by construction. ``t_peak`` is not itself a free parameter --
    it is fixed at :math:`t_\mathrm{peak} = 5\sigma_\mathrm{rise}`, five
    Gaussian widths after explosion.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``sigma_rise``
         - :math:`\sigma_\mathrm{rise}`
         - Gaussian width of the rise.
       * - ``tau_decline``
         - :math:`\tau_\mathrm{decline}`
         - Exponential decline timescale.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "sigma_rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=0.5 * u.day,
            description="Gaussian width of the rise.",
            latex=r"\sigma_\mathrm{rise}",
        ),
        "tau_decline": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=3.0 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{decline}",
        ),
    }

    @classmethod
    def _eval(
        cls, t: NDArray[np.float64], **parameters: CGSParameterValue
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        sigma_rise = parameters["sigma_rise"]
        t_peak = 5 * sigma_rise
        tau_decline = parameters["tau_decline"]

        is_rising = t <= t_peak
        log_rise = -0.5 * ((t - t_peak) / sigma_rise) ** 2
        log_decline = -(t - t_peak) / tau_decline

        return np.log(amplitude) + np.where(is_rising, log_rise, log_decline)


class BazinLightcurve(Lightcurve):
    r"""
    A smooth asymmetric transient following the Bazin functional form.

    .. math::

        L(t) =
        A\,
        \frac{
            \exp[-(t-t_0)/\tau_\mathrm{fall}]
        }{
            1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]
        }.

    The model combines a logistic-like rise with an exponential decline and is
    commonly useful as a generic phenomenological approximation to supernova-like
    transients.

    A finite maximum exists when

    .. math::

        \tau_\mathrm{fall} > \tau_\mathrm{rise}.

    Unlike most of the other pulse models in this module, ``amplitude`` is the
    normalization of the Bazin function and is not, in general, exactly the peak
    luminosity.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Characteristic transition time.
       * - ``rise``
         - :math:`\tau_\mathrm{rise}`
         - Logistic rise timescale.
       * - ``fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Luminosity normalization.",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Characteristic transition time.",
            latex=r"t_0",
        ),
        "rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Logistic rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "fall": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.3),
            scale=30.0 * u.day,
            description="Exponential decline timescale.",
            latex=r"\tau_\mathrm{fall}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t0 = parameters["t0"]
        rise = parameters["rise"]
        fall = parameters["fall"]

        x = t - t0

        return np.log(amplitude) - x / fall - np.logaddexp(0.0, -x / rise)


class PowerLawLightcurve(Lightcurve):
    r"""
    A power-law decline beginning at a reference time.

    .. math::

        L(t) =
        \begin{cases}
            0, & t < t_\mathrm{ref}, \\[4pt]
            A \left(\dfrac{t}{t_\mathrm{ref}}\right)^{-\alpha},
            & t \ge t_\mathrm{ref}.
        \end{cases}

    The luminosity is therefore exactly :math:`A` at ``t_ref``. This form is
    useful for generic afterglows and other transients whose post-onset emission
    is approximately scale free.

    Restricting the model to ``t >= t_ref`` avoids the formal divergence of a
    declining power law as :math:`t \rightarrow 0`.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Luminosity at the reference time.
       * - ``t_ref``
         - :math:`t_\mathrm{ref}`
         - Reference time at which the power law begins.
       * - ``index``
         - :math:`\alpha`
         - Positive power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Luminosity at the reference time.",
            latex=r"A",
        ),
        "t_ref": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Reference time at which the power law begins.",
            latex=r"t_\mathrm{ref}",
        ),
        "index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.dimensionless_unscaled,
            description="Positive power-law decline index.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_ref = parameters["t_ref"]
        index = parameters["index"]

        with np.errstate(divide="ignore", invalid="ignore"):
            log_luminosity = np.log(amplitude) - index * np.log(t / t_ref)

        return np.where(t >= t_ref, log_luminosity, -np.inf)


class BrokenPowerLawLightcurve(Lightcurve):
    r"""
    A sharply broken power-law transient with a rise and decline.

    .. math::

        L(t) =
        A
        \begin{cases}
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{\alpha_\mathrm{rise}},
                & 0 < t \le t_\mathrm{peak}, \\[6pt]
            \left(\dfrac{t}{t_\mathrm{peak}}\right)^{-\alpha_\mathrm{decline}},
                & t > t_\mathrm{peak}.
        \end{cases}

    Both indices are defined to be positive. The two branches meet exactly at
    ``t_peak``, where :math:`L(t_\mathrm{peak}) = A`. For positive rise index,
    the luminosity tends continuously to zero as :math:`t \rightarrow 0`.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of the power-law break and peak.
       * - ``rise_index``
         - :math:`\alpha_\mathrm{rise}`
         - Positive pre-peak power-law index.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive post-peak power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Time of the power-law break and peak.",
            latex=r"t_\mathrm{peak}",
        ),
        "rise_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive pre-peak power-law index.",
            latex=r"\alpha_\mathrm{rise}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive post-peak power-law decline index.",
            latex=r"\alpha_\mathrm{decline}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        rise_index = parameters["rise_index"]
        decline_index = parameters["decline_index"]

        with np.errstate(divide="ignore"):
            log_time = np.log(t / t_peak)

        log_shape = np.where(
            t <= t_peak,
            rise_index * log_time,
            -decline_index * log_time,
        )

        return np.log(amplitude) + log_shape


class SmoothBrokenPowerLawLightcurve(Lightcurve):
    r"""
    A smoothly broken power-law transient.

    The asymptotic behavior is

    .. math::

        L(t) \propto
        \begin{cases}
            t^{\alpha_\mathrm{rise}}, & t \ll t_\mathrm{peak}, \\
            t^{-\alpha_\mathrm{decline}}, & t \gg t_\mathrm{peak},
        \end{cases}

    with a smooth transition between the two branches. The implementation uses

    .. math::

        f(x) =
        \left[
            x^{-s\alpha_\mathrm{rise}}
            +
            x^{s\alpha_\mathrm{decline}}
        \right]^{-1/s},

    but rescales the argument and normalization so that the maximum occurs
    exactly at ``t_peak`` and

    .. math::

        L(t_\mathrm{peak}) = A.

    The positive ``smoothness`` parameter :math:`s` controls the sharpness of
    the transition: larger values approach a sharply broken power law.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``rise_index``
         - :math:`\alpha_\mathrm{rise}`
         - Positive asymptotic rise index.
       * - ``decline_index``
         - :math:`\alpha_\mathrm{decline}`
         - Positive asymptotic decline index.
       * - ``smoothness``
         - :math:`s`
         - Sharpness of the transition between power-law branches.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "rise_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive asymptotic rise index.",
            latex=r"\alpha_\mathrm{rise}",
        ),
        "decline_index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive asymptotic decline index.",
            latex=r"\alpha_\mathrm{decline}",
        ),
        "smoothness": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=2.0 * u.dimensionless_unscaled,
            description="Sharpness of the transition between power-law branches.",
            latex=r"s",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        rise_index = parameters["rise_index"]
        decline_index = parameters["decline_index"]
        smoothness = parameters["smoothness"]

        # For
        #
        #   f(y) = [y^(-s a) + y^(s b)]^(-1/s),
        #
        # the maximum occurs at
        #
        #   y_peak = (a / b)^[1 / (s(a+b))].
        #
        # Scale y so that t=t_peak corresponds exactly to y=y_peak.
        log_y_peak = np.log(rise_index / decline_index) / (
            smoothness * (rise_index + decline_index)
        )

        with np.errstate(divide="ignore"):
            log_y = np.log(t / t_peak) + log_y_peak

        log_shape = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y,
            smoothness * decline_index * log_y,
        )

        log_shape_peak = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y_peak,
            smoothness * decline_index * log_y_peak,
        )

        return np.log(amplitude) + log_shape - log_shape_peak


class DelayedExponentialLightcurve(Lightcurve):
    r"""
    A polynomial rise followed by an exponential decline.

    .. math::

        L(t) =
        A
        \left(\frac{t}{t_\mathrm{peak}}\right)^\alpha
        \exp\left[
            \alpha\left(1-\frac{t}{t_\mathrm{peak}}\right)
        \right],
        \qquad t > 0.

    This is a gamma-like transient pulse. The chosen parameterization makes

    .. math::

        L(t_\mathrm{peak}) = A

    exactly, while :math:`L \rightarrow 0` as :math:`t \rightarrow 0^+` and
    the late-time emission declines exponentially.

    The positive shape parameter :math:`\alpha` controls both the steepness of
    the rise and the relation between the peak time and exponential timescale.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``shape``
         - :math:`\alpha`
         - Positive dimensionless pulse-shape parameter.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "shape": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=2.0 * u.dimensionless_unscaled,
            description="Positive dimensionless pulse-shape parameter.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        shape = parameters["shape"]

        x = t / t_peak

        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(amplitude) + shape * np.log(x) + shape * (1.0 - x)


class LogNormalPulseLightcurve(Lightcurve):
    r"""
    A log-normal pulse in time.

    .. math::

        L(t) =
        A
        \exp\left[
            -\frac{1}{2}
            \left(
                \frac{\ln(t/t_\mathrm{peak})}{\sigma}
            \right)^2
        \right],
        \qquad t > 0.

    The pulse peaks exactly at ``t_peak`` with

    .. math::

        L(t_\mathrm{peak}) = A.

    Unlike a Gaussian pulse in linear time, a log-normal pulse is intrinsically
    asymmetric and has support only at positive times. The dimensionless width
    ``sigma`` controls the width in logarithmic time.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Peak bolometric luminosity.
       * - ``t_peak``
         - :math:`t_\mathrm{peak}`
         - Time of peak luminosity.
       * - ``sigma``
         - :math:`\sigma`
         - Width of the pulse in logarithmic time.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Peak bolometric luminosity.",
            latex=r"A",
        ),
        "t_peak": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Time of peak luminosity.",
            latex=r"t_\mathrm{peak}",
        ),
        "sigma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.4),
            scale=0.5 * u.dimensionless_unscaled,
            description="Width of the pulse in logarithmic time.",
            latex=r"\sigma",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_peak = parameters["t_peak"]
        sigma = parameters["sigma"]

        with np.errstate(divide="ignore"):
            log_time = np.log(t / t_peak)

        return np.log(amplitude) - 0.5 * (log_time / sigma) ** 2


class PlateauPowerLawLightcurve(Lightcurve):
    r"""
    A constant plateau followed by a power-law decline.

    .. math::

        L(t) =
        A
        \begin{cases}
            1, & 0 \le t \le t_\mathrm{break}, \\[4pt]
            \left(\dfrac{t}{t_\mathrm{break}}\right)^{-\alpha},
                & t > t_\mathrm{break}.
        \end{cases}

    This provides a minimal phenomenological description of a transient with a
    sustained plateau followed by scale-free fading, such as some GRB afterglow
    and engine-powered transient light curves.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Plateau bolometric luminosity.
       * - ``t_break``
         - :math:`t_\mathrm{break}`
         - Time at which the plateau transitions to a power-law decline.
       * - ``index``
         - :math:`\alpha`
         - Positive post-plateau power-law decline index.
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Plateau bolometric luminosity.",
            latex=r"A",
        ),
        "t_break": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.0 * u.day,
            description="Time at which the plateau transitions to a power-law decline.",
            latex=r"t_\mathrm{break}",
        ),
        "index": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1.5 * u.dimensionless_unscaled,
            description="Positive post-plateau power-law decline index.",
            latex=r"\alpha",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        **parameters: CGSParameterValue,
    ) -> NDArray[np.float64]:
        amplitude = parameters["amplitude"]
        t_break = parameters["t_break"]
        index = parameters["index"]

        with np.errstate(divide="ignore"):
            log_decline = -index * np.log(t / t_break)

        return np.log(amplitude) + np.where(
            t <= t_break,
            0.0,
            log_decline,
        )


class VillarLightcurve(Lightcurve):
    r"""
    A parametric supernova-like light curve.

    .. math::

        L(t) =
        A \times
        \begin{cases}
            \dfrac{1 + \beta(t - t_0)}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
                & t < t_1, \\[8pt]
            \dfrac{(1 + \beta\gamma)\,
                \exp[-(t-t_1)/\tau_\mathrm{fall}]}{1 + \exp[-(t-t_0)/\tau_\mathrm{rise}]},
                & t \ge t_1,
        \end{cases}

    with :math:`t_1 = t_0 + \gamma`. A logistic rise (timescale
    :math:`\tau_\mathrm{rise}`) turns on around :math:`t_0`; the light curve
    then declines linearly with slope :math:`\beta` (a plateau, for small
    :math:`|\beta|`) until :math:`t_1`, after which it switches to an
    exponential decline with timescale :math:`\tau_\mathrm{fall}`. The two
    branches agree exactly at :math:`t_1`, by construction.

    This is the parametric form used by :footcite:t:`2019ApJ...884...83V` to
    fit multi-band supernova photometry, adapted here to a bolometric
    luminosity rather than a per-band flux.

    .. rubric:: Parameters

    The light curve parameters are summarized below.

    .. list-table::
       :header-rows: 1
       :widths: 18 18 64

       * - Parameter
         - Symbol
         - Description
       * - ``amplitude``
         - :math:`A`
         - Overall luminosity normalization.
       * - ``t0``
         - :math:`t_0`
         - Reference time at which the logistic rise is centered.
       * - ``gamma``
         - :math:`\gamma`
         - Duration of the plateau, measured from t0.
       * - ``beta``
         - :math:`\beta`
         - Linear slope of the plateau.
       * - ``tau_rise``
         - :math:`\tau_\mathrm{rise}`
         - Logistic rise timescale.
       * - ``tau_fall``
         - :math:`\tau_\mathrm{fall}`
         - Exponential decline timescale, after the plateau.

    See Also
    --------
    m4opt.models.supernovae.VillarCoolingBlackbodySED

    References
    ----------
    .. footbibliography::
    """

    _LIGHTCURVE_TYPE = "bolometric"

    _DEFAULT_PARAMETERS: ClassVar[dict[str, Parameter]] = {
        "amplitude": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=1e43 * _BOL_LUM_UNIT,
            description="Overall luminosity normalization.",
            latex=r"A",
        ),
        "t0": Parameter(
            prior=NormalPrior(mean=0.0, sigma=1.0),
            scale=1.0 * u.day,
            description="Reference time at which the logistic rise is centered.",
            latex=r"t_0",
        ),
        "gamma": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=10.0 * u.day,
            description="Duration of the plateau, measured from t0.",
            latex=r"\gamma",
        ),
        "beta": Parameter(
            prior=NormalPrior(mean=0.0, sigma=0.5),
            scale=1e-2 / u.day,
            description="Linear slope of the plateau.",
            latex=r"\beta",
        ),
        "tau_rise": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=5.0 * u.day,
            description="Logistic rise timescale.",
            latex=r"\tau_\mathrm{rise}",
        ),
        "tau_fall": Parameter(
            prior=LogNormalPrior(mean=0.0, sigma=0.5),
            scale=30.0 * u.day,
            description="Exponential decline timescale, after the plateau.",
            latex=r"\tau_\mathrm{fall}",
        ),
    }

    @classmethod
    def _eval(
        cls,
        t: NDArray[np.float64],
        *,
        amplitude: CGSParameterValue,
        t0: CGSParameterValue,
        gamma: CGSParameterValue,
        beta: CGSParameterValue,
        tau_rise: CGSParameterValue,
        tau_fall: CGSParameterValue,
    ) -> NDArray[np.float64]:
        x = t - t0
        t1 = t0 + gamma
        is_rise = t < t1

        with np.errstate(invalid="ignore"):
            linear = np.where(
                is_rise,
                1.0 + beta * x,
                (1.0 + beta * gamma) * np.exp(-(t - t1) / tau_fall),
            )
            log_shape = np.log(linear) - np.logaddexp(0.0, -x / tau_rise)

        return np.log(amplitude) + log_shape
