"""Generic phenomenological bolometric light curve shapes."""

import numpy as np
from astropy import units as u

from ._base import LightcurveModel
from ._parameters import PriorParameter
from ._priors import LogNormalPrior, NormalPrior

__all__ = (
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
)

_L_UNIT = u.erg / u.s


class TopHatLightcurve(LightcurveModel):
    """A constant luminosity for a fixed duration, zero before and after.

    .. math::

        L(t) = A \\text{ for } 0 \\le t \\le \\Delta t\\text{, else } 0
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    duration = PriorParameter(
        default=1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, duration):
        return np.where(time <= duration, amplitude.to_value(_L_UNIT), 0.0) * _L_UNIT


class GaussianPulseLightcurve(LightcurveModel):
    """A Gaussian pulse in luminosity, peaked at ``t_peak`` with ``L(t_peak) = A``.

    .. math::

        L(t) = A \\exp\\left[-\\frac{(t-t_\\mathrm{peak})^2}{2\\sigma^2}\\right]
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_peak = PriorParameter(
        default=1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.day),
    )
    sigma = PriorParameter(
        default=1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, t_peak, sigma):
        return amplitude * np.exp(-0.5 * ((time - t_peak) / sigma) ** 2)


class FREDLightcurve(LightcurveModel):
    """A fast-rise, exponential-decay pulse (the Norris et al. 1996 GRB pulse shape).

    .. math::

        L(t) = A \\exp\\left(2\\sqrt{\\tau_1/\\tau_2} - \\tau_1/t - t/\\tau_2\\right)

    Peaks exactly at :math:`t_\\mathrm{peak}=\\sqrt{\\tau_1\\tau_2}`, where
    :math:`L=A`; :math:`\\tau_1 \\ll \\tau_2` gives the characteristic fast
    rise, slow decay.
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    rise = PriorParameter(
        default=0.5 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(0.5), 0.5, u.day),
    )
    decay = PriorParameter(
        default=3.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(3.0), 0.5, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, rise, decay):
        # t = 0 divides by zero here; that's the correct limit (rise/t -> inf
        # drives L -> 0), not an error, so the warning is suppressed.
        with np.errstate(divide="ignore"):
            return amplitude * np.exp(
                2 * np.sqrt(rise / decay) - rise / time - time / decay
            )


class GREDLightcurve(LightcurveModel):
    """A Gaussian rise followed by an exponential decline, peaked at ``L=A``.

    .. math::

        L(t) = A \\times
        \\begin{cases}
            \\exp\\left[-(t-t_\\mathrm{peak})^2/2\\sigma^2\\right] & t \\le t_\\mathrm{peak} \\\\
            \\exp\\left[-(t-t_\\mathrm{peak})/\\tau\\right] & t > t_\\mathrm{peak}
        \\end{cases}

    ``t_peak`` is not itself free -- it is fixed at
    :math:`t_\\mathrm{peak}=5\\sigma_\\mathrm{rise}`.
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    sigma_rise = PriorParameter(
        default=0.5 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(0.5), 0.5, u.day),
    )
    tau_decline = PriorParameter(
        default=3.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(3.0), 0.5, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, sigma_rise, tau_decline):
        t_peak = 5 * sigma_rise
        rise = np.exp(-0.5 * ((time - t_peak) / sigma_rise) ** 2)
        decline = np.exp(-(time - t_peak) / tau_decline)
        return amplitude * np.where(time <= t_peak, rise, decline)


class BazinLightcurve(LightcurveModel):
    """A smooth asymmetric transient following the Bazin functional form.

    .. math::

        L(t) = A\\,\\frac{\\exp[-(t-t_0)/\\tau_\\mathrm{fall}]}
        {1+\\exp[-(t-t_0)/\\tau_\\mathrm{rise}]}

    ``amplitude`` is the Bazin normalization, not in general the exact peak
    luminosity; a finite peak requires :math:`\\tau_\\mathrm{fall} >
    \\tau_\\mathrm{rise}`.
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t0 = PriorParameter(
        default=10.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(10.0), 0.5, u.day),
    )
    rise = PriorParameter(
        default=5.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(5.0), 0.5, u.day),
    )
    fall = PriorParameter(
        default=30.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(30.0), 0.3, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, t0, rise, fall):
        x = ((time - t0) / u.day).to_value(u.dimensionless_unscaled)
        rise_days = rise.to_value(u.day)
        fall_days = fall.to_value(u.day)
        log_shape = -x / fall_days - np.logaddexp(0.0, -x / rise_days)
        return amplitude * np.exp(log_shape)


class PowerLawLightcurve(LightcurveModel):
    """A power-law decline beginning at a reference time, :math:`L(t_\\mathrm{ref})=A`.

    .. math::

        L(t) = A(t/t_\\mathrm{ref})^{-\\alpha}, \\quad t \\ge t_\\mathrm{ref}
        \\text{, else } 0
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_ref = PriorParameter(
        default=1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.day),
    )
    index = PriorParameter(
        default=1.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_ref, index):
        with np.errstate(divide="ignore", invalid="ignore"):
            shape = (time / t_ref).to_value(u.dimensionless_unscaled) ** (-index)
        return np.where(time >= t_ref, amplitude * shape, 0.0 * _L_UNIT)


class BrokenPowerLawLightcurve(LightcurveModel):
    """A sharply broken power-law transient with rise and decline, peaked at ``L=A``.

    .. math::

        L(t) = A(t/t_\\mathrm{peak})^{\\alpha_\\mathrm{rise}} \\text{ for }
        t \\le t_\\mathrm{peak}\\text{, else } A(t/t_\\mathrm{peak})^{-\\alpha_\\mathrm{decline}}

    Both indices are positive.
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_peak = PriorParameter(
        default=10.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(10.0), 0.5, u.day),
    )
    rise_index = PriorParameter(
        default=2.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(2.0), 0.5, u.dimensionless_unscaled),
    )
    decline_index = PriorParameter(
        default=1.5 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(1.5), 0.5, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_peak, rise_index, decline_index):
        with np.errstate(divide="ignore"):
            log_time = np.log((time / t_peak).to_value(u.dimensionless_unscaled))
        log_shape = np.where(
            time <= t_peak, rise_index * log_time, -decline_index * log_time
        )
        return amplitude * np.exp(log_shape)


class SmoothBrokenPowerLawLightcurve(LightcurveModel):
    """A smoothly broken power-law transient, peaked at ``L(t_peak) = A``.

    Asymptotes to :math:`t^{\\alpha_\\mathrm{rise}}` for :math:`t \\ll
    t_\\mathrm{peak}` and :math:`t^{-\\alpha_\\mathrm{decline}}` for :math:`t
    \\gg t_\\mathrm{peak}`, with the positive ``smoothness`` parameter
    controlling the sharpness of the transition (larger is sharper).
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_peak = PriorParameter(
        default=10.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(10.0), 0.5, u.day),
    )
    rise_index = PriorParameter(
        default=2.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(2.0), 0.5, u.dimensionless_unscaled),
    )
    decline_index = PriorParameter(
        default=1.5 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(1.5), 0.5, u.dimensionless_unscaled),
    )
    smoothness = PriorParameter(
        default=2.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(2.0), 0.4, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_peak, rise_index, decline_index, smoothness):
        # f(y) = [y^(-s a) + y^(s b)]^(-1/s) peaks at y_peak = (a/b)^[1/(s(a+b))];
        # shift so t=t_peak lands exactly at y=y_peak.
        log_y_peak = np.log(rise_index / decline_index) / (
            smoothness * (rise_index + decline_index)
        )
        with np.errstate(divide="ignore"):
            log_y = (
                np.log((time / t_peak).to_value(u.dimensionless_unscaled)) + log_y_peak
            )
        log_shape = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y, smoothness * decline_index * log_y
        )
        log_shape_peak = -(1.0 / smoothness) * np.logaddexp(
            -smoothness * rise_index * log_y_peak,
            smoothness * decline_index * log_y_peak,
        )
        return amplitude * np.exp(log_shape - log_shape_peak)


class DelayedExponentialLightcurve(LightcurveModel):
    """A polynomial rise followed by an exponential decline, peaked at ``L=A``.

    .. math::

        L(t) = A(t/t_\\mathrm{peak})^\\alpha
        \\exp\\left[\\alpha(1-t/t_\\mathrm{peak})\\right]
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_peak = PriorParameter(
        default=5.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(5.0), 0.5, u.day),
    )
    shape = PriorParameter(
        default=2.0 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(2.0), 0.5, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_peak, shape):
        x = (time / t_peak).to_value(u.dimensionless_unscaled)
        with np.errstate(divide="ignore", invalid="ignore"):
            return amplitude * np.exp(shape * np.log(x) + shape * (1.0 - x))


class LogNormalPulseLightcurve(LightcurveModel):
    """A log-normal pulse in time, peaked at ``L(t_peak) = A``.

    .. math::

        L(t) = A\\exp\\left[-\\frac{1}{2}\\left(\\frac{\\ln(t/t_\\mathrm{peak})}
        {\\sigma}\\right)^2\\right]
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_peak = PriorParameter(
        default=5.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(5.0), 0.5, u.day),
    )
    sigma = PriorParameter(
        default=0.5 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(0.5), 0.4, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_peak, sigma):
        with np.errstate(divide="ignore"):
            log_time = np.log((time / t_peak).to_value(u.dimensionless_unscaled))
        return amplitude * np.exp(-0.5 * (log_time / sigma) ** 2)


class PlateauPowerLawLightcurve(LightcurveModel):
    """A constant plateau followed by a power-law decline.

    .. math::

        L(t) = A \\text{ for } t \\le t_\\mathrm{break}\\text{, else }
        A(t/t_\\mathrm{break})^{-\\alpha}
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t_break = PriorParameter(
        default=1.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(1.0), 0.5, u.day),
    )
    index = PriorParameter(
        default=1.5 * u.dimensionless_unscaled,
        unit=u.dimensionless_unscaled,
        prior=LogNormalPrior(np.log(1.5), 0.5, u.dimensionless_unscaled),
    )

    @staticmethod
    def evaluate(time, amplitude, t_break, index):
        with np.errstate(divide="ignore"):
            log_decline = -index * np.log(
                (time / t_break).to_value(u.dimensionless_unscaled)
            )
        return amplitude * np.exp(np.where(time <= t_break, 0.0, log_decline))


class VillarLightcurve(LightcurveModel):
    """A parametric supernova-like light curve (Villar et al. 2019).

    .. math::

        L(t) = A \\times
        \\begin{cases}
            \\dfrac{1+\\beta(t-t_0)}{1+\\exp[-(t-t_0)/\\tau_\\mathrm{rise}]}, & t < t_1 \\\\[6pt]
            \\dfrac{(1+\\beta\\gamma)\\exp[-(t-t_1)/\\tau_\\mathrm{fall}]}
                {1+\\exp[-(t-t_0)/\\tau_\\mathrm{rise}]}, & t \\ge t_1
        \\end{cases}

    with :math:`t_1=t_0+\\gamma`: a logistic rise turns on around
    :math:`t_0`, followed by a linear plateau of slope :math:`\\beta` until
    :math:`t_1`, then an exponential decline. The two branches agree exactly
    at :math:`t_1`. Originally fit to multi-band supernova photometry;
    adapted here to a bolometric luminosity.
    """

    amplitude = PriorParameter(
        default=1e43 * _L_UNIT,
        unit=_L_UNIT,
        prior=LogNormalPrior(np.log(1e43), 0.5, _L_UNIT),
    )
    t0 = PriorParameter(
        default=0.0 * u.day, unit=u.day, prior=NormalPrior(0.0, 1.0, u.day)
    )
    gamma = PriorParameter(
        default=10.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(10.0), 0.5, u.day),
    )
    beta = PriorParameter(
        default=0.0 * u.day**-1,
        unit=u.day**-1,
        prior=NormalPrior(0.0, 0.5e-2, u.day**-1),
    )
    tau_rise = PriorParameter(
        default=5.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(5.0), 0.5, u.day),
    )
    tau_fall = PriorParameter(
        default=30.0 * u.day,
        unit=u.day,
        prior=LogNormalPrior(np.log(30.0), 0.5, u.day),
    )

    @staticmethod
    def evaluate(time, amplitude, t0, gamma, beta, tau_rise, tau_fall):
        x = time - t0
        t1 = t0 + gamma
        is_rise = time < t1
        with np.errstate(invalid="ignore"):
            linear = np.where(
                is_rise,
                (1.0 + beta * x).to_value(u.dimensionless_unscaled),
                ((1.0 + beta * gamma) * np.exp(-(time - t1) / tau_fall)).to_value(
                    u.dimensionless_unscaled
                ),
            )
            log_shape = np.log(linear) - np.logaddexp(
                0.0, (-x / tau_rise).to_value(u.dimensionless_unscaled)
            )
        return amplitude * np.exp(log_shape)
