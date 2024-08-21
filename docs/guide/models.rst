*************************************
Sensitivity Modeling (`m4opt.models`)
*************************************

|M4OPT| can adaptively adjust the exposure times of planned observations in
order to obtain a desired signal-to-noise, given a model of the foreground
signal and the observing conditions that you provide. |M4OPT| can also be used
merely as a general-purpose astronomical exposure time calculator.

You specify the foreground signal and background noise using
:ref:`Astropy models <astropy:astropy-modeling>`. The :mod:`m4opt.models`
submodule provides a curated selection of Astropy model subclasses for modeling
the spectral denergy distributions of sources and the surface brightness
spectrum of the sky. Models can be combined by adding, subtracting,
multiplying, or dividing them.

Observing State
===============

Some models (particularly sky background models) vary with sky position and
time. You can construct such models by explicitly passing a fixed sky position
and time using the ``.at()`` class method. For example::

    from m4opt.models.background import ZodiacalBackground
    model = ZodiacalBackground.at(target_coord=coord, obstime=time)

However, sometimes you may want to leave the sky position and time unspecified
when you are the model, to be specified at a later point in your code, or to be
specified implicitly by the scheduler. To do that, simply construct the model
components directly, without any target coordinate or time. For example::

    from m4opt.models.background import Airglow, ZodiacalBackground
    model = Airglow() + ZodiacalBackground()

Then, when you need to evaluate the model for a _specific_ target and time, you
can specify their values within a code block using a ``with:`` statement::

    from m4opt.models import state
    with state.set_observing(target_coord=coord, obstime=time):
        model_value = model(1000 * u.angstrom)

See also the Examples section for
:class:`~m4opt.models.background.ZodiacalBackground`.

.. automodapi:: m4opt.models
    :no-inheritance-diagram:

.. automodapi:: m4opt.models.background
    :no-inheritance-diagram:
