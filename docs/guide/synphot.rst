**************************************
Synthetic Photometry (`m4opt.synphot`)
**************************************

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
time. You may want to leave the sky position and time unspecified when you are
constructing the model, to be specified at a later point in your code, or to be
specified implicitly by the scheduler. To do that, simply construct the model
components directly, without any target coordinate or time. For example::

    from m4opt.models.background import Airglow, ZodiacalBackground
    model = Airglow() + ZodiacalBackground()

Then, when you need to evaluate the model for a _specific_ target and time, you
can specify their values within a code block using a ``with:`` statement::

    from m4opt.models import observing
    with observing(observer_location=loc, target_coord=coord, obstime=time):
        model_value = model(1000 * u.angstrom)

See also the Examples section for
:class:`~m4opt.models.background.ZodiacalBackground`.

.. automodapi:: m4opt.synphot
.. automodapi:: m4opt.synphot.background
.. automodapi:: m4opt.synphot.extinction
