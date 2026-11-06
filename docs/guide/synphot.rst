**************************************
Synthetic Photometry (`m4opt.synphot`)
**************************************

|M4OPT| can adaptively adjust the exposure times of planned observations in
order to obtain a desired signal-to-noise, given a model of the foreground
signal and the observing conditions that you provide. |M4OPT| can also be used
merely as a general-purpose astronomical exposure time calculator.

You specify the foreground signal and background noise using
:ref:`Astropy models <astropy:astropy-modeling>`. The :mod:`m4opt.synphot`
submodule provides a curated selection of Astropy model subclasses for modeling
the spectral denergy distributions of sources and the surface brightness
spectrum of the sky. Models can be combined by adding, subtracting,
multiplying, or dividing them.

Bandpasses
==========

A bandpass is the fraction of incident photons that are detected as a function
of wavelength, combining the filter transmission, the throughput of the
optics, and the quantum efficiency of the detector. Bandpasses are
represented by :class:`synphot.spectrum.SpectralElement` instances. Here are
three ways to make one.

From tabular data
-----------------

If you have a table of wavelengths and throughputs, then create a bandpass
with the :class:`synphot.models.Empirical1D` model, which interpolates
linearly between the tabulated points::

    import numpy as np
    from astropy import units as u
    from synphot import Empirical1D, SpectralElement

    wavelength = np.asarray([3000, 3100, 3900, 4000]) * u.angstrom
    throughput = np.asarray([0, 0.8, 0.8, 0])
    bandpass = SpectralElement(
        Empirical1D, points=wavelength, lookup_table=throughput
    )

If the table is in a FITS or ASCII file in a format that synphot understands,
then you can read it directly with
:meth:`~synphot.spectrum.BaseSpectrum.from_file`::

    bandpass = SpectralElement.from_file("bandpass.fits")

From a named filter
-------------------

Use :func:`~m4opt.synphot.bandpass_from_svo` to look up a filter by name in the
`SVO Filter Profile Service <http://svo2.cab.inta-csic.es/theory/fps/>`_::

    from m4opt.synphot import bandpass_from_svo

    bandpass = bandpass_from_svo("Palomar/ZTF.r")

From analytic models
--------------------

You can build a bandpass from any one-dimensional model, such as
:class:`~synphot.models.Box1D` for an idealized top-hat filter or
:class:`~synphot.models.Gaussian1D`::

    from synphot import Box1D, Gaussian1D

    top_hat = SpectralElement(Box1D, amplitude=0.8, x_0=3500, width=1000)
    gaussian = SpectralElement(Gaussian1D, amplitude=0.8, mean=3500, stddev=200)

Bandpasses can be multiplied together. For example, here is a
filter with a constant 90% transmission combined with a detector whose quantum
efficiency is tabulated::

    filter_transmission = SpectralElement(Box1D, amplitude=0.9, x_0=3500, width=1000)
    detector_qe = SpectralElement(
        Empirical1D,
        points=[2500, 3500, 4500] * u.angstrom,
        lookup_table=[0.5, 0.7, 0.6],
    )
    bandpass = filter_transmission * detector_qe

.. plot::
    :caption: The bandpasses from the examples above.

    import numpy as np
    from astropy import units as u
    from astropy.visualization import quantity_support
    from matplotlib import pyplot as plt
    from synphot import Box1D, Empirical1D, Gaussian1D, SpectralElement

    bandpasses = {
        "tabular": SpectralElement(
            Empirical1D,
            points=[3000, 3100, 3900, 4000] * u.angstrom,
            lookup_table=[0, 0.8, 0.8, 0],
        ),
        "top hat": SpectralElement(Box1D, amplitude=0.8, x_0=3500, width=1000),
        "Gaussian": SpectralElement(Gaussian1D, amplitude=0.8, mean=3500, stddev=200),
        "filter × detector": SpectralElement(
            Box1D, amplitude=0.9, x_0=3500, width=1000
        )
        * SpectralElement(
            Empirical1D,
            points=[2500, 3500, 4500] * u.angstrom,
            lookup_table=[0.5, 0.7, 0.6],
        ),
    }

    wavelength = np.linspace(2500, 4500, 1000) * u.angstrom
    with quantity_support():
        ax = plt.axes()
        for label, bandpass in bandpasses.items():
            ax.plot(wavelength, bandpass(wavelength), label=label)
        ax.set_ylabel("Throughput")
        ax.legend()

To use a bandpass for exposure time calculations, pass it to
:class:`~m4opt.synphot.Detector` in the ``bandpasses`` dictionary.

Observing State
===============

Some models (particularly sky background models) vary with sky position and
time. You may want to leave the sky position and time unspecified when you are
constructing the model, to be specified at a later point in your code, or to be
specified implicitly by the scheduler. To do that, simply construct the model
components directly, without any target coordinate or time. For example::

    from m4opt.synphot.background import GalacticBackground, ZodiacalBackground
    model = GalacticBackground() + ZodiacalBackground()

Then, when you need to evaluate the model for a _specific_ target and time, you
can specify their values within a code block using a ``with:`` statement::

    from m4opt.synphot import observing
    with observing(observer_location=loc, target_coord=coord, obstime=time):
        model_value = model(1000 * u.angstrom)

See also the Examples section for
:class:`~m4opt.synphot.background.ZodiacalBackground`.

.. automodapi:: m4opt.synphot
.. automodapi:: m4opt.synphot.background
.. automodapi:: m4opt.synphot.extinction
