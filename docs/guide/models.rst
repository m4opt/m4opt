********************************
Spectral Models (`m4opt.models`)
********************************

One of the core features of |M4OPT|, provided by :mod:`m4opt.synphot`, is the ability
to adaptively adjust planned exposure times to achieve a desired signal-to-noise
ratio. The same machinery can also be used for end-to-end simulations of instrument
observations, given a model for the brightness of the source.

:mod:`m4opt.models` provides the infrastructure for constructing these source
models. It defines a small set of composable abstractions for spectral shapes,
bolometric light curves, and full spectral energy distributions, together with a
curated library of concrete transient models: generic light curve shapes,
blackbody and power-law spectra, and composite models for supernovae and tidal
disruption events. See :doc:`synphot` for how the models defined here are turned
into simulated observations.

.. note::

    :mod:`m4opt.models` is designed to be easily extensible. Even if the model
    that you need is not already implemented, it is usually straightforward to
    write it yourself by subclassing one of the base classes described in
    `Building Custom Models`_.

Every model is *probabilistic*: each of its physical parameters carries a prior
distribution, so a model can either be evaluated at fixed, literal parameter
values, or sampled to produce a whole population of simulated sources, for
example to test how sensitive a survey strategy is to the diversity of a
transient population, rather than to one fiducial light curve.

Using Spectral Models
======================

A spectral model is, at its core, a rule for the spectral luminosity

.. math::

    L_\nu(\nu, t \mid \Theta)

as a function of observer-frame frequency :math:`\nu`, time since some reference
epoch :math:`t` (for a transient, usually the time of explosion), and a set of
named physical parameters :math:`\Theta`: for example, the peak luminosity, the
photospheric temperature, or the rise and decay timescales.

Crucially, a model is not tied to one fixed choice of :math:`\Theta`. Every
parameter carries a *prior* distribution, so the same model can either be
evaluated at one particular set of literal parameter values, or sampled to
produce a whole population of realizations at once. From that single rule for
:math:`L_\nu`, a model gives you everything else you would want to observe:
fluxes, AB magnitudes, bolometric luminosities, and band-integrated
quantities.

Models come in three flavors, depending on what they vary with:

- :class:`~m4opt.models.Spectrum`: a spectral shape that depends on frequency
  only, such as a blackbody.
- :class:`~m4opt.models.Lightcurve`: a bolometric light curve that depends on
  time only.
- :class:`~m4opt.models.SpectralModel`: a full spectral energy distribution
  that depends on both frequency and time.

Most users only ever need to work with concrete :class:`~m4opt.models.SpectralModel`
subclasses; :class:`~m4opt.models.Spectrum` and :class:`~m4opt.models.Lightcurve`
are the building blocks that a :class:`~m4opt.models.SpectralModel` is often
assembled from (see `Building Custom Models`_).

The curated model library is organized by submodule, and every class in it is
also importable directly from :mod:`m4opt.models`:

- :mod:`m4opt.models.spectra`: basic spectral shapes,
  :class:`~m4opt.models.BlackbodySpectrum`,
  :class:`~m4opt.models.PowerLawSpectrum`, and
  :class:`~m4opt.models.BrokenPowerLawSpectrum`.
- :mod:`m4opt.models.lightcurves`: generic light curve shapes, such as
  :class:`~m4opt.models.FREDLightcurve` (fast-rise, exponential-decay),
  :class:`~m4opt.models.BazinLightcurve`, and
  :class:`~m4opt.models.VillarLightcurve`, among others.
- :mod:`m4opt.models.supernovae`: composite supernova models, such as
  :class:`~m4opt.models.VillarCoolingBlackbodySED`.
- :mod:`m4opt.models.tdes`: composite tidal disruption event models, such as
  :class:`~m4opt.models.VanVelzenTDESED`.

Construct a model with no arguments to get an instance with all of its default,
physically-motivated priors:

    >>> from m4opt.models import VanVelzenTDESED
    >>> model = VanVelzenTDESED()

To override individual parameters at construction time, see `Adjusting
Parameters`_ below.

As a basic example, here is how to compute the flux of a model at one particular
frequency and time, given literal values for every parameter:

    >>> from astropy import units as u
    >>> VanVelzenTDESED.flux(
    ...     1e15 * u.Hz, 10 * u.day, redshift=0.05,
    ...     amplitude=1e44 * u.erg / u.s,
    ...     temperature=2e4 * u.K,
    ...     sigma_rise=10 * u.day,
    ...     tau_decline=50 * u.day,
    ... )
    <Quantity 2.3797319e-30 erg / (Hz s cm2)>

Note that :meth:`~m4opt.models.SpectralModel.flux`, along with every other
evaluation method described in `Computing Fluxes and Luminosities`_, is a
:class:`classmethod`: it operates purely on the parameter values passed in as
keyword arguments, not on any particular instance's stored configuration.
Constructing an instance like ``model`` above is only necessary for *sampling*
parameters from their priors (see `Sampling Parameters`_). Once you have a set
of sampled or literal parameter values in hand, you evaluate the model by calling
one of these classmethods directly on the class itself, as above.

Adjusting Parameters
---------------------

Each named parameter of a model is backed by a :class:`~m4opt.models.Parameter`:
a :class:`~m4opt.models.core.priors.Prior` to sample from, a characteristic
physical :attr:`~m4opt.models.Parameter.scale`, and an optional
:attr:`~m4opt.models.Parameter.transform` that lets the prior be defined on a
more convenient variable than the physical value itself (for example,
``transform="log10"`` lets a strictly positive, many-decades-wide parameter be
drawn from an ordinary Gaussian prior on its base-10 logarithm).

There are two ways to override a parameter's default when constructing a model:

- To replace the prior entirely, pass a whole new
  :class:`~m4opt.models.Parameter` instance as a constructor keyword argument:

      >>> from m4opt.models import Parameter
      >>> from m4opt.models.core.priors import UniformPrior
      >>> model = VanVelzenTDESED(
      ...     temperature=Parameter(
      ...         prior=UniformPrior(lower=3.5, upper=4.5),
      ...         scale=1 * u.K,
      ...         transform="log10",
      ...     )
      ... )

- To pin a parameter to one specific value instead of sampling it at all, pass a
  bare :class:`~astropy.units.Quantity` (or plain number, for a dimensionless
  parameter):

      >>> model = VanVelzenTDESED(temperature=2e4 * u.K)
      >>> model["temperature"].is_fixed
      True

  This is equivalent to calling :meth:`~m4opt.models.Parameter.fix` on the
  parameter object itself, which can also be used to release the parameter again
  with :meth:`~m4opt.models.Parameter.unfix`.

Every concrete model ships with physically-motivated default priors in its
``_DEFAULT_PARAMETERS``, and each model's own docstring includes a parameter
table listing every parameter's symbol, description, and default prior. See,
for example, :class:`~m4opt.models.BlackbodySpectrum` or
:class:`~m4opt.models.VanVelzenTDESED`.

If none of the built-in priors fit what you need, you can define your own by
subclassing :class:`~m4opt.models.core.priors.Prior`; see `Priors`_ in
`Developer Notes`_ for the full contract.

Computing Fluxes and Luminosities
----------------------------------

Every quantity a :class:`~m4opt.models.SpectralModel` can compute is available
through a consistent family of methods, distinguished by a suffix:

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - Suffix
     - Inputs / output
     - Example
   * - ``_log_cgs``
     - unit-free numbers in, natural log out
     - :meth:`~m4opt.models.SpectralModel.eval_log_cgs`
   * - ``_log``
     - physical `~astropy.units.Quantity` in, natural log out
     - :meth:`~m4opt.models.SpectralModel.eval_log`
   * - ``_cgs``
     - unit-free numbers in, linear scale out
     - :meth:`~m4opt.models.SpectralModel.eval_cgs`
   * - (none)
     - physical `~astropy.units.Quantity` in and out
     - :meth:`~m4opt.models.SpectralModel.eval`

Unless you are in a performance-sensitive inner loop that has already stripped
units for speed, use the unit-aware, no-suffix form: it is the least error-prone,
since Astropy will raise an error if you pass a quantity with incompatible units.
(Because a magnitude is already a logarithmic quantity, the ``mag*`` methods below
only come in ``_cgs`` and unit-aware forms; there is no separate ``_log`` variant.)

Within that family, there are two groups of methods, depending on whether they
describe the source itself or what a distant observer would see:

- **Rest-frame, undiluted quantities**: :meth:`~m4opt.models.SpectralModel.eval`,
  :meth:`~m4opt.models.SpectralModel.eval_bolometric`, and
  :meth:`~m4opt.models.SpectralModel.eval_spectrum` describe the source's own
  luminosity, with no distance or redshift involved. Reach for these when you are
  asking a rest-frame physics question, rather than "what would a detector
  actually see?"
- **Observer-frame, distance-diluted quantities**: :meth:`~m4opt.models.SpectralModel.flux`,
  :meth:`~m4opt.models.SpectralModel.flux_band`,
  :meth:`~m4opt.models.SpectralModel.mag`, and
  :meth:`~m4opt.models.SpectralModel.mag_band` describe the flux an observer
  would measure, redshifted and diluted by distance. These all take
  ``redshift=``, ``luminosity_distance=``, ``angular_diameter_distance=``, or
  ``proper_distance=`` (exactly one of the four) plus an optional ``cosmology=``
  to fix the source's distance; see the "basic example" above for a worked
  example using ``redshift``.

To get the flux in a particular bandpass rather than at a single frequency, use
:meth:`~m4opt.models.SpectralModel.flux_band` (or
:meth:`~m4opt.models.SpectralModel.mag_band` for its magnitude), which take a
``(nu, throughput)`` grid describing the bandpass and compute the
throughput-weighted mean flux over it; see `Building Synphot Models`_ for where
that grid typically comes from. :meth:`~m4opt.models.SpectralModel.flux` and
:meth:`~m4opt.models.SpectralModel.mag`, by contrast, evaluate the flux at one
exact frequency, without any bandpass weighting.

Similarly, to get the luminosity integrated over all frequencies rather than the
flux in one band, use :meth:`~m4opt.models.SpectralModel.eval_bolometric`
(:math:`L_\mathrm{bol}(t)`) or, for the normalized spectral shape alone (which
integrates to 1 over frequency at any fixed time), :meth:`~m4opt.models.SpectralModel.eval_spectrum`.

Sampling Parameters
--------------------

To draw a random realization of a model's parameters from their priors, call
``sample_parameters`` on an instance:

    >>> import numpy as np
    >>> model = VanVelzenTDESED()
    >>> samples = model.sample_parameters(size=5, rng=np.random.default_rng(0))
    >>> sorted(samples)
    ['amplitude', 'sigma_rise', 'tau_decline', 'temperature']

This returns a dictionary mapping each parameter name to an array of ``size``
physical-unit samples, one draw per parameter. To sample and evaluate the model
in one step, use :meth:`~m4opt.models.SpectralModel.simulate`, which is
equivalent to sampling the parameters and then calling
:meth:`~m4opt.models.SpectralModel.eval` on the result:

    >>> model.simulate(1e15 * u.Hz, 10 * u.day, size=5, rng=0).shape
    (5,)

Because ``sample_parameters`` always draws from whatever prior each parameter
currently has, no special handling is needed to sample with a custom prior:
simply override the parameter's prior at construction time, as described in
`Adjusting Parameters`_, and then call ``sample_parameters`` or
:meth:`~m4opt.models.SpectralModel.simulate` exactly as above. A parameter that
has been fixed with :meth:`~m4opt.models.Parameter.fix` is never sampled; every
draw simply returns its fixed value.

Building Synphot Models
-------------------------

Once you have a set of parameter values (literal, sampled, or a mix of both),
two methods turn a :class:`~m4opt.models.SpectralModel` into something the rest
of |M4OPT| can consume:

- :meth:`~m4opt.models.SpectralModel.as_astropy_model` wraps the model as an
  :class:`~astropy.modeling.Model`, with frequency (or wavelength) and time left
  as free inputs that you supply later, and its parameters bound to whatever
  values you passed in (which may themselves carry extra batch axes, for
  evaluating many realizations at once).
- :meth:`~m4opt.models.SpectralModel.as_source_spectrum` instead fixes the time
  to one particular value and returns a :class:`~synphot.spectrum.SourceSpectrum`, ready
  to hand directly to a :class:`~m4opt.synphot.Detector`. Because a
  :class:`~synphot.spectrum.SourceSpectrum` only makes sense as a real, physical flux, at
  least one of the ``redshift=``/``luminosity_distance=``/etc. keywords described
  in `Computing Fluxes and Luminosities`_ is required here.

Both methods share the same set of keywords for controlling exactly what
quantity is generated (wavelength or frequency, energy flux or photon flux, and
so on); see their reference documentation for the full set of options. Neither
method applies any foreground attenuation, such as Milky Way dust extinction:
multiply the returned spectrum by a separate spectral element for that, as
described in :doc:`synphot`.

See :doc:`synphot` for how a :class:`~synphot.spectrum.SourceSpectrum` built this way
plugs into exposure-time calculations via the ``observing`` context manager and
:class:`~m4opt.synphot.Detector`.

Simulating Observations
========================

Putting the pieces above together, simulating an observation of a population of
transients generally follows the same sequence of steps:

1. Pick a model class, and either sample or fix its parameters, one
   realization per simulated source (see `Sampling Parameters`_ and `Adjusting
   Parameters`_).
2. Pick a grid of observation times relative to each source's reference epoch
   (for example, days since explosion).
3. At each time, call :meth:`~m4opt.models.SpectralModel.as_source_spectrum` to
   turn that realization into a :class:`~synphot.spectrum.SourceSpectrum`.
4. Feed the resulting spectrum, together with the observing geometry, to
   :class:`m4opt.synphot.Detector` to compute exposure times or signal-to-noise
   ratios.

Because every step above is expressed as ordinary NumPy broadcasting, the whole
sequence vectorizes for free: sampling ``size=N`` parameter realizations and then
broadcasting an array-valued time (or frequency) through
:meth:`~m4opt.models.SpectralModel.eval`/:meth:`~m4opt.models.SpectralModel.as_source_spectrum`
simulates an entire population of sources at once, rather than one light curve at
a time. The only thing to keep track of is that :meth:`~m4opt.models.SpectralModel.as_source_spectrum`
does not insert batch axes for you: each batch dimension you want (one source
per sky-grid position, several times per source) needs its own explicit
trailing axis on both the sampled parameters and on ``t``, so that it lines up
with the wavelength axis the resulting spectrum will eventually be sampled at.

As a fully worked example, here is a small simulated UVEX survey: a handful of
core-collapse supernovae (:class:`~m4opt.models.VillarCoolingBlackbodySED`,
sampled from its default priors) go off at random sky positions, all at the same
explosion epoch :math:`t_0`. We then simulate UVEX observing every one of them
in both the NUV and FUV bands, densely at first to catch some of them while they
are still rising and then every 10 days out to 100 days after explosion,
computing the signal-to-noise ratio and the corresponding magnitude error at
each epoch (:math:`\sigma_\mathrm{mag} = 2.5 / (\ln 10 \cdot \mathrm{SNR})`), and
plot the resulting light curves against the underlying, continuous model:

.. plot::
    :include-source: True

    import numpy as np
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    from matplotlib import pyplot as plt

    from m4opt.missions import uvex
    from m4opt.models import VillarCoolingBlackbodySED
    from m4opt.synphot import observing

    # Chosen so that at least one event is caught while still on the rise.
    rng = np.random.default_rng(4)
    n_events = 4
    exptime = 900 * u.s
    # A fixed, local-volume distance for every event, typical of the nearby
    # core-collapse supernovae UVEX is designed to catch shortly after explosion.
    luminosity_distance = 20 * u.Mpc
    # A colorblind-safe categorical color pair, one per band.
    colors = {"NUV": "#2a78d6", "FUV": "#eb6834"}

    model = VillarCoolingBlackbodySED()

    # Random sky positions, uniform on the sphere.
    ra = rng.uniform(0, 360, n_events) * u.deg
    dec = np.rad2deg(np.arcsin(rng.uniform(-1, 1, n_events))) * u.deg
    target_coords = SkyCoord(ra, dec)

    # One parameter realization per event; every event shares the same
    # explosion epoch t0.
    raw_params = model.sample_parameters(n_events, rng=rng)
    t0 = Time("2025-06-01T00:00:00", scale="utc")

    # Dense sampling over the first week to catch the rise, then every 10 days.
    t_since_explosion = np.concatenate([[0.5, 1, 2, 4, 7], np.arange(10, 101, 10)]) * u.day
    t_dense = np.linspace(0.5, 100, 300) * u.day  # for the continuous model curve
    obs_times = t0 + t_since_explosion
    locations = uvex.observer_location(obs_times)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, tight_layout=True)

    for i, (ax, event_coord) in enumerate(zip(axes.ravel(), target_coords)):
        event_params = {name: value[i] for name, value in raw_params.items()}

        # A trailing axis on `t` keeps the time axis from colliding with the
        # wavelength axis synphot samples this spectrum at internally.
        source_spectrum = model.as_source_spectrum(
            t=t_since_explosion[:, np.newaxis],
            luminosity_distance=luminosity_distance,
            **event_params,
        )

        with observing(locations, event_coord, obs_times):
            # Whether UVEX can actually point at this event at each epoch,
            # given Earth-limb, Sun, and Moon avoidance.
            observable = uvex.constraints(locations, event_coord, obs_times)

            for band, color in colors.items():
                bandpass = uvex.detector.bandpasses[band]
                wave = bandpass.waveset
                nu = wave.to(u.Hz, equivalencies=u.spectral())
                throughput = bandpass(wave)

                # The continuous model light curve, drawn in the background.
                mag_dense = model.mag_band(
                    nu, throughput, t_dense,
                    luminosity_distance=luminosity_distance,
                    **event_params,
                )
                ax.plot(
                    t_dense.to_value(u.day), mag_dense.value,
                    color=color, alpha=0.35, lw=1.5, zorder=1,
                )

                # The actual simulated observations, with their SNR-derived
                # magnitude errors.
                mag = model.mag_band(
                    nu, throughput, t_since_explosion,
                    luminosity_distance=luminosity_distance,
                    **event_params,
                )
                snr = uvex.detector.get_snr(exptime, source_spectrum, band)

                with np.errstate(divide="ignore"):
                    mag_err = 2.5 / np.log(10) / snr
                detected = observable & (snr > 0)

                ax.errorbar(
                    t_since_explosion.to_value(u.day)[detected],
                    mag.value[detected],
                    yerr=mag_err[detected],
                    fmt="o",
                    color=color,
                    label=band,
                    zorder=2,
                )

        ax.set_title(f"event {i}")
        ax.invert_yaxis()
        ax.set_xlabel("time since explosion [day]")
        ax.set_ylabel("AB magnitude")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

Some points are simply missing from the light curves above: at those epochs,
``observable`` was ``False``, meaning UVEX's field of regard (checked here with
:attr:`~m4opt.missions.Mission.constraints`, the same constraint tree an actual
scheduling run enforces) ruled out pointing at that event, regardless of how
bright it was. Others trail off into large error bars as the supernova fades
below UVEX's sensitivity in that band. Event 0 shows the payoff of the denser
early cadence: its NUV points actually get brighter over the first several days,
tracing the rise of the underlying model curve rather than only ever catching
the decline, exactly the kind of trade-off this machinery exists to let you
explore before committing telescope time to it.

Building Custom Models
=========================

If none of the models in the curated library fit your source, the base classes
above are designed to make defining a new one straightforward. Start by picking
the narrowest base class that actually fits what you are modeling:

- :class:`~m4opt.models.Spectrum`: for a spectral shape alone, with no time
  dependence (like :class:`~m4opt.models.BlackbodySpectrum`).
- :class:`~m4opt.models.Lightcurve`: for a bolometric light curve alone, with
  no frequency dependence (like :class:`~m4opt.models.VillarLightcurve`).
- :class:`~m4opt.models.ComposedSpectralModel`: for a full spectral energy
  distribution whose *shape* does not change with time, built by pairing an
  existing :class:`~m4opt.models.Lightcurve` with an existing
  :class:`~m4opt.models.Spectrum` (like :class:`~m4opt.models.VanVelzenTDESED`,
  which pairs a Gaussian-rise/exponential-decay light curve with a
  constant-temperature blackbody).
- :class:`~m4opt.models.SpectralModel` directly: for a genuinely time-varying
  spectral shape, where no fixed pairing of one light curve and one spectrum
  will do (like :class:`~m4opt.models.VillarCoolingBlackbodySED`, whose
  blackbody temperature itself evolves with time; see its source for a worked
  example of this manual-composition pattern, including how it lets subclasses
  define supernova-subtype variants by overriding only the parameters whose
  priors differ between populations).

Whichever base class you choose, defining a new model comes down to two things:

1. Declare a class-level ``_DEFAULT_PARAMETERS`` dictionary, mapping each
   parameter's name to a :class:`~m4opt.models.Parameter` describing its default
   prior, scale, and (optionally) transform; see `Adjusting Parameters`_.
2. Implement ``_eval``: the natural log of the model's quantity (spectral
   luminosity, bolometric luminosity, or spectral shape, depending on the base
   class) in cgs units. This is the *only* method every model must implement;
   the whole public ``eval``/``eval_log``/``flux``/``mag``/... family described in
   `Computing Fluxes and Luminosities`_ is derived from it automatically. ``_eval``
   itself should stay simple: a classmethod using plain NumPy broadcasting, with
   no unit bookkeeping and no ``self``.

For a :class:`~m4opt.models.Spectrum` or a :class:`~m4opt.models.SpectralModel`,
quantities that involve an integral over frequency (a spectrum's normalization,
or a spectral model's bolometric luminosity and normalized shape) fall back to
numerical quadrature by default. If a closed-form expression exists, override the
corresponding primitive directly (``_eval_normalization`` for a
:class:`~m4opt.models.Spectrum`; ``_eval_bolometric``/``_eval_spectrum`` for a
:class:`~m4opt.models.SpectralModel`) for both speed and exactness.
:class:`~m4opt.models.PowerLawSpectrum` and :class:`~m4opt.models.BlackbodySpectrum`
are good worked examples of this.

Every new concrete model needs a test class that inherits from the matching
contract in ``m4opt.models.tests._contracts`` (``LightcurveContract``,
``SpectrumContract``, or ``SpectralModelContract``) and names the model class
under test::

    class TestMyLightcurve(LightcurveContract):
        model_class = MyLightcurve

The contract classes exercise the invariants every model of that kind must
satisfy (unit consistency across the ``eval*`` family, finiteness, correct
normalization, and so on), so this is usually all the testing a new model needs.
Each test module also enforces its own coverage: it is a test failure, not
merely a gap, if a new concrete model subclass is added without a matching
``Test*`` class.

Developer Notes
================

The following sections describe the machinery underneath the public API above,
for anyone extending :mod:`m4opt.models` itself rather than just using it.

Priors
------

:class:`~m4opt.models.core.priors.Prior` is the abstract base class every prior
distribution implements. A subclass supplies:

- ``_logpdf(x)``: the log-density at ``x``, the one method every prior must
  implement. Every other statistic (``pdf``, ``cdf``, ``logpdf``, ``logcdf``) is
  derived from it.
- ``_validate()``: called automatically after construction, to reject
  ill-formed parameters (for example, a non-positive standard deviation).
- ``support``: the ``(lower, upper)`` bounds of the distribution, defaulting to
  :math:`(-\infty, \infty)`.
- ``_sample(rng, size)``, optionally: a fast or exact sampler. If a subclass
  does not override this, sampling falls back to a generic numerical inversion
  sampler built from ``_logpdf`` alone, which works for any distribution but is
  slower than a closed-form sampler.

:mod:`m4opt.models.core.priors` ships a small catalog of concrete priors covering
the common cases: :class:`~m4opt.models.core.priors.ConstantPrior`,
:class:`~m4opt.models.core.priors.UniformPrior`,
:class:`~m4opt.models.core.priors.NormalPrior`,
:class:`~m4opt.models.core.priors.LogNormalPrior`,
:class:`~m4opt.models.core.priors.TruncatedNormalPrior`,
:class:`~m4opt.models.core.priors.ExponentialPrior`,
:class:`~m4opt.models.core.priors.PowerLawPrior`, and
:class:`~m4opt.models.core.priors.DiscretePrior`.

Parameters
----------

A :class:`~m4opt.models.Parameter` describes how one physical quantity is drawn:
a :class:`~m4opt.models.core.priors.Prior` to sample from, a characteristic
physical :attr:`~m4opt.models.Parameter.scale`, and an optional
:attr:`~m4opt.models.Parameter.transform`. Internally, sampling proceeds through
a *latent* variable :math:`z`, related to the physical value :math:`x` by

.. math::

    z = T(x / x_0), \qquad x = T^{-1}(z) \cdot x_0,

where :math:`x_0` is :attr:`~m4opt.models.Parameter.scale` and :math:`T` is
:attr:`~m4opt.models.Parameter.transform` (the identity by default). This is what
lets a strictly positive, many-decades-wide parameter be sampled from a
well-behaved :class:`~m4opt.models.core.priors.NormalPrior` via
``transform="log10"``, without every prior implementation needing to separately
handle scale and shape.

:meth:`~m4opt.models.Parameter.fix` and :meth:`~m4opt.models.Parameter.unfix`
pin and release a parameter's value, as described in `Adjusting Parameters`_;
:meth:`~m4opt.models.Parameter.fix` validates that the fixed value's units match
:attr:`~m4opt.models.Parameter.scale` and that it maps to a finite latent value.

Models
------

The shared machinery underlying :class:`~m4opt.models.Spectrum`,
:class:`~m4opt.models.Lightcurve`, and :class:`~m4opt.models.SpectralModel` lives
in a private common base class. It provides the ``Mapping[str, Parameter]``
interface used to look up a model instance's parameters by name, validation of a
subclass's ``_DEFAULT_PARAMETERS``, parameter packing to and from plain ordered
sequences (``pack_params_to_arrays`` and ``unpack_params_from_arrays``), and
parameter sampling (``sample_parameters``).

:class:`~m4opt.models.SpectralModel` itself adds the full ``eval``/``flux``/``mag``
family described in `Computing Fluxes and Luminosities`_, all derived from a
subclass's single ``_eval`` implementation, plus the numerical-quadrature
defaults for ``_eval_bolometric`` and ``_eval_spectrum`` that a subclass may
override with a closed form.

:class:`~m4opt.models.ComposedSpectralModel` builds a
:class:`~m4opt.models.SpectralModel` out of an existing
:class:`~m4opt.models.Lightcurve` and :class:`~m4opt.models.Spectrum` pair,
named by a subclass as ``_LIGHTCURVE_CLASS`` and ``_SPECTRUM_CLASS``:

.. code-block:: python

    class MySED(ComposedSpectralModel):
        _LIGHTCURVE_CLASS = FREDLightcurve
        _SPECTRUM_CLASS = BlackbodySpectrum

At class-definition time, ``__init_subclass__`` merges the two component
classes' ``_DEFAULT_PARAMETERS`` into ``MySED``'s own (any parameter the
subclass declares directly itself takes precedence, as an override on top of
that merge; this is how :class:`~m4opt.models.VanVelzenTDESED` replaces its
component classes' default priors with its own fitted ones), and rejects the
class definition if the two components share a parameter name. From then on,
``MySED`` behaves exactly like any other :class:`~m4opt.models.SpectralModel`
subclass, with one flat parameter namespace. No instance-level wiring of
components is needed, since every method here is a classmethod. Because both
component halves are already exact on their own, ``_eval_bolometric`` and
``_eval_spectrum`` are closed-form combinations of the two components' own
methods, and never fall back to numerical integration the way a plain
:class:`~m4opt.models.SpectralModel` subclass's defaults do.

.. Every class below is already documented once, flattened onto `m4opt.models`
   by the single `automodapi` call below (running `automodapi` on each
   submodule too would document every class twice). These bare `py:module`
   declarations exist only so that the submodule names themselves, referenced
   as :mod: targets above, resolve to a (contentless) page rather than being
   broken links.
.. py:module:: m4opt.models.spectra
.. py:module:: m4opt.models.lightcurves
.. py:module:: m4opt.models.supernovae
.. py:module:: m4opt.models.tdes
.. py:module:: m4opt.models.core.priors

.. automodapi:: m4opt.models
    :include-all-objects:
