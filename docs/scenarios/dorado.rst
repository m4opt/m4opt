Dorado
======

Dorado is a concept for an ultraviolet telescope on a small satellite designed
for target-of-opportunity follow-up of gravitational-wave events. Here are some
key statistics about the baseline design.

:Field of View:             7.1° x 7.1° square
:Central Wavelength:        195 nm
:Orbit:                     625 km circular sun-synchronous LEO
:Readout Overhead:          0 s [#f1]_
:Maximum Slew Velocity:     0.872° s\ :sup:`-1`
:Maximum Slew Acceleration: 0.244° s\ :sup:`-2`

..  todo:: Specify effective area curve, pixel scale, sharpness of PSF.

Objective
---------

The Dorado science operations center receives a gravitational-alert that
includes the time of the event and a 3D localization map providing the
probability density function of the location of the source as a function of
right ascension, declination, and luminosity distance.

..  todo:: Pick a specific example time and localization.

A model kilonova light curve gives the predicted luminosity of the source in
the Dorado band as a function of time. The Dorado threshold for detection is
modeled using the CCD S/N equation for PSF photometry, including attenuation
of the source due to spatially varying Galactic dust extinction and sky noise
due to spatially varying Galactic, zodiacal, and airglow backgrounds.

..  todo:: Pick model light curve. Elaborate on dust extinction and sky
    background.

Dorado must choose fields to observe from a fixed set of reference pointings in
order to simplify data processing (image subtraction). The set of reference
pointings should be designed to efficiently cover the entire sky without
excessive overlap. The reference pointing grid should be designed as part of
this example problem.

Dorado's objective is to plan up to 2 orbits of observations in order to
maximize the probability of detecting the source at least twice with S/N≥5.

Decision Variables
------------------

The decision variables consist of:

*   The number of observations
*   For each observation:

    -   Its pointing
    -   Its start time
    -   Its exposure time

Constraints
-----------

Dorado observations are subject to the following constraints:

*   The telescope must point:

    -   At least 28° from any part of the Earth directly illuminated by the Sun
    -   At least 6° from the Earth limb
    -   At least 46° from the center of the Sun
    -   At least 23° from the center of the Moon
    -   At least 10° from the Galactic plane

*   The trapped particle flux at the position of the spacecraft (estimated for
    solar maximum conditions) is:

    -   ≤1 cm\ :sup:`-2` s\ :sup:`-1` for protons with energies ≥20 MeV
    -   ≤100 cm\ :sup:`-2` s\ :sup:`-1` for electrons with energies ≥1 MeV


..  [#f1] Dorado has a frame transfer CCD. One image can be read out while the
    CCD is collecting photons for the next image, so there is no readout
    overhead between exposures.

Pseudocode
----------

.. code-block:: python

    from astropy import units as u
    import m4opt

    # Factory method or subclass?
    observer = m4opt.Observer.in_orbit('path/to/ephemeris.tle')

    # `when=m4opt.constraints.When.OBSERVING` is the default.
    # These constraints apply only during an observation.
    m4opt.constraints.SunSeparationConstraint(46 * u.deg).add(observer)
    m4opt.constraints.MoonSeparationConstraint(6 * u.deg).add(observer)
    m4opt.constraints.GalacticLatitudeConstraint(10 * u.deg).add(observer)
    m4opt.constraints.TrappedParticleFluxConstraint(
        flux=1*u.cm**-2*u.s**-1, energy=20*u.MeV,
        particle='p', solar='max').add(observer)
    m4opt.constraints.TrappedParticleFluxConstraint(
        flux=100*u.cm**-2*u.s**-1, energy=1*u.MeV,
        particle='e', solar='max').add(observer)

    # Constraints with when=m4opt.constraints.When.ALWAYS must
    # always be satisfied, otherwise we kill the instrument or spacecraft!
    m4opt.constraints.SunSeparationConstraint(10 * u.deg).add(
        observer, when=m4opt.constraints.When.ALWAYS)

    # load/define observer features
    instrument_settings = {"readNoise":4, "darkNoise":3, "nPix":1000, "gain":1.0, "nBins":10, "plate_scale":25, "other":(other)}
    observer.setInstrument(instrument_settings)

    # or preload set instrument
    observer.chooseInstrument("dorado")

    # set observer conditions
    zodiacal = m4opt.ZodiacalBackground()
    galactic = m4opt.GalacticBackground()
    airglow = m4opt.AirglowBackground()

    observer.background.add(zodiacal, galactic, airglow)
    observer.background.set_extinction(True) # dust extinction

    # get target set
    # requires location and spectrum, will we consider extinction?
    skymap = m4opt.Skymap('./file_of_skymap')
    targets = m4opt.targets_from_skymap(Skymap)

    # or just read from file
    #target_list = m4opt.read_targets('./file_to_targets')

    # need potential observing plan to evaluate?
    # given observing_plan = [targets, times]
    # and observer for instrument, we can do:

    # get list of exposure times for each target
    SNR_tol = 5.*u.dimensionless_unscaled

    et = []
    for target,time in zip(obs_plan.targets, obs_plan.times):
        et.append(m4opt.Exposure.exposure_time(target, SNR_tol, observer.Instrument, observer.background, time))

    observer.add(m4opt.constraints.TimeConstraint(obs_plan.times[2:end] - obs_plan.times[1:end-1], '<=', et[1:end-1]))

    # for exposure time calculation in Exposure(?) module
    def exposure_time(target, snr, instrument, background, time):

        source_count = get_source_count_rate(target, instrument.bandpass, extinction=background.extinction) * instrument.APERATURE_CORRECTION
        background_count = get_background_count_rate(instrument.bandpass, target.coords, time=time, night=isNight(time))

        return _exptime(snr, source_count, background_count, instrument.DARK_NOISE, instrument.READ_NOISE, instrument.NPIX)
