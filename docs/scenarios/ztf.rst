ZTF
===

Zwicky Transient Facility (ZTF) is a ground-based telescope run by Caltech at
Palomar Observatory. It was designed for wide-field time-domain surveys and
often performs Target of Opportunity (ToO) follow-up of gravitational wave
events

:Field of View:             7°.50 N-S x 7°.32 E-W; 47.7 deg :math:`^2` light
                            sensitive area
:Filters:                   r-, g-, i-
:Location:                  Palomar Observatory
:Readout Overhead:          8.2 s
:Maximum Slew Velocity:     2.5° s\ :sup:`-1`
:Maximum Slew Acceleration: 0.4° s\ :sup:`-2`

..  todo:: Other important values folks think I should add?

Objective
---------

ZTF receives a gravitational wave alert, which has an event time and a 3-D
probability density skymap :math:`w` in right ascension, declination and
luminosity distance. The skymap, along with a a model lightcurve :math:`L`
that describes the expected time dependence of the source's flux and a time
interval :math:`t` to :math:`t + \Delta T` to observe during are used to
produce a schedule.

This schedule is produced by choosing a subset :math:`C` of the set
of 1778 fixed reference images for ZTF, denoted :math:`P` and arranging them
into a schedule :math:`S`. Restrictions :math:`S` must fulfil to be a valid
schedule are laid out in the Constraints section.

The objective when creating this schedule is to maximize
the sum of the probability density observed since it correlates with detection
probability. For a pixel to be marked as observed, an exposure long enough
to have sufficient signal-to-noise ratio (≥ 5) needs to be made.
The length of the exposure needed primarily depends on :math:`L` and how long
after the event the exposure is taken.

..  todo:: Use the same CCD S/N equation for PSF photometry as Dorado? It seems
           like this is a larger question for the group as a whole

Decision Variables
------------------

The decision variables consist of:

*   The number of observations
*   For each observation:

    -   Its pointing
    -   Its start time
    -   Its exposure time
*   For each HEALPix pixel on the sky:

    -   Whether it is observed

Constraints
-----------

ZTF observations are subject to the following constraints:

*   The telescope must

    -   observe fields with an airmass below 2.5
    -   observe fields greater than 20 degrees from the center of the moon
    -   observe fields not in ZTF pointing limits

        -   \|HA\| < 5.95 hours
        -   Not HA < -17.6 deg and Dec < -22 deg
        -   Not west of HA -17.6 deg, Dec < -45 deg
        -   Not \|HA\| > 3 deg and Dec < -46
        -   Dec :math:`\leq` 87.5

*   The telescope can only make observations after the sun is 18° below
    the horizon
*   Observations can only be made during the provided time interval
*   All fields in :math:`C` must be observed twice, once in r- and once in g-
    bands with at least 30 minutes of time between.
*   For a HEALPix pixel to be marked as observed, an exposure long enough to
    have SNR ≥ 5 of it must be taken.

Pseudocode
----------

.. code-block:: python

    from astropy import units as u
    from ligo.skymap.io import read_sky_map
    import m4opt

    observer = m4opt.Observer.on_ground('path/to/ztf.tle')
    skymap, data = read_sky_map('path/to/skymap.fits')
    start_time = data['gpstime']

    m4opt.objective.ToOObjective(skymap).add(observer)

    m4opt.constraints.SunElevationConstraint(-18 * u.deg).add(observer)
    m4opt.constraints.MoonSeparationConstraint(20 * u.deg).add(observer)
    m4opt.constraints.AirmassConstraint(2.5).add(observer)
    m4opt.constraints.TimeConstraint(
        start=start_time, end=start_time + 24 * u.hour).add(observer)
    m4opt.constraints.CadenceConstraint(
         count=2, time=30 * u.minute, filters=['r', 'g']).add(observer)
    m4opt.constraints.SNRConstraint(
        lightcurve='path/to/lightcurve.dat', SNR=5).add(observer)
    '''
    default pointing constraints automatically stored in ztf file, and
    no additional ones are to be added
    '''

    ...
