*************************************************
Field of Regard Constraints (`m4opt.constraints`)
*************************************************

The field of regard is the region of the sky that a detector is allowed to
point at. These classes model various constraints on the field of regard, as
functions of the location of the detector in space (an
:class:`~astropy.coordinates.EarthLocation` instance), the target at which the
detector is pointed (a :class:`~astropy.coordinates.SkyCoord` instance), and
the time of the observation (a :class:`~astropy.time.Time` instance).

Sampled observing windows
=========================

:func:`~m4opt.constraints.visibility_windows` evaluates a constraint on an
explicit time grid and returns closed start/end pairs for every target.
Scalar targets return a one-element list; targets with no usable windows return
an empty ``(0, 2)`` time array. For example::

    from astropy import units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from m4opt.constraints import DeclinationConstraint, visibility_windows

    times = Time("2025-01-01") + [0, 1, 2, 3] * u.hour
    location = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)
    targets = SkyCoord([0, 90] * u.deg, [0, 60] * u.deg)
    windows = visibility_windows(
        DeclinationConstraint(-30 * u.deg, 30 * u.deg),
        location, targets, times,
        min_duration=30 * u.min,
        time_margin=1 * u.min,
    )

Pass sampled spacecraft locations from ``mission.observer_location(times)``
and ``mission.constraints`` to use the same interface for a mission.
``min_duration`` filters windows after ``time_margin`` removes time from both
ends. Include all overheads that must fit within the window in the minimum
occupied duration. An observation's entire occupied interval must fit inside
one returned pair; separate windows cannot be combined across interruptions.

Window boundaries use the first and last passing sample, without extrapolation.
The constraint may change between samples, so choose an appropriate time grid
and check scheduled exposures at finer spacing. A time margin shrinks sampled
windows; it does not guarantee continuous visibility. Transition refinement
and spacecraft-specific angular margins can be added separately.

.. automodapi:: m4opt.constraints
