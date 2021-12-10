The SED Machine and SED Machine v2
==================================

The SED Machine is a combination imager and low-resolution spectrograph
System on the Palomar 1.6 meter telescope. A modified version is being built
for the Kitt Peak 2.1 meter telescope. The SED Machine is capable of
simultaneous u,g,r,i imaging in addition to the low-resolution, integral field
unit (IFU) spectrograph. The SED Machine v2 will have single shot imaging
in u, g, r, i, and z in addition to the IFU.

Here are some key statistics about the systems:

SED Machine:
:Latitude:             33.3563°
:Longitude:        -116.8648°
:Elevation:          1742 m

SED Machine v2:
:Latitude:             31.9599°
:Longitude:        -111.5997°
:Elevation:          2096 m

..  todo:: Specify effective sensitivities.

Objective
---------

In addition to ranked priority lists of individual objects from individual
instrument partners, both SEDM and SEDMv2 have public follow-up
allocations that require coordination. The scheduler must also be capable of
taking in higher priority observations (such as those arising from
a gravitational-alert) to interrupt and reschedule on-the-fly.

Each object will come with at least a right ascension, declination, and program
priority, and potentially a candidate brightness to adjust the required exposure
times.

..  todo:: Print out an example list of objects with these features.

The scheduler must choose observations consistent with both telescope
constraints (airmass, moon distance, etc.) and program allocations / priorities.
The objective is to minimize waste between the program allocations, in
addition to preventing any repeated observations in the public programs
between SEDM and SEDMv2.

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

SEDM and SEDMv2 observations are subject to the following constraints:

*   The telescope must point:

    -   At least an airmass of 3
    -   At least 30° from the center of the Moon


Pseudocode
----------

.. code-block:: python

.. todo:: the ifu is not considered here because I'm not clear if it has more constraints/mode considerations needed.

    sched = Scheduler(... possibly some scheduler-algorithm specific bits, but defaults should be good enough...)

    targets = [Target(...) for coo in targetcoos]
    for target, pri, mag in zip(targets, priorities, magnitudes):
        target.priority = pri
        target.flux = mag.to_flux()  # or maybe just magnitudes
        target.exptime = None if mag < blah else exptime_from_mag(mag)

    sched.add_constraint(AirmassConstraint(3, palomar)) #
    sched.add_constraint(AirmassConstraint(3, kpno))
    sched.add_constraint(MoonSeparationConstraint(30*u.deg, palomar))
    sched.add_constraint(MoonSeparationConstraint(30*u.deg, kpno))
    sched.add_constraint(GreedyPriorityConstraint())  # always favor higher priority when all other things are equal, say
    sched.add_constraint(ExptimeConstraint(defaulttimeforbrightesttargets, 'fromtarget', 'consecutive'))


    for observatory in (kpno, palomar):
        blocks = sched(targets, Time('date-to-schedule'), observatory)
        # if need to schedule multiple nights in the future
        for date in ...:
            blocks = sched(targets, blocks, Time(date), observatory)

        for block in blocks:
            print(block.target)
            print(block.start_time)
            print(block.total_time)
        plot_blocks(blocks)  # makes a night plan in plot form

        def on_too_interrupt(time_interrupt):
            return sched(targets, Time.now() + time_interrupt)
