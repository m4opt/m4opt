ZTF Target Of Opportunity Follow-up Scenario
============================================

This is a first draft at writing this up, with more of the bells and whistles
|M4OPT| is gonna have compared to MUSHROOMS

Given a probability density skymap *w* in right ascension and declination, a
model lightcurve *L* for the source, a time interval *t* to *t + ΔT*, and the
set of 1778 fixed pointings for ZTF, *P*, the goal is to choose a subset *C* of
fields in *P* that can produce a valid schedule and maximize the chance of
detecting the target. A schedule, *S*, is represented as a sequence of pairs of
an element of *P* and a real number, where the real number is the exposure time
plus any overhead time. Each exposure has a quality factor *q_i* depending on what
time an exposure begins, referencing *L*; how long an
exposure lasts; the amount of new probability density observed by an exposure,
referencing *w*; and other atmospheric and observing factors, such as airmass
and lunar proximity, referencing *t* and the RA and DEC associated with the
field observed.
