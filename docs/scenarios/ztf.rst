ZTF Target Of Opportunity Follow-up Scenario
============================================

This is a second draft at writing this up, with more of the bells and whistles
|M4OPT| is gonna have compared to MUSHROOMS

This is a target of opportunity search following up a GW localization map. 
Given a probability density skymap *w* in right ascension and declination, a
model lightcurve *L* for the source that describes the expected time 
dependence of the flux from the source, a time interval *t* to *t + ΔT*, and the
set *P* of 1778 fixed pointings for ZTF. Each exposure has a quality metric
*q_i* depending on several factors: what time an exposure begins; how long an
exposure lasts; the amount of probability density within by an exposure
footprint; and other atmospheric and observing factors, such as airmass
and lunar proximity. The goal is to choose a subset *C* of
fields in *P* that can produce a valid schedule and maximize the sum of
observation quality metrics. A valid schedule is one where every field in *C*
is observed twice during the given interval, once in r- and g- bands, with 30
minutes between observations under a realistic plan that considers factors like
telescope slew time and filter-change and readout overhead times.

B & Leo
