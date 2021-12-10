ZTF Target Of Opportunity Follow-up Scenario
============================================

This is a target of opportunity search following up a GW localization map. 
Given a probability density skymap :math:`w` in right ascension and declination
, a model lightcurve :math:`L` for the source that describes the expected time
dependence of the flux, a time interval :math:`t` to :math:`t + \Delta T`, and
the set :math:`P` of 1778 fixed pointings for ZTF. Each exposure has a quality
metric :math:`q_i` depending on several factors: what time an exposure begins;
how long an exposure lasts; the amount of probability density within by an
exposure footprint; whether any of the exposure footprint has been previously
observed; and other atmospheric and observing factors, such as airmass and
lunar proximity. The goal is to choose a subset :math:`C` of fields in
:math:`P` that can produce a valid schedule and maximize the sum of observation
quality metrics. A valid schedule is one where every field in :math:`C` is
observed twice during the given interval, once in r- and once in g- bands,
with 30 minutes between observations under a realistic plan that considers
factors like telescope slew time and filter-change and readout overhead times.

B & Leo
