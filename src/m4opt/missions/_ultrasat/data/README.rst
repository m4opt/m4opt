Sky Grid
--------

ULTRASAT includes two predefined sky grids:

- **All-Sky Survey (ALLSS) grid**: Used during the first 6 months of the mission. Overlapping 7° fields ensure no point on the sky is farther than 7° from a grid center.

- **Low-Cadence Survey (LCS) grid** : 240 non-overlapping fields (7° radius) covering the entire sky, used for the low-cadence extragalactic survey.
  Each field includes visibility annotations (180-day or 45-day periods) and average UV extinction.

Both grids are based on the baseline survey strategy discussed within the ULTRASAT Working Groups
(https://www.weizmann.ac.il/ultrasat/for-scientists/working-groups/working-groups).

Throughput
----------

``throughput.ecsv`` is ULTRASAT's total throughput on axis, accounting for all
optical elements, the detector quantum efficiency, and obscuration. It
reproduces the values quoted in Table 1 of Shvartzvald et al. (2024): a
mean throughput of 0.2498 over the 230-290 nm operation waveband, and a mean
out-of-band transmission of 2.88e-5 above 300 nm.

It is taken from the tabulated throughput that the ULTRASAT team distributes
with `sncosmo <https://sncosmo.readthedocs.io>`_ as
``bandpasses/ultrasat/{Wavelength.dat,Rdeg.dat,ULTRASAT_TR.dat}``, which gives
the throughput on a grid of wavelength and radial distance from the optical
axis. We keep the innermost radial column and resample it at 1 A below 3200 A
and 10 A above, which reproduces integrated background counts to within 0.01%.
