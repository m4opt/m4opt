*******
Changes
*******

2.12.1 (unreleased)
===================

- Add SCIP as a MILP solver backend, selected automatically if neither CPLEX
  nor Gurobi is installed, or explicitly with ``M4OPT_SOLVER=scip``. SCIP is
  open source under the Apache 2.0 license, so it can be deployed where an
  academic CPLEX or Gurobi license does not reach. Install it with ``pip
  install "m4opt[scip]"``.

- Add Gurobi as an alternative MILP solver backend. The solver is selected
  automatically from whichever of ``cplex`` or ``gurobipy`` is installed,
  preferring CPLEX, and can be chosen explicitly with the ``M4OPT_SOLVER``
  environment variable or :func:`m4opt.milp.set_backend`. Install the solver of
  your choice with ``pip install "m4opt[cplex]"`` or ``pip install
  "m4opt[gurobi]"``.

- Fix the ZTF sky grid, whose right ascensions were truncated by a fixed-width
  table reader so that all 1778 fields fell within 10 degrees of R.A. 0. ZTF
  schedules were empty as a result.

- Allow ``--bandpass`` to be repeated so that successive visits cycle through
  several bandpasses. Visits are grouped into contiguous blocks of a single
  bandpass, so a schedule exchanges the filter only once per block boundary.

- Add ``Mission.filter_exchange_time`` and set it to 110 s for ZTF.

- Use ULTRASAT's tabulated throughput curve rather than a Gaussian
  approximation, which had a red leak some four orders of magnitude too
  large and nearly doubled the predicted zodiacal background.

- Add ``EarthshineBackground``, a model of sunlight reflected off the Earth,
  scaled by the angular distance from the Earth's limb and by the solar
  illumination of that part of the limb, and include it in the ULTRASAT
  stray light budget.

- Fix the ULTRASAT readout noise, which was set to the noise budget's
  variance (6 e-/pix) rather than its RMS.

2.12.0 (2026-08-28)
===================

- Add ``EclipticLatitudeConstraint``.

- Add ``AntiSolarSeparationConstraint`` to keep targets away from the
  anti-solar point, where the nominal spacecraft roll angle is undefined.

- Fix ``LogicalNotConstraint``, which ignored its operand and always
  evaluated to a scalar ``True``.

- Fix ``footprint`` and ``footprint_healpix`` for empty compound ``Regions``.

- Add ``m4opt.utils.functional.apply`` method.

- Add the function ``count_intersect1d_combinations`` to calculate the overlap
  of pairwise combinations of arrays, parallelized with OpenMP.

  This operation is needed to calculate per-pixel cadence distributions from
  HEALPix observation footprints. The specialized parallel version is necessary
  because ``count_intersect1d`` cannot be effectively parallelized using Python
  techniques like ``multiprocessing``.

2.11.0 (2026-08-18)
===================

- Speed up ``solve_tsp`` by adding extra cuts.

- Update the UVEX chip gaps.

2.10.0 (2026-08-13)
===================

- Speed up ``count_intersect1d`` by 4-5x.

2.9.1 (2026-08-09)
==================

- Fix an issue with publishing abi3 wheels.

2.9.0 (2026-08-09)
==================

- Add ``count_intersect1d`` utility function for calculating cadence
  distributions accounting for field overlaps.

2.8.1 (2026-07-31)
==================

- Adjust UVEX sky grid for new FOV model. The old grid contained 5412 fields
  while the new grid contains 4962 fields.

2.8.0 (2026-07-31)
==================

- Add chip gaps for UVEX.

- Add Cerenkov background for ULTRASAT.

- ``m4opt.observer.EarthFixedObserverLocation`` is no longer a subclass of
  ``astropy.coordinates.EarthLocation``.

- Implement a helioecliptic longitude constraint.

2.7.0 (2026-06-29)
==================

- Raise a RuntimeError if the SGP4 package was installed without its
  well-tested binary implementation.

- Add support for jerk-limited slews.

- Fix some corner cases in the scheduler when no fields are observable.

2.6.0 (2026-04-05)
==================

- Add support for partitioning graphs with node and edge weights.

- Add function to convert circle sky regions to polygons.

- Add function to calculate approximate orientation of UVEX for a ground pass.

2.5.0 (2026-03-20)
==================

- Add support for pointlike FOVs and FOVs that are nonconvex polygons.

- The function ``m4opt.fov.footprint_healpix`` now has a default value for the
  ``target_coord`` argument.

2.4.0 (2026-03-06)
==================

- Add exposure time models for Vera C. Rubin Observatory and Zwicky Transient
  Facility.

- Update installation instructions for users of academic editions of CPLEX.

- Drop support for Python 3.12.

2.3.1 (2026-01-23)
==================

- Fix an issue where MILP optimization could terminate early before CPLEX had
  found a best bound. This was prone to happen after MIP restarts.

2.3.0 (2026-01-22)
==================

- When available memory is limited by the ``--memory`` option, spool CPLEX's
  node file to disk.

- If the solution is aborted because the best bound falls below the objective
  lower cutoff, then record the solution status as
  `aborted, lower cutoff reached`.

2.2.1 (2026-01-16)
==================

- Fix crash for sky maps with invalid pixels when using a fixed absolute
  magnitude.

2.2.0 (2026-01-15)
==================

- Add support to the scheduler for pointwise distance distributions but fixed
  absolute magnitude.

2.1.0 (2025-12-31)
==================

- Add the method ``m4opt.milp.Model.to_stream``.

- Add the method ``m4opt.utils.optimization.partition_graph_color``.

2.0.1 (2025-06-12)
==================

- Allow passing any options to METIS.

2.0.0 (2025-06-09)
==================

- Allow each mission to have one or several different sky grids.

- Add support for combining constraints using boolean operators
  (``lhs | rhs``, ``lhs & rhs``, ``~op``).

- The ``Mission.constraints`` property no longer accepts a list of constraints.
  To combine multiple constraints, use boolean operators.

- The ``Mission.detector`` property is now optional. Only adaptive exposure
  time observing strategies require it to be defined.

- Add Earth radiation belt constraint.

- Add two new missions: Vera C. Rubin Observatory and Zwicky Transient
  Facility.

- Add a mixed integer programming Traveling Salesman solver.

1.0.0 (2025-04-07)
==================

- Add citation file.

- Refactor obsever position classes to support both Earth-fixed and
  Earth-orbiting observers.

- Add basic positional astronomy constraints on right ascension, declination,
  altitutide, azimuth, and airmass.

- Add at-night constraint for Earth-fixed observers.

- Add logical constraints (and, or, not).

- Add an exact Traveling Salesman solver as a utility function.

- Move the DustExtinction class to the m4opt.synphot.extinction module
  to prepare for adding other sources of extinction (e.g., atmospheric).

- Add optional zoom inset to animation.

0.1.1 (2025-02-24)
==================

- Update PyPI long description. No functional changes in this release.

0.1.0 (2025-02-24)
==================

- First release.
