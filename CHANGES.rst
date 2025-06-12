*******
Changes
*******

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
