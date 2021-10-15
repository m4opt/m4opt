Requirements
============


.. req:: The software must support the following kinds of observers:
   :id: OBSERVER

    * :np:`(TERRESTRIAL) Fixed relative to the surface of the Earth`
    * :np:`(EARTH_SATELLITE) In an Earth orbit described by a TLE`


.. req:: The software must support the following kinds of constraints:
   :id: CONSTRAINT

    * :np:`(AIRMASS) Limits on airmass or sun elevation angle`
    * :np:`(ALTITUDE) Limits on the altitude angle above the horizon`


.. req:: Unit tests that exceed solver license limits must be skipped:
   :id: PROBLEM_SIZE_LIMITS

   * :np:`(CPLEX) For CPLEX`
   * :np:`(GUROBI) For Gurobi`


.. needtable::
   :columns: id;title;incoming as "Tested by"
   :style: table
   :types: req
