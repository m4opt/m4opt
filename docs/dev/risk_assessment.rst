#################
Deploying |M4OPT|
#################

|M4OPT| is an observation planning toolkit. It can be deployed in much the same
way as one would use the `Astropy` Python library, i.e., ``import m4opt``.
However, since |M4OPT| will be deployed in planning operations, there are several
risks that users must be aware of.

1. Lack of Licensing for Gurobi and CPLEX

|M4OPT| utilizes mixed-integer linear programming in order to find and schedule
optimal observing plans across multiple observatories. It primarily uses
commercial libraries, specifically Gurobi and CPLEX. These require licenses
(or connection to a license server) for operation, although academic licenses are also
available.

|M4OPT| is also planned to support some non-commercial, open-source solvers. However,
these are not anticipated to perform nearly as well as the commercial
solvers.

2. Access failures due to lack of internet access

Since |M4OPT| uses :doc:`astropy:coordinates/index` to identify observatory and
target locations, users should know that
:meth:`astropy.coordinates.SkyCoord.from_name` and
:meth:`astropy.coordinates.EarthLocation.of_site` both require
:doc:`an internet connection to access <astropy:coordinates/remote_methods>`
remote data. If |M4OPT| must be deployed offline, users should consider saving
a list of :class:`~astropy.coordinates.SkyCoord` and
:class:`~astropy.coordinates.EarthLocation` corresponding to needed locations,
:ref:`as mentioned in the Astropy documentation <astropy:iers-working-offline>`.

Lack of internet access may also affect connection to a license server (see point 1).

Risk Assessment Matrix
----------------------
A risk assessment matrix is a tool used in systems engineering to aid in the
evaluation of project risks. It is used to identify risks, assess their
frequency (or likelihood), and evaluate their potential impact (e.g., damage
or service interruption).

In the matrix, the likelihood (on the vertical axis) and
consequence (on the horizontal axis) are measured from 1 - 5, with 5 being
the most likely (or most consequential). The cells in the matrix are shaded
red, yellow, or green: these correspond to high, medium, or low risk.

|M4OPT|'s risk assessment matrix is as follows:

.. table::
    :class: risktable

    +------------+---+---------+----------+---------+---------+---------+
    |            | 5 |         |          |         |         |         |
    +            +---+---------+----------+---------+---------+---------+
    |            | 4 |         |          |  `2`_   |         |         |
    +            +---+---------+----------+---------+---------+---------+
    | Likelihood | 3 |         |          |         |         |         |
    +            +---+---------+----------+---------+---------+---------+
    |            | 2 |         |    `1`_  |         |         |         |
    +            +---+---------+----------+---------+---------+---------+
    |            | 1 |         |          |         |         |         |
    +            +---+---------+----------+---------+---------+---------+
    |            |   |    1    |     2    |    3    |    4    |    5    |
    +------------+---+---------+----------+---------+---------+---------+
    |            |   |                 Consequences                     |
    +------------+---+---------+----------+---------+---------+---------+


List of Risks
-------------

ID: Example
^^^^^^^^^^^

Title: Example Title

Affinity: Code, External, Cost, Schedule, Project Requirements, Process, Resources

Description/Status: This is an example risk that is described here.

Mitigation: Plan to mitigate (if any)

Likelihood: Very Low, Low, Moderate, High, Very High

Consequence: Very Low, Low, Moderate, High, Very High

.. _1:

ID: 1
^^^^^

Title: Lack of Internet Access

Affinity: External

Description/Status: Some `astropy` commands require internet access, namely
:meth:`astropy.coordinates.SkyCoord.from_name` and
:meth:`astropy.coordinates.EarthLocation.of_site`. Depending on how targets and
observatory locations will be defined in the user code, this may impact |M4OPT|
usability.

Mitigation: Raise awareness and document wherever needed.

Likelihood: Low

Consequence: Low

.. _2 :

ID: 2
^^^^^

Title: Lack of Solver Licensing

Affinity: External

Description/Status: |M4OPT| has dependencies on several mixed-integer linear
programming solvers, specifically CPLEX and Gurobi. These libraries require
paid commercial licenses, though academic licenses are available. This will
impact program performance if user does not have access to
CPLEX or Gurobi.

Mitigation: Provide interface to open-source libraries. Document differences
in results and run-time between open-source MILP solvers, CPLEX, and Gurobi.

Likelihood: High

Consequence: Moderate
