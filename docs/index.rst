###################################################################
|M4OPT|: Multi-Mission Multi-Messenger Observation Planning Toolkit
###################################################################

|M4OPT| is an open-source toolkit for multi-facility scheduling of astrophysics
observing campaigns. It focuses on extremely rapid follow-up of gravitational
wave (GW) and neutrino events with heterogeneous networks of space and
ground-based observatories.

|M4OPT| uses the versatile mathematical framework of `mixed integer
programming`__ to model and solve complex observation scheduling problems.
Although |M4OPT| is open source, for the largest problems it can leverage two
industrial-strength commercial MIP solvers: `CPLEX`__ or `Gurobi`__. Both
solvers are available for free for academic users.

__ https://en.wikipedia.org/wiki/Integer_programming
__ https://www.ibm.com/products/ilog-cplex-optimization-studio
__ https://www.gurobi.com

|M4OPT| is designed from the `Astropy affiliated pacakge`__ template, and is
meant to follow those standards, including interoperability with the
`Astropy`__ ecosystem. It also complies with `NASA Procedural Requirements
(NPR) 7150`__ for `Class C software`__ and is suitable for non-safety-critical
ground software applications for `Class D NASA payloads`__.

__ https://www.astropy.org/affiliated/
__ https://www.astropy.org
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=7150&s=2C
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002C_&page_name=AppendixD
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=8705&s=4A

***************
Getting Started
***************

.. toctree::
   :maxdepth: 1

   install
   solvers
   scenarios/index

******************
User Documentation
******************

.. toctree::
   :maxdepth: 1

   constraints
   models

***********************
Developer Documentation
***********************

.. toctree::
   :maxdepth: 1

   changes
   contributing
   testing
   related

***************
Project Details
***************

.. toctree::
   :maxdepth: 1

   npr7150
   risk_assessment

*****
Index
*****

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _feedback@astropy.org: mailto:feedback@astropy.org
.. _affiliated packages: https://www.astropy.org/affiliated/
