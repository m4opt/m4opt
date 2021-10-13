|M4OPT|: Multi-Mission Multi-Messenger Observation Planning Toolkit
===================================================================

|M4OPT| is an open-source toolkit for multi-facility scheduling of astrophysics
observing campaigns. It focuses on extremely rapid follow-up of gravitational
wave (GW) and neutrino events with heterogeneous networks of space and
ground-based observatories.

|M4OPT| uses the versatile mathematical framework of `mixed integer
programming`_ (MIP) to model and solve complex observation scheduling problems.
Although |M4OPT| is open source, for the largest problems it can leverage two
industrial-strength commercial MIP solvers: `CPLEX`_ or `Gurobi`_. Both solvers
are available for free for academic users.

|M4OPT| is designed to be an `Astropy affiliated pacakge`_ to interoperate with
the `Astropy`_ ecosystem. It also complies with `NASA Procedural Requirements
(NPR) 7150 <NPR 7150.2C>`_ for `Class C software <NPR 7150.2C software
classifications>`_ and is suitable for non-safety-critical ground software
applications for `Class D NASA payloads <NPR 8705.4A>`_.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   install

.. _`mixed integer programming`: https://en.wikipedia.org/wiki/Integer_programming
.. _`CPLEX`: https://www.ibm.com/products/ilog-cplex-optimization-studio
.. _`Gurobi`: https://www.gurobi.com
.. _`Astropy affiliated pacakge`: https://www.astropy.org/affiliated/
.. _`Astropy`: https://www.astropy.org
.. _`NPR 7150.2C`: https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=7150&s=2C
.. _`NPR 7150.2C software classifications`: https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002C_&page_name=AppendixD
.. _`NPR 8705.4A`: https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=8705&s=4A