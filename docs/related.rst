Related Work and Recommended Reading
====================================
|M4OPT| builds off of the work of several previous scheduling programs, software releases, and projects linked here.
Additionally, linked here are several tutorials, talks and other sources of documentation the developers found useful
when creating |M4OPT| and other MILP-based schedulers.

Previous Projects
-----------------
* `Astroplan Scheduling Software`_: Astroplan is an open-source python package designed as a toolbox to
  help plan and schedule observing runs. Astroplan does not make use of MILP scheduling, but serves as an inspiration of
  the type of general-use scheduling package we hope |M4OPT| can be.
* `gwemopt`_: Gwemopt is a graviational-wave follow-up scheduling software which uses a greedy algorithm. It was
  used to schedule Zwicky Transient Facility (ZTF) target of opportunity (ToO) follow-up during O3.
* `dorado-scheduling`_: Dorado is a proposed space mission for ultraviolet graviational-wave follow-up, for which an
  MILP-based ToO scheduler was developed. This scheduler makes use of IBM's CPLEX Optimization Studio.
* `MUSHROOMS`_: MUSHROOMS is a MILP-based gravitational-wave follow-up scheduler designed with ZTF in mind. It developed
  for an REU project and makes use of Gurobi for optimization.
* `Spike`_: Spike is an observation planning and scheduling software released by the Space Telescope Science Institute
  that makes use of Constraint Satisfaction Problems to produce schedules.
* `Zwicky Transient Facility (ZTF) Scheduler`_: ZTF currently implements a mixed-integer programming scheduler to
  produce nightly schedules with the goal of maximizing transient discovery rate. As with Dorado Scheduling and
  MUSHROOMS, lessons learnt when producing this algorithm will be applied during |M4OPT| development.

.. _`Astroplan Scheduling Software`: https://github.com/astropy/astroplan
.. _`Gwemopt`: https://github.com/mcoughlin/gwemopt
.. _`Dorado Scheduling`: https://github.com/nasa/dorado-scheduling
.. _`MUSHROOMS`: https://github.com/bparazin/MUSHROOMS
.. _`Spike`: https://www.stsci.edu/scientific-community/software/spike
.. _`Zwicky Transient Facility (ZTF) Scheduler`: https://arxiv.org/abs/1905.02209

Documentation and Tutorials
---------------------------

* `Gurobi Documentation`_: This is the extensive documentation for Gurobi, one of the two commercial optimizers used in
  this project.
* `Docplex Documentation`_: This the extensive documentation for Docplex, the Python decision optimization library
  produced by IBM/CPLEX. This includes both mathematical programming and constraint programming optimization.
* `YALMIP Tutorials`_: This is a very useful resource as an introduction to linear programming, and has a lot of useful
  information about translating common logical relationships and functions into integer linear programming terms.

.. _`Gurobi Documentation`: https://www.gurobi.com/documentation/9.1/refman/index.html
.. _`Docplex Documentation`: http://ibmdecisionoptimization.github.io/docplex-doc/index.html
.. _`YALMIP Tutorials`: https://yalmip.github.io/tutorial/logicprogramming

Conferences & Workshops
-----------------------
The following conferences and workshops all concern themselves with organizing a dialogue on planning and scheduling
research, either within space science, as in the case of IWPSS, or interdisciplinarily as in the case of SPARK and ICAPS

* `International Workshops on Planning and Scheduling for Space (IWPSS)`_
* `Scheduling and Planning Applications woRKshop (SPARK)`_
* `International Conference on Automated Planning and Scheduling (ICAPS)`_

.. _`International Workshops on Planning and Scheduling for Space (IWPSS)`: https://sites.google.com/view/iwpss/
.. _`Scheduling and Planning Applications woRKshop (SPARK)`: https://icaps21.icaps-conference.org/workshops/SPARK/
.. _`International Conference on Automated Planning and Scheduling (ICAPS)`: https://www.icaps-conference.org/
