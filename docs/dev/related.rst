Related Work and Recommended Reading
====================================
|M4OPT| builds on several related software packages and products.
Additionally, linked here are several tutorials, talks and other sources of
documentation the developers found useful when creating |M4OPT| and other
MILP-based schedulers.

Related Projects
-----------------
* `Astroplan Scheduling Software`_: Astroplan is an open-source python
  package designed as a toolbox to help plan and schedule observing runs.
  Astroplan does not make use of MILP scheduling, but serves as an inspiration
  of the type of general-use scheduling package we hope |M4OPT| can be.
* `gwemopt`_: Gwemopt is a graviational-wave follow-up scheduling software
  which uses a greedy algorithm. It was used to schedule Zwicky Transient
  Facility (ZTF) target of opportunity (ToO) follow-up during O3.
* `dorado-scheduling`_: Dorado is a proposed space mission for ultraviolet
  graviational-wave follow-up, for which an MILP-based ToO scheduler was
  developed. This scheduler makes use of IBM's CPLEX Optimization Studio.
* `MUSHROOMS`_: MUSHROOMS is a MILP-based gravitational-wave follow-up
  scheduler designed with ZTF in mind. It was developed for an REU project and
  makes use of Gurobi for optimization.
* `Spike`_: Spike is an observation planning and scheduling software released
  by the Space Telescope Science Institute that makes use of Constraint
  Satisfaction Problems to produce schedules.
* `Zwicky Transient Facility (ZTF) Scheduler`_: ZTF currently implements a
  mixed-integer programming scheduler to produce nightly schedules with the
  goal of maximizing transient discovery rate. As with Dorado Scheduling and
  MUSHROOMS, lessons learnt when producing this algorithm will be applied
  during |M4OPT| development.
* `Las Cumbres Observatory Adaptive Scheduler`_: MILP scheduling of heterogeneous telescope networks, using Google OR Tools or Gurobi.

.. _`Astroplan Scheduling Software`: https://github.com/astropy/astroplan
.. _`Gwemopt`: https://github.com/mcoughlin/gwemopt
.. _`dorado-scheduling`: https://github.com/nasa/dorado-scheduling
.. _`MUSHROOMS`: https://github.com/bparazin/MUSHROOMS
.. _`Spike`: https://www.stsci.edu/scientific-community/software/spike
.. _`Zwicky Transient Facility (ZTF) Scheduler`: https://arxiv.org/abs/1905.02209
.. _`Las Cumbres Observatory Adaptive Scheduler`: https://observatorycontrolsystem.github.io/components/adaptive_scheduler/

Documentation
-------------

* `Gurobi Documentation`_: This is the extensive documentation for Gurobi,
  one of the two commercial optimizers used in this project.
* `Docplex Documentation`_: This the extensive documentation for Docplex, the
  Python decision optimization library produced by IBM/CPLEX. This includes
  both mathematical programming and constraint programming optimization.

.. _`Gurobi Documentation`: https://www.gurobi.com/documentation/9.1/refman/index.html
.. _`Docplex Documentation`: http://ibmdecisionoptimization.github.io/docplex-doc/index.html

Mixed Integer Programming References
------------------------------------

* `YALMIP Tutorials`_: This website is a very useful resource as an
  introduction to linear programming, and has a lot of useful information about
  translating common logical relationships and functions into integer linear
  programming terms.
* `Gurobi Python Examples`_: Gurobi maintains a large database of jupyter
  notebook examples of using MIP techniques to solve many common problems. This
  is very useful to see typical problem formulations, and more advanced
  tutorials give useful working example of more complicated concepts.

.. _`YALMIP Tutorials`: https://yalmip.github.io/tutorial/logicprogramming
.. _`Gurobi Python Examples`: https://www.gurobi.com/resource/modeling-examples-using-the-gurobi-python-api-in-jupyter-notebook/

Conferences & Workshops
-----------------------
The following conferences and workshops all concern themselves with organizing
a dialogue on planning and scheduling research, either within space science, as
in the case of IWPSS, or interdisciplinarily as in the case of SPARK and ICAPS

* `International Workshops on Planning and Scheduling for Space (IWPSS)`_
* `Scheduling and Planning Applications woRKshop (SPARK)`_
* `International Conference on Automated Planning and Scheduling (ICAPS)`_

.. _`International Workshops on Planning and Scheduling for Space (IWPSS)`: https://sites.google.com/view/iwpss/
.. _`Scheduling and Planning Applications woRKshop (SPARK)`: https://icaps21.icaps-conference.org/workshops/SPARK/
.. _`International Conference on Automated Planning and Scheduling (ICAPS)`: https://icaps21.icaps-conference.org/
