.. highlight:: sh

Installation
============

.. important::
    M4OPT currently supports Python 3.12, 3.13, or 3.14.

The recommended way to install |M4OPT| is using :doc:`pip:index`::

    $ pip install m4opt

.. rubric:: Optional: Third-Party Solvers

For the largest problems, |M4OPT| needs a MIP solver: `IBM ILOG CPLEX
Optimization Studio`__ (just "CPLEX" for short), `Gurobi Optimizer`__, or
`SCIP`__. CPLEX and Gurobi are commercial products, both available for free to
academic users (students, staff, and faculty at accredited educational
institutions), and both find better schedules than SCIP. SCIP is open source
under the Apache 2.0 license, which makes it the option to reach for when
deploying somewhere that an academic license does not cover.

If you are going to use the scheduling features of |M4OPT|, then you should
follow the instructions below to install one of them. If you do *not* intend to
use the scheduling features of |M4OPT|, then you may skip this step.

__ https://www.ibm.com/products/ilog-cplex-optimization-studio
__ https://www.gurobi.com
__ https://www.scipopt.org

.. toctree::
   :maxdepth: 2

   cplex
   gurobi
   scip
   deploy
