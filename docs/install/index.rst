.. highlight:: sh

Installation
============

The recommended way to install |M4OPT| is using :doc:`pip:index`::

    $ pip install m4opt

.. rubric:: Optional: Third-Party Solvers

For the largest problems, |M4OPT| requires a commercial MIP solver: `IBM ILOG
CPLEX Optimization Studio`__ (just "CPLEX" for short) or `Gurobi Optimizer`__.
Both products are available for free for academic users (students, staff, and
faculty at accredited educational institutions).

If you are going to use the scheduling features of |M4OPT|, then you should
follow the instructions below to install CPLEX *or* Gurobi. If you do *not*
intend to use the scheduling features of |M4OPT|, then you may skip this step.

__ https://www.ibm.com/products/ilog-cplex-optimization-studio
__ https://www.gurobi.com

.. toctree::
   :maxdepth: 2

   cplex
   gurobi
   deploy
