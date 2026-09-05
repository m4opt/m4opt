.. highlight:: sh

Install SCIP
============

`SCIP`__ is an open-source MIP solver distributed under the Apache 2.0 license,
so unlike CPLEX and Gurobi it carries no restrictions on where it may be run.
It is installed together with its Python interface, `PySCIPOpt`__, by::

    $ pip install "m4opt[scip]"

No license file or registration is needed.

__ https://www.scipopt.org
__ https://pypi.org/project/PySCIPOpt/

.. note::
    SCIP needs longer than CPLEX or Gurobi to reach a schedule of the same
    quality, and with a variable exposure time (:option:`--absmag-mean`
    together with :option:`--appmag-dist`) it covers appreciably less
    probability in the same budget. Prefer a commercial solver where its
    license permits, and reach for SCIP when deploying somewhere that a
    commercial license does not cover.

.. warning::
    SCIP does not tighten its dual bound on these models, so the gap it
    reports is meaningless and the schedule it returns carries no claim of
    optimality. The schedule itself is valid; only the bound is uninformative.
