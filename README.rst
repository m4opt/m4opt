Multi-Mission Multi-Messenger Observation Planning Toolkit
----------------------------------------------------------

.. image:: https://img.shields.io/pypi/v/m4opt
    :target: https://pypi.org/project/m4opt/
    :alt: Python Package Index status
.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge
.. image:: https://codecov.io/gh/m4opt/m4opt/branch/main/graph/badge.svg?token=L837JHNTUV
    :target: https://codecov.io/gh/m4opt/m4opt
    :alt: Code coverage status
.. image:: https://readthedocs.org/projects/m4opt/badge/?version=latest
    :target: https://m4opt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://m4opt.readthedocs.io/en/latest/_images/example.gif
    :alt: Visualization of an example observing plan for UVEX generated M4OPT

M4OPT is an open-source toolkit for multi-facility scheduling of astrophysics
observing campaigns. It focuses on extremely rapid follow-up of gravitational
wave (GW) and neutrino events with heterogeneous networks of space and
ground-based observatories.

M4OPT uses the versatile mathematical framework of `mixed integer
programming`__ to model and solve complex observation scheduling problems.
Although M4OPT is open source, for the largest problems it can leverage two
industrial-strength commercial MIP solvers: `CPLEX`__ or `Gurobi`__. Both
solvers are available for free for academic users.

__ https://en.wikipedia.org/wiki/Integer_programming
__ https://www.ibm.com/products/ilog-cplex-optimization-studio
__ https://www.gurobi.com

M4OPT is designed from the `Astropy affiliated package`__ template, and is
meant to follow those standards, including interoperability with the
`Astropy`__ ecosystem. It also complies with `NASA Procedural Requirements
(NPR) 7150`__ for `Class C software`__ and is suitable for non-safety-critical
ground software applications for `Class D NASA payloads`__.

__ https://www.astropy.org/affiliated/
__ https://www.astropy.org
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=7150&s=2C
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?Internal_ID=N_PR_7150_002C_&page_name=AppendixD
__ https://nodis3.gsfc.nasa.gov/displayDir.cfm?t=NPR&c=8705&s=4A

Features
--------

*   **Global**: jointly and globally solves the problems of tiling (the set of
    telescope boresight orientations and roll angles) and the scheduling (which
    tile is observed at what time), rather than solving each sub-problem one at
    a time
*   **Optimal**: generally solves all the way to optimality, rather than
    finding merely a "good enough" solution
*   **Fast**: solve an entire orbit in about 5 minutes
*   **General**: does not depend on heuristics of any kind
*   **Flexible**: problem is formulated in the versatile framework of
    `mixed integer programming <https://en.wikipedia.org/wiki/Integer_programming>`_

License
-------

This project is Copyright (c) M4OPT Developers and licensed under
the terms of the BSD 3-Clause license. This package is based upon
the `Astropy package template <https://github.com/astropy/package-template>`_
which is licensed under the BSD 3-clause license. See the licenses folder for
more information.

How to Cite
-----------

If you use M4OPT in your research, then please cite the following paper:

   Singer, L. P., Criswell, A. W., Leggio, S. C., et al. (2025). Optimal Follow-Up of Gravitational-Wave Events with the UltraViolet EXplorer (UVEX) (Version 1). https://doi.org/10.48550/ARXIV.2502.17560

Contributing
------------

We love contributions! m4opt is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
m4opt based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
