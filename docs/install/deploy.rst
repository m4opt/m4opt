###########################
Production Deployment Notes
###########################

Review the following deployment notes if you are planning to use |M4OPT| in any of the following production applications:

- **Operations**: In a mission operations center or observatory control room for real time, automated planning of observations
- **HPC Cluster**: In a high performance computing cluster for large-scale simulations
- **Cloud**: In a cloud instance or serverless container as part of a web application

Risk Matrix
-----------

We have categorized these deployment notes using a *risk matrix*, a visualization used in systems engineering to aid in the evaluation of project risks. In the matrix, the likelihood (on the vertical axis) and consequence (on the horizontal axis) are measured from very low to very high. The cells in the matrix are shaded red, yellow, or green: these correspond to high, medium, or low risk.

.. table::
    :class: risktable

    +-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                               | **Consequence**                                                                                                                                               |
    +-------------------+-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                               | Very Low                      | Low                           | Moderate                      | High                          | Very High                     |
    +-------------------+-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    | **Likelihood**    | Very High |                               |                               |                               |                               |                               |
    |                   +-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                   | High      |                               |                               |                               |                               |                               |
    |                   +-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                   | Moderate  |                               |                               |                               |                               |                               |
    |                   +-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                   | Low       |                               |                               |                               |                               | :ref:`cplex-memory`           |
    |                   +-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+
    |                   | Very Low  |                               |                               | :ref:`astropy-data`           |                               | :ref:`cplex-license`          |
    +-------------------+-----------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+-------------------------------+

Deployment Notes
----------------

.. _`astropy-data`:

Astropy Data Sources
^^^^^^^^^^^^^^^^^^^^

In normal use, |M4OPT| downloads and caches a variety of Astropy-related data sources that may be time-consuming to download or may be unavailable if your network connection is down:

- Well-known observatory locations for :meth:`astropy.coordinates.SkyCoord.from_name`
- Dust map for :obj:`m4opt.synphot.DustExtinction`
- Precise Earth orientation data for :class:`~astropy.coordinates.SkyCoord` :class:`~astropy.coordinates.EarthLocation` transformations (see :ref:`Astropy documentation on working offline <astropy:iers-working-offline>`)

.. rubric:: Mitigation

Run `m4opt prime <../guide/cli.html#m4opt-prime>`_ once before deployment to download and cache data sources. Ensure that you have a reliable Internet connection.

.. _`cplex-license`:

CPLEX License
^^^^^^^^^^^^^

Free versions of CPLEX have a limit on problem size. A full, unlimited problem size, version of CPLEX is required. If the CPLEX license is not correctly configured, then all invocations of the scheduler will fail.

.. rubric:: Mitigation

Follow the :doc:`insructions to install CPLEX </install/cplex>`. If you are using a "developer subscription" or "download-and-go" style license, then ensure that you have a reliable connection to the Internet for CPLEX to reach IBM's license server.

.. _`cplex-memory`:

CPLEX Memory
^^^^^^^^^^^^

CPLEX's memory usage grows as it explores potential solutions. By default, there is no limit on how much memory CPLEX may use. For challenging problems, CPLEX may exhaust available memory.

.. rubric:: Mitigation

Make sure that you reserve at least 8 GiB, and preferably 16 GiB or more, for running |M4OPT|. Set the ``--memory`` command-line option for the `m4opt schedule <../guide/cli.html#m4opt-prime>`_ command to at least 4 GiB less than the maximum amount of memory that you want it to use.
