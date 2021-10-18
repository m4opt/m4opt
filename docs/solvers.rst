Install third-party solvers
===========================

For the largest problems, |M4OPT| requires a commercial MIP solver: `IBM ILOG
CPLEX Optimization Studio`__ (just "CPLEX" for short) or `Gurobi`__. Both
products are available for free for academic users (students, staff, and
faculty at accredited educational institutions).

__ https://www.ibm.com/products/ilog-cplex-optimization-studio
__ https://www.gurobi.com

Install CPLEX
-------------

CPLEX comes with a variety of tools and interfaces including a full-featured
integrated development environment (IDE). However, we require only two
components, the lightweight `cplex`__ and `docplex`__ Python packages.

__ https://pypi.org/project/cplex/
__ https://pypi.org/project/docplex/

Academic users
~~~~~~~~~~~~~~

Academic users cannot use the versions of cplex and docplex that are installed
automatically when you :doc:`install M4OPT using pip <install>`. Instead, they
need to use the Python packages that come with the full academic software
distribution of IBM ILOG CPLEX Optimization Studio, following the instructions
below.

1. In a Web browser, navigate to the `IBM Academic Initiative Data Science`__
   site.

__ https://www.ibm.com/academic

2. Register or log in using your institutional email address (ending in .edu).

3. Scroll to the middle of the page and navigate to
   :menuselection:`Software --> IBM ILOG CPLEX Optimization Studio`.

4. Follow the download and installation instructions.

5. In the Python environment in which you have installed |M4OPT|, run this
   command::

        python /opt/ibm/CPLEX_Studio201/python/setup.py install

   Replace ``/opt/ibm/CPLEX_Studio201`` with the path where the software was
   installed. On macOS, this may be in the ``/Applications`` directory.

All others
~~~~~~~~~~

IBM offers a `variety of license options for CPLEX`__ and differentiates between
"development" and "production" use.

__ https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing

A particularly cost-effective and flexible option for non-production use is the
"Developer Subscription". It allows a single authorized user to install and run
an unlimited number of concurrent solves with an unlimited number of threads on
an unlimited number of machines.

If you purchase a Developer Subscription, then activating the full version of
CPLEX is particularly straightforward, because no software downloads are
required. Follow these two steps:

1. IBM will have sent you your CPLEX Studio API key by email. Find this key.

2. Set the environment variable ``CPLEX_STUDIO_KEY`` by running this command,
   replacing ``xxxxxxxxxx`` with your API key::

       export CPLEX_STUDIO_KEY=xxxxxxxxxx

   Consider adding this command to your login shell's profile script.

Install Gurobi
--------------

Gurobi for Python is distributed as the lightweight `gurobipy`__ Python
package. It is installed automatically when you :doc:`install M4OPT using pip
<install>`.

__ https://pypi.org/project/gurobipy/

No matter what kind of license that you have, to make gurobipy fully functional
you simply need to have your ``gurobi.lic`` license file present in your home
directory.

Depending on your license type, you may need additional command-line tools to
retrieve and manage your license file, such as the ``grbgetkey`` command
mentioned in the instructions below. These command-line tools are not included
in gurobipy, but are included in the full Gurobi Optimizer distribution.

**However, once your license file is set up, you no longer need the full
distribution. We recommend that you use the gurobipy Python package that was
automatically installed by pip, rather than the one that came with the Gurobi
Optimizer installer.**

Academic users
~~~~~~~~~~~~~~

1. Make sure that your computer is connected to your campus network or VPN.

2. In a Web browser, navigate to `Gurobi Academic Programs and Licenses`__.

__ https://www.gurobi.com/academia/academic-program-and-licenses/

3. Follow the instructions under the "Individual Academic Licenses" heading to
   download and install Gurobi and retrieve your ``gurobi.lic`` file using
   the ``grbgetkey`` command.

4. Once you have retrieved the ``gurobi.lic`` file on your computer, it
   does not need to be connected to your campus network or VPN to use Gurobi.

All others
~~~~~~~~~~

For commerical and government users, `Gurobi offers a variety of license
types`__. Contact `Gurobi sales`__ for pricing.

__ https://www.gurobi.com/products/licensing-options/
__ https://www.gurobi.com/products/purchase-gurobi/
