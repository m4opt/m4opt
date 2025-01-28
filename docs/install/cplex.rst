.. highlight:: sh

Install CPLEX
=============

CPLEX comes with a variety of tools and interfaces including a full-featured
integrated development environment (IDE). However, we require only two
components, the lightweight `cplex`__ and `docplex`__ Python packages.

__ https://pypi.org/project/cplex/
__ https://pypi.org/project/docplex/

Academic users
--------------

Academic users cannot use the versions of cplex and docplex that are installed
automatically when you :doc:`install M4OPT using pip <index>`. Instead,
they need to use the Python packages that come with the full academic software
distribution of IBM ILOG CPLEX Optimization Studio, following the instructions
below.

1. In a Web browser, navigate to the `IBM Academic Initiative Data Science`__
   site.

__ https://www.ibm.com/academic

2. Register or log in using your institutional email address (for example, one that ends in .edu). If you encounter issues with being recognized as part of an academic institution, you can refer to the guide on `creating an IBM Cloud account`__ for assistance.

__ https://github.com/academic-initiative/documentation/blob/main/academic-initiative/how-to/How-to-create-an-IBM-Cloud-account/readme.md

3. Navigate to
   :menuselection:`Data Science --> IBM ILOG CPLEX Optimization Studio`.

4. Follow the download and installation instructions.

   If installing on a remote system, be sure to copy the appropriate installer
   file (e.g. `cplex_studio2211.linux_x86_64.bin`) to that system. Then,
   update the permissions of the file to make it executable and run it
   with e.g. `./cplex_studio2211.linux_x86_64.bin`. The installation will
   proceed in the command line.

5. In the Python environment in which you have installed |M4OPT|, follow the
   instructions at `Does CPLEX Optimization Studio 22.1.1 support Python 3.11?
   <https://www.ibm.com/support/pages/does-cplex-optimization-studio-2211-support-python-311>`_
   to update your Python environment to use your academic license version of
   CPLEX.

All others
----------

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

2. Set the environment variables ``CPLEX_STUDIO_KEY`` and
   ``CPLEX_STUDIO_KEY_SERVER`` by running these commands, replacing
   ``xxxxxxxxxx`` with your API key::

       export CPLEX_STUDIO_KEY=xxxxxxxxxx
       export CPLEX_STUDIO_KEY_SERVER=https://scx-cos.docloud.ibm.com/cos/query/v1/apikeys

   Consider adding these commands to your login shell's profile script.
