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

1. In a Web browser, navigate to the `IBM SkillsBuild`__ site.

__ https://skillsbuild.org

2. Click the :guilabel:`Log in` or :guilabel:`Sign up` button in the top-right
   corner of the page. Select :guilabel:`College software downloads`. Register
   using your institutional email address (for example, one that ends in
   `.edu`).

   .. note::
      If you encounter issues with being recognized as part of an academic
      institution, you can refer to the guide on `creating an IBM Cloud account`__
      for assistance.

      __ https://github.com/academic-initiative/documentation/blob/main/academic-initiative/how-to/How-to-create-an-IBM-Cloud-account/readme.md

3. Navigate to
   :guilabel:`Data Science` in the left sidebar and then click
   :guilabel:`ILOG CPLEX Optimization Studio`.

4. Following the onscreen instructions, download and run the appropriate
   installer for your operating system.

   .. hint::
      If you are installing CPLEX on a remote Linux system, copy the installer
      file (e.g. :file:`cplex_studio2211.linux_x86_64.bin`) to that system.
      Launch the installer by running the command
      ``sh cplex_studio2211.linux_x86_64.bin``.

5. Make a note of where the installer placed ILOG CPLEX Optimization Studio
   (for example, :file:`/opt/ibm/ILOG/CPLEX_Studio2211`). In the Python
   environment in which you have installed |M4OPT|, run the following command
   to enable your full academic version, updating the path as appropriate for
   your system::

         $ docplex config --upgrade /opt/ibm/ILOG/CPLEX_Studio2211

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
