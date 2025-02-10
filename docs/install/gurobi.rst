.. highlight:: sh

Install Gurobi (not currently used)
===================================

Gurobi for Python is distributed as the lightweight `gurobipy`__ Python
package. It is installed automatically when you :doc:`install M4OPT using pip
<index>`.

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
--------------

1. Make sure that your computer is connected to your campus network or VPN.

2. In a Web browser, navigate to `Gurobi Academic Programs and Licenses`__.

__ https://www.gurobi.com/academia/academic-program-and-licenses/

3. Follow the instructions under the "Individual Academic Licenses" heading to
   download and install Gurobi and retrieve your ``gurobi.lic`` file using
   the ``grbgetkey`` command.

4. Once you have retrieved the ``gurobi.lic`` file on your computer, it
   does not need to be connected to your campus network or VPN to use Gurobi.

All others
----------

For commercial and government users, `Gurobi offers a variety of license
types`__. Contact `Gurobi sales`__ for pricing.

__ https://www.gurobi.com/products/licensing-options/
__ https://www.gurobi.com/products/purchase-gurobi/
