Testing
=======

|M4OPT| has extensive unit tests which are run automatically in `GitHub Actions <https://github.com/m4opt/m4opt/actions>`_. Code coverage is `reported on Codecov.io <https://app.codecov.io/gh/m4opt/m4opt>`_. You can also follow these instructions to run the unit tests locally, on your own computer:

1. Clone the |M4OPT| git repository onto your computer by running this command::

    git clone https://github.com/m4opt/m4opt.git

2. Enter the cloned repository directory::

    cd m4opt

3. Install |M4OPT| using :doc:`pip <pip:index>` in :ref:`editable mode <pip:editable-installs>` with the ``test`` :ref:`extra <python-packaging-guide:dependency-specifiers-extras>`::

    pip install -e .[test]

4. Run the unit tests::

    pytest
