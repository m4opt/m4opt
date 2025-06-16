from importlib import resources

import pytest

from . import sphinx_roots

pytest_plugins = ["sphinx.testing.fixtures"]


@pytest.fixture(scope="session")
def rootdir():
    yield resources.files(sphinx_roots)
