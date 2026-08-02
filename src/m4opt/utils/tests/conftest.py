from importlib import resources

import pytest

from . import sphinx_roots


@pytest.fixture(scope="session")
def rootdir():
    yield resources.files(sphinx_roots)
