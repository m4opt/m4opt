from functools import partial
from pathlib import Path

import pytest
from pytest_regressions.file_regression import FileRegressionFixture


@pytest.fixture
def generated_file(
    request: pytest.FixtureRequest, file_regression: FileRegressionFixture
):
    """
    Test reproducibility of a generated data file.

    This fixture checks that a string matches a generated data file.
    The data file is placed in the current directory and has a name that is
    inferred from the name of the test function. For example, if the name of
    the test is ``test_hello_world_dot_txt``, then the filename is
    ``hello_world.txt``.

    Examples
    --------

    ::
        def test_hello_world_dot_txt():
            return "Hello world"
    """
    stem, _, suffix = request.node.name.removeprefix("test_").rpartition("_dot_")
    path = (Path(request.path).parent / stem).with_suffix(f".{suffix}")
    return partial(file_regression.check, fullpath=path)
