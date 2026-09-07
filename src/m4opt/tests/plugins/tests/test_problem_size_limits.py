"""Tests for skipping problems that exceed the solver's licensed size."""

import pytest

CONFTEST = """
from m4opt.tests.plugins.problem_size_limits import (  # noqa: F401
    pytest_runtest_call,
    pytest_runtest_setup,
)
"""

SOURCE = '''
import pytest
from docplex.mp.utils import DOcplexLimitsExceeded


def too_big():
    raise DOcplexLimitsExceeded(10000, 10000)


@pytest.fixture
def solved():
    too_big()


def test_from_a_fixture(solved):
    pass


def test_from_the_body():
    too_big()


def test_unrelated_error_is_not_skipped():
    raise ValueError("something else")
'''


@pytest.fixture
def run(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(SOURCE)
    return pytester.runpytest()


def test_the_size_limit_becomes_a_skip(run):
    """A solve in a fixture is raised during setup, not during the call."""
    run.assert_outcomes(skipped=2, failed=1)


def test_an_unrelated_error_still_fails(run):
    """Only the size limit is turned into a skip."""
    run.stdout.fnmatch_lines(["*test_unrelated_error_is_not_skipped*"])
