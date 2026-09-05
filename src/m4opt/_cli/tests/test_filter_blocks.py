import numpy as np
import pytest

from ...milp import Model
from ..schedule import add_filter_block_constraints

EXPTIME = 30.0
EXCHANGE = 110.0


@pytest.fixture
def solved():
    """Schedule four fields over three visits in the bandpass pattern g, r, g."""
    n_fields, visits = 4, 3
    filter_changes = [True, True]
    with Model() as model:
        field_vars = model.binary_vars(n_fields)
        time_vars = model.continuous_vars((n_fields, visits), ub=1e4)
        half_exptime = np.full(n_fields, 0.5 * EXPTIME)

        # Keep the fields from piling up on top of each other.
        i, j = np.triu_indices(n_fields, 1)
        for visit in range(visits):
            model.add_constraints_(
                model.abs(time_vars[i, visit] - time_vars[j, visit]) >= EXPTIME
            )

        add_filter_block_constraints(
            model, field_vars, time_vars, half_exptime, filter_changes, EXCHANGE
        )
        model.add_constraints_(field_vars >= 1)
        model.minimize(model.sum(time_vars.ravel()))
        solution = model.solve()
        assert solution is not None
        yield solution.get_values(time_vars)


def test_visits_are_grouped_into_blocks(solved):
    """Every visit of one index finishes before any visit of the next begins."""
    starts = solved - 0.5 * EXPTIME
    ends = solved + 0.5 * EXPTIME
    assert (ends[:, :-1].max(axis=0) <= starts[:, 1:].min(axis=0) + 1e-6).all()


def test_blocks_are_separated_by_the_exchange_time(solved):
    """The gap between blocks leaves room to exchange the filter."""
    gaps = solved[:, 1:].min(axis=0) - solved[:, :-1].max(axis=0) - EXPTIME
    assert (gaps >= EXCHANGE - 1e-6).all()


def test_only_observed_fields_constrain_the_blocks():
    """A field that is not observed does not stretch the blocks around it."""
    with Model() as model:
        field_vars = model.binary_vars(2)
        time_vars = model.continuous_vars((2, 2), ub=1e4)
        add_filter_block_constraints(
            model, field_vars, time_vars, np.zeros(2), [True], EXCHANGE
        )
        # The second field is dropped, so its times are free to interleave.
        model.add_constraint_(field_vars[0] >= 1)
        model.add_constraint_(field_vars[1] <= 0)
        model.add_constraint_(time_vars[1, 1] <= time_vars[0, 0])
        assert model.solve() is not None
