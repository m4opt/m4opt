import pytest
from astropy import units as u
from click import UsageError

from ... import constraints
from ..schedule import _parse_constraint_spec


def test_parse_constraint_spec_with_quantities():
    constraint = _parse_constraint_spec(
        '{"type":"DeclinationConstraint","args":["-40 deg","90 deg"]}'
    )

    assert isinstance(constraint, constraints.DeclinationConstraint)
    assert constraint.min == -40 * u.deg
    assert constraint.max == 90 * u.deg


def test_parse_constraint_spec_with_typed_keyword_quantity():
    constraint = _parse_constraint_spec(
        '{"type":"MoonSeparationConstraint","kwargs":{"min":"20 deg"}}'
    )

    assert isinstance(constraint, constraints.MoonSeparationConstraint)
    assert constraint.min == 20 * u.deg


@pytest.mark.parametrize()
def test_parse_constraint_spec_with_nested_logical_constraint():
    constraint = _parse_constraint_spec(
        '{"type":"LogicalOrConstraint","args":['
        '{"type":"DeclinationConstraint","args":["-90 deg","-40 deg"]},'
        '{"type":"DeclinationConstraint","args":["40 deg","90 deg"]}]}'
    )

    assert isinstance(constraint, constraints.LogicalOrConstraint)
    assert len(constraint._operands) == 2
    assert all(
        isinstance(operand, constraints.DeclinationConstraint)
        for operand in constraint._operands
    )


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("not JSON", "invalid constraint JSON"),
        ('{"type":"not-a-constraint"}', "unknown constraint type"),
        (
            '{"type":"MoonSeparationConstraint","args":[20]}',
            "must be strings with units",
        ),
        ('{"type":"LogicalOrConstraint"}', "requires at least one constraint operand"),
    ],
)
def test_parse_invalid_constraint_spec(spec, message):
    with pytest.raises(UsageError, match=message):
        _parse_constraint_spec(spec)
