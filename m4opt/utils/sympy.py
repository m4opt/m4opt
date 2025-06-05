"""Symbolic algebra utilities for SymPy."""

from collections.abc import Collection

from sympy import Add, Basic, Expr

from .functional import groupby_unsorted

__all__ = ("collect_dependence",)


def collect_dependence(
    expr: Expr, symbols: Collection[Basic]
) -> dict[tuple[Basic, ...], Basic]:
    """
    Collect terms in an expression that depend on like combinations of symbols.

    Examples
    --------
    >>> from m4opt.utils.sympy import collect_dependence
    >>> from sympy.abc import a, b, c
    >>> from sympy import sin, Symbol
    >>> symbols = [a, b, c]
    >>> expr = a + b + c
    >>> collect_dependence(expr, symbols)
    {(c,): c, (b,): b, (a,): a}
    >>> expr = 42 + a * (sin(a) + sin(b)) + b * (b + c) + c
    >>> collect_dependence(expr, symbols)
    {(): 42, (c,): c, (b,): b**2, (b, c): b*c, (a,): a*sin(a), (a, b): a*sin(b)}
    """

    def depends_on(term: Basic):
        return tuple(term.has(symbol) for symbol in symbols)

    return {
        tuple(symbol for keep, symbol in zip(key, symbols) if keep): Add(*terms)
        for key, terms in groupby_unsorted(
            expr.expand().as_ordered_terms(), key=depends_on
        )
    }
