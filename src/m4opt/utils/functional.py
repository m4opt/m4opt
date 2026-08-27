"""Functional programming utilities."""

from collections.abc import Callable, Iterable
from itertools import groupby
from operator import itemgetter
from typing import TypeVar

__all__ = ("apply", "groupby_unsorted")

Item = TypeVar("Item")
Key = TypeVar("Key")
first = itemgetter(0)
second = itemgetter(1)


def apply[*Ts, R](func: Callable[[*Ts], R], args: tuple[*Ts]) -> R:
    """Invoke a function, unpacking arguments.

    The function is called such that ``apply(foo, (bar, bat))`` is equivalent
    to ``foo(bar, bat)``.

    >>> from m4opt.utils.functional import apply
    >>> def add(a, b):
    ...     return a + b
    ...
    >>> args = (1, 2)
    >>> apply(add, args)
    3
    >>> add(*args)
    3
    """
    return func(*args)


def groupby_unsorted[Item, Key](
    iterable: Iterable[Item], key: Callable[[Item], Key]
) -> Iterable[tuple[Key, Iterable[Item]]]:
    """Group items like :obj:`itertools.groupby`, but without requiring the input to be sorted."""
    return (
        (key, map(second, values))
        for key, values in groupby(
            sorted(((key(item), item) for item in iterable), key=first), key=first
        )
    )
