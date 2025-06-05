"""Functional programming utilities."""

from collections.abc import Callable, Iterable
from itertools import groupby
from operator import itemgetter
from typing import TypeVar

__all__ = ("groupby_unsorted",)

Item = TypeVar("Item")
Key = TypeVar("Key")
first = itemgetter(0)
second = itemgetter(1)


def groupby_unsorted(
    iterable: Iterable[Item], key: Callable[[Item], Key]
) -> Iterable[tuple[Key, Iterable[Item]]]:
    """Group items like :obj:`itertools.groupby`, but without requiring the input to be sorted."""
    return (
        (key, map(second, values))
        for key, values in groupby(
            sorted(((key(item), item) for item in iterable), key=first), key=first
        )
    )
