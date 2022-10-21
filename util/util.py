from typing import Sequence, TypeVar, Iterable

import itertools


def flatten_list(t):
    return [item for sublist in t for item in sublist]


T = TypeVar("T")


def get_only_element(l: Sequence[T]) -> T:
    if len(l) != 1:
        raise ValueError(f"Expected 1 element but only got {len(l)}. {l if len(l) < 10 else ''}".strip())
    return l[0]


def repeat_val(val: T, n: int) -> Iterable[T]:
    yield from (val for _ in range(n))


def inner_chain(*iterable_of_iterables: Iterable[Iterable]):
    return [
        itertools.chain(*vals)
        for vals in zip(*iterable_of_iterables)
    ]


