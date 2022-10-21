from typing import Sequence, TypeVar, Iterable, Dict, Union, List

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


K = TypeVar("K")
V = TypeVar("V")


def sort_dict_by_values(d: Dict[K, V], reverse=True) -> Dict[K, V]:
    return {
        k: v
        for v, k in sorted(
            ((v, k) for k, v in d.items()),
            reverse=reverse
        )
    }


def normalize_dict_vals(d: Dict[K, Union[float, int]]) -> Dict[K, float]:
    val_sum = sum(d.values())
    return {
        k: v / val_sum
        for k, v in d.items()
    }


def strip_prefixes(strings: Iterable[str], prefix: str) -> Iterable[str]:
    prefix_len = len(prefix)
    for s in strings:
        assert s.startswith(prefix)
        yield s[prefix_len:]
