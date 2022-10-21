from typing import Sequence, Iterable, TypeVar, Callable, List, Tuple, Any, Union
import math
import hashlib
import random
import string
import numpy as np
import pickle


T = TypeVar("T")


def id_generator(size=6, chars=string.ascii_uppercase + string.digits, seed=None):
    # ref https://stackoverflow.com/questions/2257441/
    if seed:
        rng = random.Random(seed)
        return ''.join(rng.choice(chars) for _ in range(size))
    else:
        return ''.join(random.choice(chars) for _ in range(size))


T = TypeVar("T")


def deterministic_hash(
    val: Any,
    seed: Union[str, int],
    digest_bytes: int = 8
) -> int:
    seed = str(seed)
    if not isinstance(val, str):
        byte_val = pickle.dumps(val) + str.encode(seed)
    else:
        byte_val = str.encode(val + "::" + seed)
    hash = hashlib.blake2b(
        byte_val,
        digest_size=digest_bytes
    )
    hash_int = int(hash.hexdigest(), 16)
    return hash_int


class DeterministicSplitter:
    """Used to perform a pseudorandom selection of a split for example as a
    function of the input
    """
    def __init__(
        self,
        split_weights: Sequence[float],
        seed: int = None
    ):
        weight_sum = sum(split_weights)
        self._splits_weights = [w / weight_sum for w in split_weights]
        if seed is None:
            self._seed_str = ""
        else:
            self._seed_str = id_generator(seed=seed)

    def get_split_from_example(
        self,
        x: str
    ) -> int:
        """
        Args:
            x: the value to partition
        :returns
            The partition index. It will between 0 to `(len(self.split_weights) - 1)`
        """
        DIGEST_BYTES = 8
        MAX_VAL = 2 ** (8 * DIGEST_BYTES)
        hash_int = deterministic_hash(x, self._seed_str, digest_bytes=DIGEST_BYTES)
        current_ceiling = 0
        for i, probability in enumerate(self._splits_weights):
            current_ceiling += probability * MAX_VAL
            if hash_int <= math.ceil(current_ceiling):
                return i
        raise RuntimeError(f"Unreachable code reached. Splits {self._splits_weights}")

    def split_items(
        self,
        items: Iterable[T],
        key: Callable[[T], str] = lambda v: v
    ) -> Tuple[List[T]]:
        out = tuple([] for _ in self._splits_weights)
        for item in items:
            out[self.get_split_from_example(key(item))].append(item)
        return out


def partition_list(
    elements,
    split_weights: Sequence[float],
    min_per_split: int = 1
) -> List[Sequence]:
    """Takes a list and cuts into parts at the proportions given by `split_weights`. Ordering
    is preserved and each part with have at least `min_per_split` elements"""
    if len(elements) < len(split_weights):
        raise ValueError()
    num_elements = len(elements)
    weight_sum = sum(split_weights)
    split_weights = [w / weight_sum for w in split_weights]
    split_counts_float = [num_elements * weight for weight in split_weights]
    split_counts_int = [max(round(v), min_per_split) for v in split_counts_float]
    diff = sum(split_counts_int) - num_elements
    if diff != 0:
        # Need to reconile the counts
        direction = -1 if diff > 0 else 1
        amount_cheated = [
            (should_be - actual, i)
            for i, (should_be, actual) in enumerate(zip(split_counts_float, split_counts_int))
            if should_be > min_per_split
        ]
        amount_cheated.sort(reverse=direction > 0)
        _, indexes = zip(*amount_cheated[:abs(diff)])
        for i in indexes:
            split_counts_int[i] += direction
    cum_sum = np.cumsum(split_counts_int)
    assert cum_sum[-1] == num_elements
    return [
        elements[cum - val: cum]
        for cum, val in zip(cum_sum, split_counts_int)
    ]


def duplicate_upto_set_prob_mass(
    num_dups: int,
    elements: Sequence[T],
    weights: Sequence[float],
    dup_mass: float
) -> Tuple[List[List[T]], Sequence[Tuple[T, float]]]:
    """Creates duplicates of values `num_dups` times such that the weights
    of those values have a probability of at at least `dup_mass`.
    If `dup_mass` is 0 or None then don't return anything"""
    duped_vals_with_mass = [[] for _ in range(num_dups)]
    if dup_mass == 0 or dup_mass is None:
        return duped_vals_with_mass, list(zip(elements, weights))
    assert 0 < dup_mass < 1
    weight_sum = sum(weights)
    norm_weights = [w / weight_sum for w in weights]
    most_probable = sorted(zip(elements, norm_weights), reverse=True, key=lambda x: x[1])
    cur_duped_mass = 0.0
    dup_index = 0
    for i, (val, weight) in enumerate(most_probable):
        for o in duped_vals_with_mass:
            o.append((val, weight))
        cur_duped_mass += weight
        if cur_duped_mass >= dup_mass:
            dup_index = i
            break
    rest_which_wasnt_duped = most_probable[dup_index+1:]
    return duped_vals_with_mass, rest_which_wasnt_duped


class WeirdSplitter:
    """A weird/fancy splitter of elements. It duplicates elements up to a certain
    probability mass, then it randomly splits the rest making sure at least one
    element appears uniquely in each split.

    If when the `duplicate_prob_mass` is 0,
    """
    def __init__(
        self,
        split_weights: Sequence[float],
        seed: int = None,
        duplicate_prob_mass: float = 0.0,
        min_unique_elements_per_split: int = 1
    ):
        weight_sum = sum(split_weights)
        self._splits_weights = [w / weight_sum for w in split_weights]
        assert 0 <= duplicate_prob_mass < 1.0
        self.duplicate_prob_mass = duplicate_prob_mass
        self.min_unique_elements_per_split = min_unique_elements_per_split
        if seed is None:
            self._seed_str = ""
        else:
            self._seed_str = id_generator(seed=seed)

    def split_items(
        self,
        vals: Sequence[T],
        base_weights: Sequence[float],
        key: Callable[[T], str] = lambda v: str(v),
    ) -> List[List[Tuple[T, float]]]:
        split_vals, rest = duplicate_upto_set_prob_mass(
            len(self._splits_weights), vals, base_weights, self.duplicate_prob_mass)
        sort_hashes = [
            deterministic_hash(key(val), self._seed_str)
            for val, weight in rest
        ]
        vals_sored = sorted(zip(sort_hashes, rest), key=lambda v: v[0])
        for split, rest_split in zip(split_vals, partition_list(
                vals_sored, self._splits_weights, self.min_unique_elements_per_split)):
            assert len(rest_split) > 0
            split.extend(val_and_weight for sort_hash, val_and_weight in rest_split)
        return split_vals
