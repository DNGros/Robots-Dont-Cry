import collections
import json


class FrozenDict(collections.Mapping):
    """
    from https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
    """
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = hash(tuple(sorted(self._d.items())))

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        return self._hash