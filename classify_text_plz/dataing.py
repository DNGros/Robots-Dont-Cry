import random
from typing import Iterable, Union, Optional, Sequence, Callable, Tuple, Set
from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
from classify_text_plz.util import get_only_element
import enum


class DataSplit(enum.Enum):
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"

    @classmethod
    def from_maybe_string(cls, val: Union[str, 'DataSplit']) -> 'DataSplit':
        if isinstance(val, DataSplit):
            return val
        elif isinstance(val.lower(), str):
            return DataSplit(val)
        raise ValueError(val)

    @classmethod
    def from_maybe_string_but_no_fail(cls, val: Union[str, 'DataSplit']) -> 'DataSplit':
        if isinstance(val, DataSplit):
            return val
        try:
            val = DataSplit(val)
        except ValueError:
            pass
        return val


class MyTextData:
    def __init__(
        self,
        already_split_datas: Optional[Sequence[Union[
            'MyTextDataSplit'
        ]]] = None
    ):
        self._already_split_datas = already_split_datas
        self._splits = {
            split.split_kind: split
            for split in already_split_datas
        }
        assert all(type(split) is type(already_split_datas[0]) for split in already_split_datas)

    def get_all_splits(self) -> Iterable[Tuple[Union[str, DataSplit], 'MyTextDataSplit']]:
        yield from self._splits.items()

    def get_split_data(self, split: Union[DataSplit, str]) -> 'MyTextDataSplit':
        split = DataSplit.from_maybe_string_but_no_fail(split)
        if split not in self._splits:
            raise KeyError(f"Split {split} not recognized. Options in dataset are {list(self._splits.keys())}")
        return self._splits[split]

    def apply_func_to_text(self, func: Callable[[str], str]):
        assert isinstance(func("Test string"), str)
        for split in self._already_split_datas:
            split.apply_func_to_text(func)

    def is_continuous(self) -> bool:
        return self.is_soft_binary()

    def get_unique_labels(self) -> Set[CLASS_LABEL_TYPE]:
        return set(self.get_split_data(DataSplit.TRAIN).get_labels_quantized())

    def is_soft_binary(self) -> bool:
        return isinstance(self._already_split_datas[0], MyTextDataSplitSoftBinary)


class MyTextDataSplit:
    def __init__(
        self,
        split_kind: Union[DataSplit, str],
        text: Iterable[str],
        labels: Iterable[CLASS_LABEL_TYPE],
    ):
        self.split_kind = DataSplit.from_maybe_string_but_no_fail(split_kind)
        self._text = list(text)
        self._labels = list(labels)
        assert len(self._text) == len(self._labels)

    def get_text(self) -> Iterable[str]:
        yield from self._text

    def get_labels(self) -> Iterable[CLASS_LABEL_TYPE]:
        yield from self._labels

    def get_labels_quantized(self) -> Iterable[Union[int, bool, str]]:
        yield from self._labels

    def get_text_and_labels(self) -> Iterable[Tuple[str, CLASS_LABEL_TYPE]]:
        yield from zip(self.get_text(), self.get_labels())

    def apply_func_to_text(self, func: Callable[[str], str]):
        self._text = [func(txt) for txt in self._text]


class MyTextDataSplitSoftBinary(MyTextDataSplit):
    def __init__(
        self,
        split_kind: Union[DataSplit, str],
        text: Iterable[str],
        labels: Iterable[float],
    ):
        super().__init__(split_kind, text, labels)

    def get_labels_quantized(self) -> Iterable[bool]:
        yield from (
            label > 0.5
            for label in self._labels
        )


def create_my_data_splits(
    texts: Iterable[str],
    labels: Iterable[CLASS_LABEL_TYPE],
    train_val_test_probs: Tuple[float, float, float],
) -> Union[Tuple[MyTextDataSplit, MyTextDataSplit], Tuple[MyTextDataSplit, MyTextDataSplit, MyTextDataSplit]]:
    """Creates a MyTextDataSplit for each of the three splits."""
    assert sum(train_val_test_probs) == 1.0
    text_and_labels = list(zip(texts, labels))
    random.shuffle(text_and_labels)
    count = len(text_and_labels)
    texts, labels = zip(*text_and_labels)
    train, val, test = train_val_test_probs
    out = (
        MyTextDataSplit(DataSplit.TRAIN, texts[:int(count * train)], labels[:int(count * train)]),
        MyTextDataSplit(DataSplit.VAL, texts[int(count * train):int(count * (train + val))], labels[int(count * train):int(count * (train + val))]),
        MyTextDataSplit(DataSplit.TEST, texts[int(count * (train + val)):], labels[int(count * (train + val)):]),
    )
    if test == 0:
        out = out[:2]
    return out




