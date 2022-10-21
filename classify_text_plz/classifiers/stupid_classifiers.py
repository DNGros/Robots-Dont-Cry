from collections import Counter
from operator import itemgetter
from typing import Dict, Callable, Hashable

from more_itertools import groupby_transform

from classify_text_plz.modeling import TextModelTrained, TextModelMaker, Prediction
from classify_text_plz.dataing import MyTextData, DataSplit, MyTextDataSplit
import statistics

from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
from classify_text_plz.util import normalize_dict_vals
import random


def _class_probs_by_partition(
    data: MyTextDataSplit,
    partition_function: Callable[[str], Hashable]
) -> Dict[Hashable, Dict[CLASS_LABEL_TYPE, float]]:
    return dict(groupby_transform(
        iterable=sorted(
            zip(
                map(
                    partition_function,
                    data.get_text(),
                ),
                data.get_labels_quantized(),
            ),
            key=itemgetter(0)
        ),
        keyfunc=lambda p_y: p_y[0],
        valuefunc=lambda p_y: p_y[1],
        reducefunc=lambda ys: normalize_dict_vals(Counter(ys))
    ))


class MostCommonClassModelMaker(TextModelMaker):
    """
    Returns the distribution of the labels.

    Also supports a partition_function. A different distribution
    is calculated for reach result of the partition_function called
    on the text. This is useful for applications such multitask
    learning where several tasks are all in the data, but want to
    find the most common label differently for each task.
    """
    def __init__(
        self,
        partition_function: Callable[[str], Hashable] = None,
    ):
        if partition_function is None:
            partition_function = lambda x: 0
        self._partition_function = partition_function

    def fit(
        self,
        data: MyTextData,
    ) -> TextModelTrained:
        return AlwaysMostCommonModelTrained(
            _class_probs_by_partition(
                data.get_split_data(DataSplit.TRAIN),
                self._partition_function,
            ),
            self._partition_function
        )


class AlwaysMostCommonModelTrained(TextModelTrained):
    def __init__(self, class_probs_by_partition, partition_function):
        self._class_probs_by_partition = class_probs_by_partition
        self._partition_function = partition_function

    def predict_text(self, text: str):
        return Prediction(
            self._class_probs_by_partition[self._partition_function(text)]
        )

    def __str__(self):
        return f"<Model Always Return '{self._class_probs_by_partition}'>"


class RandomGuessModelMaker(TextModelMaker):
    """
    Returns a hard choice according the label distribution.

    Also supports a partition_function. A different distribution
    is calculated for reach result of the partition_function called
    on the text. This is useful for applications such multitask
    learning where several tasks are all in the data, but want to
    find the most common label differently for each task.
    """
    def __init__(
            self,
            partition_function: Callable[[str], Hashable] = None,
    ):
        if partition_function is None:
            partition_function = lambda x: 0
        self._partition_function = partition_function

    def fit(self, data: MyTextData) -> TextModelTrained:
        return RandomGuessTrained(
            _class_probs_by_partition(
                data.get_split_data(DataSplit.TRAIN),
                self._partition_function,
            ),
            self._partition_function
        )


class RandomGuessTrained(TextModelTrained):
    def __init__(
        self,
        class_probs_by_partition,
        partition_function: Callable[[str], Hashable] = None,
    ):
        self._class_probs_by_partition = class_probs_by_partition
        self._partition_function = partition_function

    def predict_text(self, text: str):
        class_probs = self._class_probs_by_partition[
            self._partition_function(text)
        ]
        guess = random.choices(
            tuple(class_probs.keys()), weights=tuple(class_probs.values()), k=1)[0]
        probs = {
            label: 0.0 if label != guess else 1.0
            for label in class_probs.keys()
        }
        return Prediction(probs)

    def __str__(self):
        return f"<RandomGuess>"

