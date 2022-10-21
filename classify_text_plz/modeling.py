from dataclasses import dataclass
from typing import Dict

from classify_text_plz.dataing import MyTextData
import abc

from classify_text_plz.typehelpers import CLASS_LABEL_TYPE
from classify_text_plz.util import sort_dict_by_values


class Prediction:
    def __init__(
        self,
        label_to_prob: Dict[CLASS_LABEL_TYPE, float],
        presorted: bool = False,
        guarantee_all_classes: bool = False,
    ):
        if not presorted:
            label_to_prob = sort_dict_by_values(label_to_prob, reverse=True)
        self._label_to_prob = label_to_prob
        self._guarantee_all_classes = guarantee_all_classes

    def most_probable_label(self) -> CLASS_LABEL_TYPE:
        return next(iter(self._label_to_prob))

    def get_prob_of(self, label: CLASS_LABEL_TYPE) -> float:
        if not self._guarantee_all_classes:
            return self._label_to_prob.get(label, 0)
        else:
            if label not in self._label_to_prob:
                raise ValueError(f"Label {label} not in prediction. {self._label_to_prob}")
            return self._label_to_prob[label]


class TextModelTrained(abc.ABC):
    @abc.abstractmethod
    def predict_text(self, text: str) -> Prediction:
        pass

    def get_model_name(self) -> str:
        return self.__class__.__name__


class TextModelMaker:
    def fit(self, data: MyTextData) -> TextModelTrained:
        pass