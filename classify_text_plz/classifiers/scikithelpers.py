import nltk
from sklearn.pipeline import Pipeline

from classify_text_plz.dataing import MyTextData, DataSplit
from classify_text_plz.modeling import TextModelTrained, Prediction, TextModelMaker


def preproc_for_bow(text: str):
    text = " ".join(nltk.tokenize.word_tokenize(text))
    text = text.strip()
    return text


class PlzScikitModelMaker(TextModelMaker):
    def __init__(self, is_soft_binary):
        self._pipeline = None
        self._name = self.__class__.__name__
        self._is_soft_binary = is_soft_binary

    def fit(self, data: MyTextData) -> TextModelTrained:
        if not self._is_soft_binary:
            train_labels = list(
                data.get_split_data(DataSplit.TRAIN).get_labels_quantized())
            index_to_label = list(set(train_labels))
            label_to_index = {
                label: i
                for i, label in enumerate(index_to_label)
            }
        else:
            train_labels = list(data.get_split_data(DataSplit.TRAIN).get_labels())
            index_to_label = None
            label_to_index = None
        self._pipeline.fit(
            [
                #preproc_for_bow(txt)
                txt
                for txt in data.get_split_data(DataSplit.TRAIN).get_text()
            ],
            [
                label_to_index[label] if not self._is_soft_binary else label
                for label in train_labels
            ]
        )
        return PlzTrainedScikitModel(
            self._name, self._pipeline, index_to_label, self._is_soft_binary)


class PlzTrainedScikitModel(TextModelTrained):
    def __init__(self, name: str, model: Pipeline, index_to_label, is_soft_binary: bool):
        self._name, self._model, self._label_map = name, model, index_to_label
        self._is_soft = is_soft_binary

    def predict_text(self, text: str):
        if not self._is_soft:
            pred = self._model.predict_proba([text])[0]
            return Prediction(
                label_to_prob={
                    label: score
                    for label, score in zip(self._label_map, pred)
                },
                presorted=False,
                guarantee_all_classes=True
            )
        else:
            pred = self._model.predict([text])[0]
            return Prediction(
                label_to_prob={
                    True: pred,
                    False: 1 - pred,
                },
                presorted=False,
                guarantee_all_classes=True
            )

    def get_model_name(self) -> str:
        return self._name