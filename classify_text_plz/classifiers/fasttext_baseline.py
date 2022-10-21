import statistics
import unicodedata
import nltk
from pathlib import Path

import fasttext
from fasttext.FastText import _FastText

from classify_text_plz.dataing import MyTextData, DataSplit, MyTextDataSplit
from classify_text_plz.modeling import TextModelMaker, TextModelTrained, Prediction
from classify_text_plz.proc import str_to_ascii
import tempfile

from classify_text_plz.util import strip_prefixes


def to_fast_text_file(data_split: MyTextDataSplit) -> Path:
    lines = []
    for text, label in data_split.get_text_and_labels():
        text = preproc_for_fasttext(text)
        assert "\n" not in text, text
        lines.append(f"__label__{label} {text}")
    file = tempfile.NamedTemporaryFile('w', delete=False)
    print(lines[:4])
    file.write("\n".join(lines))
    file.close()
    return Path(file.name)


def preproc_for_fasttext(text: str):
    text = " ".join(nltk.tokenize.word_tokenize(text))
    text = str_to_ascii(text)
    text = text.strip()
    return text


class FastTextModelMaker(TextModelMaker):
    def __init__(self, epoch: int = 10, wordNgrams: int = 3):
        self._epoch = epoch
        self._wordNgrams = wordNgrams

    def fit(self, data: MyTextData) -> TextModelTrained:
        fasttext_file = to_fast_text_file(data.get_split_data(DataSplit.TRAIN))
        #print(fasttext_file)
        #print(fasttext_file.read_text()[:10000])
        ft_model = fasttext.train_supervised(
            str(fasttext_file),
            epoch=self._epoch,
            wordNgrams=self._wordNgrams,
            autotunePredictions=True,
            #autotuneValidationFile=str(to_fast_text_file(data.get_split_data(DataSplit.VAL))),
            #autotuneDuration=int(0.5*60),
            dim=300,
        )
        return FastTextTrained(ft_model)


class FastTextTrained(TextModelTrained):
    def __init__(self, ft_model: _FastText):
        self._model = ft_model

    def predict_text(self, text: str):
        labels, prob = self._model.predict(
            preproc_for_fasttext(text), k=100)
        assert len(labels) == len(prob)
        labels = strip_prefixes(labels, "__label__")
        return Prediction({
            label: prob
            for label, prob in zip(labels, prob)
        })

    def get_model_name(self) -> str:
        return "FastText"

