import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors, KNeighborsTransformer, KNeighborsClassifier, \
    KNeighborsRegressor

from classify_text_plz.classifiers.scikithelpers import preproc_for_bow, PlzTrainedScikitModel, \
    PlzScikitModelMaker
from classify_text_plz.dataing import MyTextData, DataSplit
from classify_text_plz.modeling import TextModelMaker, TextModelTrained
from sklearn.pipeline import Pipeline


class TfidfKnnModelMaker(PlzScikitModelMaker):
    def __init__(self, k=2, is_soft=False,):
        super().__init__(is_soft)
        self._name = "KNN"
        Module = KNeighborsClassifier if not is_soft else KNeighborsRegressor
        self._pipeline = Pipeline([
            ('vect', CountVectorizer(
                analyzer='word',
                tokenizer=nltk.tokenize.word_tokenize,
                lowercase=True,
                stop_words='english',
            )),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('nearest', Module(
                n_neighbors=k,
                metric='euclidean',
                n_jobs=-1,
                weights='distance'
            )),
        ])



