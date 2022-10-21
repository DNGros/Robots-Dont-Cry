import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from classify_text_plz.classifiers.scikithelpers import preproc_for_bow, PlzTrainedScikitModel, \
    PlzScikitModelMaker
from classify_text_plz.dataing import MyTextData, DataSplit
from classify_text_plz.modeling import TextModelMaker, TextModelTrained
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

class LemmaTokenizer:
    def __init__(self):
        import nltk
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class BowModelLogRegressModel(PlzScikitModelMaker):
    def __init__(
        self,
        balance_class_weight: bool = False,
    ):
        super().__init__(False)
        self._name = "BOWLogisticRegression"
        self._pipeline = Pipeline([
            ('vect', CountVectorizer(
                analyzer='word',
                #tokenizer=nltk.tokenize.word_tokenize,
                tokenizer=LemmaTokenizer(),
                lowercase=True,
                #stop_words='english',
            )),
            ('tfidf', TfidfTransformer()),
            ('clf', LogisticRegression(
                class_weight='balanced' if balance_class_weight else None)),
        ])


if __name__ == "__main__":
    print(LemmaTokenizer()("I love to eat a lovely tasty taco. I loved my tacos"))
    wnl = WordNetLemmatizer()
    print(wnl.lemmatize("bats"))
    print(wnl.lemmatize("running"))

