from classify_text_plz.classifiers.stupid_classifiers import MostCommonClassModelMaker
from classify_text_plz.dataing import MyTextData, MyTextDataSplit, DataSplit


def test_most_common_simple():
    maker = MostCommonClassModelMaker()
    data = MyTextData(
        (MyTextDataSplit(
            DataSplit.TRAIN,
            *zip(*[
                ("a", True),
                ("b", False),
                ("a", True),
                ("a", False),
                ("c", False),
            ])
        ),)
    )
    model = maker.fit(data)
    pred = model.predict_text("a")
    assert pred.get_prob_of(True) == 2 / 5
    assert pred.get_prob_of(False) == 3 / 5
    pred = model.predict_text("b")
    assert pred.get_prob_of(True) == 2 / 5
    assert pred.get_prob_of(False) == 3 / 5


def test_most_common_partion():
    def partition(x):
        v = "a" in x
        print("PARTITION", x, v)
        return v
    maker = MostCommonClassModelMaker(
        partition_function=partition,
    )
    data = MyTextData(
        (MyTextDataSplit(
            DataSplit.TRAIN,
            *zip(*[
                ("a", True),
                ("b", False),
                ("a", True),
                ("a", False),
                ("c", False),
            ])
        ),)
    )
    model = maker.fit(data)
    print(model)
    pred = model.predict_text("a")
    assert pred.get_prob_of(True) == 2 / 3
    assert pred.get_prob_of(False) == 1 / 3
    pred = model.predict_text("b")
    assert pred.get_prob_of(True) == 0
    assert pred.get_prob_of(False) == 1
