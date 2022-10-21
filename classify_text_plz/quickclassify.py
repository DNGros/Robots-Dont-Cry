from typing import Dict, List, Union, Tuple

from tokenizers.normalizers import Sequence

from classify_text_plz.classifiers.bow_logreg import BowModelLogRegressModel
from classify_text_plz.classifiers.deeplearn.bertbaseline import BertlikeModelMaker
from classify_text_plz.classifiers.fasttext_baseline import FastTextModelMaker
from classify_text_plz.classifiers.stupid_classifiers import MostCommonClassModelMaker, RandomGuessModelMaker
from classify_text_plz.classifiers.tfidf_nn import TfidfKnnModelMaker
from classify_text_plz.dataing import MyTextData, DataSplit, MyTextDataSplit
from classify_text_plz.evaling import PlzEvaluator, PlzTextMetric, EvalResult
from classify_text_plz.modeling import TextModelMaker


class QuickClassifyModelDefaults:
    def __init__(
        self,
        data: MyTextData,
        most_common: bool = True,
        random_guess: bool = True,
        bow_logistic_regression: bool = True,
        tfidf_knn: bool = True,
        bert: bool = True,
    ):
        self.model_makers = []
        if most_common:
            self.model_makers.append(
                (MostCommonClassModelMaker(), False)
             )
        if random_guess:
            self.model_makers.append(
                (RandomGuessModelMaker(), False)
            )
        if bow_logistic_regression:
            self.model_makers.append(
                (BowModelLogRegressModel(), True)
            )
        if tfidf_knn:
            self.model_makers.append(
                (TfidfKnnModelMaker(k=5, is_soft=data.is_soft_binary()), True)
            )
        if bert:
            self.model_makers.append((BertlikeModelMaker(
               'bert-base-uncased', epoch=5,
               gradient_accumulation_steps=5,
            ), True))

    def __iter__(self):
        return iter(self.model_makers)




def classify_this_plz(
    data: MyTextData,
    evaluator: PlzEvaluator = None,
    include_test_split: bool = False,
    model_makers: List[Union[Tuple[TextModelMaker, bool], TextModelMaker]] = None,
) -> Dict[str, EvalResult]:
    evaluator = PlzEvaluator() if evaluator is None else evaluator
    all_results = {}
    if model_makers is None:
        model_makers = QuickClassifyModelDefaults(data)
    for model_maker, dump_vals in [
        maker_dump if isinstance(maker_dump, tuple) else (maker_dump, False)
        for maker_dump in model_makers
    ]:
        model = model_maker.fit(data)
        print(f"#"*80)
        print(f"Evaluate model {model.get_model_name()}")
        eval_result = evaluator.print_eval(
            data,
            model,
            #splits=[DataSplit.TRAIN, DataSplit.VAL] + ([DataSplit.TEST] if include_test_split else []),
            dump_split_highest_loss=None if dump_vals else {},
            dump_split_lowest_loss=None if dump_vals else {},
        )
        all_results[model.get_model_name()] = eval_result
    return all_results